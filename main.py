import wandb, click
import torch, os, sys, gc
import pandas as pd
from diffusers import AutoencoderKL

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from lib.generate_decode import generate, decode_vae
from lib.registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  cacher_registry, 
  data_registry, 
  metric_registry
)
# fill solver/scheduler/cacher/quantizer registries
import_dir(os.path.join(ROOT, 'lib', 'solvers'))
import_dir(os.path.join(ROOT, 'lib', 'schedulers'))
import_dir(os.path.join(ROOT, 'lib', 'cachers'))
# import_dir(os.path.join(ROOT, 'lib', 'quantizers'))
import_dir(os.path.join(ROOT, 'lib', 'datas'))
import_dir(os.path.join(ROOT, 'lib', 'metrics'))

def construct_pipeline(solver, scheduler, cacher_quantizer, is_optimize=False, add_vae=True):
  '''
  - constructs pipeline from components
  - torch.compile works after `torch compile !cp /usr/include/crypt.h /home/ekneudachina/.conda/envs/philurame_venv/include/python3.9/`
  - you can also optimize vae with `pipe.vae.decode = torch.compile(pipe.vae.decode, mode='reduce-overhead', fullgraph=True)`
  '''
  # fetch pipe, solver, scheduler classes
  PipeClass = cacher_registry[cacher_quantizer]
  SolverClass = solver_registry[solver]
  SchedulerClassMixin = scheduler_registry[scheduler]

  # load pipe
  pipe = PipeClass.from_pretrained()

  # construct and initialize solver-scheduler class
  class SolverSchedulerConstructor(SolverClass, SchedulerClassMixin):
    def set_timesteps(self, *args, **kwargs):
      SchedulerClassMixin.set_timesteps(self, *args, **kwargs)
  pipe.scheduler = SolverSchedulerConstructor.from_config()

  # some optimizations
  pipe = pipe.to('cuda')
  pipe.set_progress_bar_config(disable=True)  
  if is_optimize:
    pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead', fullgraph=True)
  if add_vae:
    pipe.vae = AutoencoderKL.from_pretrained(
      'madebyollin/sdxl-vae-fp16-fix',
      use_safetensors=True,
      torch_dtype=torch.float16,
    ).to('cuda')
  return pipe

def calc_metrics(**kwargs):
  '''
  - calculate metrics by looping over all metrics in metrics_registry
  '''
  res_metrics = {}
  for metric_name in metric_registry:
    metricInstance = metric_registry[metric_name](**kwargs)
    if hasattr(metricInstance, 'to'):
      metricInstance = metricInstance.to('cpu')
    res_metrics[metric_name] = metricInstance()

    # try to free memory
    del metricInstance
    gc.collect()
    torch.cuda.empty_cache()
  return res_metrics

# check if metrics are already calculated
def is_already_calculated(save_path, csv_path):
  cacher_quantizer, dataset_nfe, solver_scheduler = save_path.split('/')[-3:]
  dataset, nfe = dataset_nfe.split('_')
  nfe = int(nfe)
  solver, scheduler = solver_scheduler.replace('.pt', '').split('_')
  if not os.path.exists(csv_path): return False
  df_already_calculated = pd.read_csv(csv_path, index_col=0)
  return df_already_calculated.query(
    "dataset == @dataset and \
    solver == @solver and \
    scheduler == @scheduler and \
    cacher_quantizer == @cacher_quantizer and \
    nfe == @nfe"
    ).shape[0] > 0

##########################################################################################
# MAIN
##########################################################################################

@click.command()
@click.option('--key', type=str, required=True, help='wandb key')
@click.option('--max_samples', type=int, required=True, help='max num of images to generate')
@click.option('--dataset', type=str, required=True, help='PARTI|COCO')
@click.option('--generate', type=int, required=True, help='generate or calc metrics')
@click.option('--nfe', type=int, required=True, help='num inference steps')
@click.option('--solver', type=str, required=True, help='supported methods are in lib/solvers')
@click.option('--scheduler', type=str, required=True, help='supported methods are in lib/schedulers')
@click.option('--cacher_quantizer', type=str, required=True, help='ssupported methods are in lib/cachers and lib/quantizers')
def main(**kwargs):
  key = kwargs['key']
  max_samples = kwargs['max_samples'] 
  dataset = kwargs['dataset']
  is_generate = kwargs['generate']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  cacher_quantizer = kwargs['cacher_quantizer']

  data_path = os.path.join(ROOT, 'DATA')
  save_path = os.path.join(data_path, cacher_quantizer, f'{dataset}_{nfe}', f'{solver}_{scheduler}.pt')

  print('\n'+'#'*50, 
        *[f'{k}={v}' for k, v in kwargs.items()],
        f'{save_path=}',
        '#'*50+'\n', sep='\n')
  sys.stdout.flush()

  assert 0 < max_samples <= 10000
  assert isinstance(nfe, int) and nfe > 0
  assert bool(is_generate) != os.path.exists(save_path)
  assert bool(is_generate) or not is_already_calculated(save_path, csv_path=os.path.join(data_path, f'METRICS_{"SOLVERS" if cacher_quantizer=="NONE" else "CACHERS"}.csv'))

  data = data_registry[dataset](data_path, max_samples)
  pipe = construct_pipeline(
    solver, scheduler, cacher_quantizer, 
    is_optimize=is_generate and cacher_quantizer=='NONE', 
    add_vae=not is_generate
  )

  if is_generate: 
    ########################################
    # GENERAtE
    ########################################

    # generate latents
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gen_latents = generate(pipe, data.anns, nfe)
    torch.save(gen_latents, save_path)
  
  else:
    ########################################
    # METRICS
    ########################################

    # decode to images first
    is_already_decoded = torch.load(save_path, mmap=True).shape[1:] == (3, 1024, 1024)
    if is_already_decoded:
      gen_imgs = torch.load(save_path, map_location='cpu')
    else: 
      gen_latents = torch.load(save_path, map_location='cpu')
      gen_imgs = decode_vae(pipe, gen_latents)
      del gen_latents
    
    gc.collect()
    torch.cuda.empty_cache()

    # optimal LPIPS is not supported yet, so path_ddim is not needed actually
    path_ddim = os.path.join(data_path, cacher_quantizer, f'{dataset}_{nfe}', f'DDIM_{scheduler}.pt')
    metrics = calc_metrics(
      path_ddim=path_ddim, 
      fake_imgs=gen_imgs, 
      real_imgs=data.imgs, 
      real_anns=data.anns,
      pipe=pipe, 
      nfe=nfe
      )

    ########################################
    # LOG
    ########################################

    # choose your wandb project
    project_name = 'SDXL_METRICS_' + ('CACHERS' if cacher_quantizer != 'NONE' else 'SOLVERS')
    wandb.login(key=key, relogin=True)
    run = wandb.init(
      project = project_name,
      entity  = "philurame",
      name = f'{dataset}_{cacher_quantizer}_{solver}_{scheduler}_{nfe}',
      config = kwargs,
      save_code = True,
      mode='offline'
    )

    wandb.log(metrics)
    wandb.finish()

    # try to sync with wandb online:
    run_path = [x for x in os.listdir('wandb') if run.id in x][0]
    run_path = os.path.join('wandb', run_path)
    os.system(f'wandb sync {run_path}')

    print('__DONE')
    sys.stdout.flush()


if __name__ == '__main__':
  main()