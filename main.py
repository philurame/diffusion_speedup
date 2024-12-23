import wandb, click
import torch, os, sys, gc
from diffusers import AutoencoderKL

from lib.generate_decode import generate, decode_vae
from lib.solver_scheduler import solver_registry, scheduler_registry
from lib.cacher_quantizer import cacher_quantizer_registry

from lib.data import data_registry
from lib.metric import metrics_registry

from utils.other import add_sample_imgs
from utils.other import is_already_calculated

##########################################################################################
##########################################################################################

def load_pipe(solver, scheduler, cacher_quantizer, is_optimize, add_vae):
  PipeClass = cacher_quantizer_registry[cacher_quantizer]
  SolverClass = solver_registry[solver]
  SchedulerClassMixin = scheduler_registry[scheduler]

  pipe = PipeClass.from_pretrained()

  class SolverSchedulerConstructor(SolverClass, SchedulerClassMixin):
    def set_timesteps(self, *args, **kwargs):
      SchedulerClassMixin.set_timesteps(self, *args, **kwargs)
  pipe.scheduler = SolverSchedulerConstructor.from_config()

  pipe = pipe.to('cuda')
  pipe.set_progress_bar_config(disable=True)  

  if is_optimize:
    # torch compile !cp /usr/include/crypt.h /home/ekneudachina/.conda/envs/philurame_venv/include/python3.9/
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode='reduce-overhead', fullgraph=True)
    pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead', fullgraph=True)
  if add_vae:
    pipe.vae = AutoencoderKL.from_pretrained(
      'madebyollin/sdxl-vae-fp16-fix',
      use_safetensors=True,
      torch_dtype=torch.float16,
    ).to('cuda')
  return pipe

def load_data(data_path, dataset, max_samples):
  return data_registry[dataset](data_path, max_samples)

def get_metrics(**kwargs):
  res_metrics = {}
  for metric_name in metrics_registry:
    metricInstance = metrics_registry[metric_name](**kwargs)
    if hasattr(metricInstance, 'to'):
      metricInstance = metricInstance.to('cpu')
    res_metrics[metric_name] = metricInstance()

    # try to free memory
    del metricInstance
    gc.collect()
    torch.cuda.empty_cache()

  return res_metrics

@click.command()
@click.option('--key', type=str, required=True, help='wandb key')
@click.option('--root', type=str, required=True, help='root dir full path')
@click.option('--max_samples', type=int, required=True, help='max num of images to generate')
@click.option('--dataset', type=str, required=True, help='PARTI | COCO')
@click.option('--generate', type=int, required=True, help='generate or calc metrics')
@click.option('--nfe', type=int, required=True, help='num inference steps')
@click.option('--solver', type=str, required=True, help='supported methods are in pipe_method.py')
@click.option('--scheduler', type=str, required=True, help='supported methods are in pipe_method.py')
@click.option('--cacher_quantizer', type=str, required=True, help='supported methods are in pipe_method.py')
def main(**kwargs):
  key = kwargs['key']
  root = kwargs['root']
  max_samples = kwargs['max_samples'] 
  dataset = kwargs['dataset']
  is_generate = kwargs['generate']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  cacher_quantizer = kwargs['cacher_quantizer']

  data_path = os.path.join(root, 'DATA')
  save_path = os.path.join(data_path, cacher_quantizer, f'{dataset}_{nfe}', f'{solver}_{scheduler}.pt')

  print('\n'+'#'*50, 
        *[f'{k}={v}' for k, v in kwargs.items()],
        f'{save_path=}',
        '#'*50+'\n', sep='\n')
  sys.stdout.flush()

  assert dataset in ['PARTI', 'COCO']
  assert isinstance(nfe, int) and nfe > 0
  assert os.path.exists(root)
  assert bool(is_generate) != os.path.exists(save_path)
  assert bool(is_generate) or not is_already_calculated(save_path, calculated_path=os.path.join(data_path, 'METRICS.csv'))
  assert 0 < max_samples <= 10000

  data = load_data(data_path, dataset, max_samples)
  pipe = load_pipe(solver, scheduler, cacher_quantizer, 
                   is_optimize=is_generate and cacher_quantizer == 'NONE', 
                   add_vae=not is_generate
                   )

  if is_generate: 
    ########################################
    # GENERAtE
    ########################################
  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gen_latents = generate(pipe, data.anns, nfe)
    torch.save(gen_latents, save_path)
  
  else:
    ########################################
    # METRICS
    ########################################

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
    metrics = get_metrics(
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

    print('_ALL_DONE')
    sys.stdout.flush()


if __name__ == '__main__':
  main()