import wandb, click
import torch, os, sys, gc

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from lib.generate_decode import decode_vae
from lib.registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  model_registry, 
  data_registry, 
  metric_registry
)
import_dir(os.path.join(ROOT, 'lib')) # fill solver/scheduler/cacher/quantizer registries

def construct_pipeline(solver, scheduler, model_name, half=True, **pipe_kwargs):
  '''construct a pipeline with given model_name, solver and scheduler'''
  
  # fetch pipe, solver, scheduler classes
  PipeClass   = model_registry[model_name]
  SolverClass = solver_registry[solver]
  SchedulerClass = scheduler_registry[scheduler]

  # load pipe
  pipe = PipeClass.from_pretrained(half=half, **pipe_kwargs)

  # combine solver and scheduler
  class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
  pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)

  pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
  return pipe

def calc_metrics(metric_names, **data):
  '''calculate metrics by looping over all metrics in metrics_registry'''

  res_metrics = {}
  for metric_name in metric_names:
    metricInstance = metric_registry[metric_name]()
    res_metrics[metric_name] = metricInstance(**data)

    gc.collect()
    torch.cuda.empty_cache()
  return res_metrics

##########################################################################################
# MAIN
##########################################################################################
@click.command()
@click.option('--dataset', type=str, required=True, help='PARTI|COCO|(CIFAR)')
@click.option('--model_name', type=str, required=True, help='which pipe to use')
@click.option('--solver', type=str, required=True, help='supported methods are in lib/solvers')
@click.option('--scheduler', type=str, required=True, help='supported methods are in lib/schedulers')
@click.option('--nfe', type=int, required=True, help='num inference steps')
@click.option('--key', type=str, required=True, help='wandb key')
@click.option('--metric_names', type=str, required=True, help='list of metrics separated by comma')
def main(**kwargs):
  max_samples = 10_000
  key = kwargs['key']
  dataset = kwargs['dataset']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  model_name = kwargs['model_name']
  metric_names = kwargs['metric_names'].split(',')

  data_path = os.path.join(ROOT, 'DATA')
  save_path = os.path.join(data_path, model_name, f'{dataset}_{nfe}', f'{solver}_{scheduler}.pt')

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    f'{save_path=}',
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  data = data_registry[dataset](data_path, max_samples)
  pipe = construct_pipeline(solver, scheduler, model_name, half=True)

  ########################################
  # METRICS
  ########################################

  if os.path.exists(save_path):
    gen_latents = torch.load(save_path, map_location='cpu')[:max_samples]
    gen_imgs = decode_vae(pipe, gen_latents)
    del gen_latents
  else:
    gen_imgs = None
  
  gc.collect()
  torch.cuda.empty_cache()

  metrics = calc_metrics(
    metric_names=metric_names,
    imgs_gen=gen_imgs, 
    imgs_real=data.imgs, 
    anns=data.anns,
    pipe=pipe, 
    nfe=nfe
  )
  print(metrics)
  sys.stdout.flush()

  ########################################
  # LOG
  ########################################

  # choose your wandb project
  project_name = 'DIFFUSION_METRICS'
  wandb.login(key=key, relogin=True)
  run = wandb.init(
    project = project_name,
    entity  = "philurame",
    name = f'{dataset}_{model_name}_{solver}_{scheduler}_{nfe}',
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