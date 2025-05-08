import wandb, click, tqdm
import torch, os, sys, gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # DIFFUSION_SPEEDUP
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

def seed_everything(seed=42):
  import random
  import numpy as np
  import torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

##########################################################################################
# MAIN
##########################################################################################
@click.command()
@click.option('--dataset', type=str, required=True, help='PARTI|COCO|(CIFAR)')
@click.option('--model_name', type=str, required=True, help='which pipe to use')
@click.option('--solver', type=str, required=True, help='supported methods are in lib/solvers')
@click.option('--scheduler', type=str, required=True, help='supported methods are in lib/schedulers')
@click.option('--nfe', type=int, required=True, help='num inference steps')
@click.option('--metric_names', type=str, required=True, help='list of metrics separated by comma')
@click.option('--key', type=str, default=None)
@click.option('--wandb_project_name', type=str, default='DIFFUSION_METRICS')
@click.option('--max_samples', type=int, default=30_000)
@click.option('--batch_size', type=int, default=16)
@click.option('--device', type=int, default=-1)
def main(**kwargs):
  kwargs['device'] = 'cuda' if kwargs['device'] == -1 else f"cuda:{kwargs['device']}"

  dataset = kwargs['dataset']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  model_name = kwargs['model_name']
  metric_names = kwargs['metric_names'].split(',')
  device = kwargs['device']
  key = kwargs['key']
  project_name = kwargs['wandb_project_name']
  max_samples = kwargs['max_samples']
  batch_size = kwargs['batch_size']
  
  data_path = os.path.join(ROOT, 'DATA')
  save_path = os.path.join(data_path, model_name, f'{dataset}_{nfe}', f'{solver}_{scheduler}.pt')

  if dataset == 'COCO':
    path_ddim_200 = os.path.join(data_path, 'imgs_ddim200_224.pt')
  elif dataset == 'COCO30':
    path_ddim_200 = os.path.join(data_path, 'imgs30_DDIM200_224.pt')
  else:
    path_ddim_200 = ''

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    f'{save_path=}',
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()
  
  # wandb.login(key=key, relogin=True)
  # api = wandb.Api()
  # if api.runs(f"philurame/{kwargs['wandb_project_name']}", filters={'config.dataset': dataset, 'config.model_name': model_name, 'config.solver': solver, 'config.scheduler': scheduler, 'config.nfe': nfe}).__len__() > 0:
  #   print('run already exists!')
  #   sys.exit(0)

  data = data_registry[dataset](data_path, max_samples)
  
  if solver not in solver_registry:
    print('SLOVER NOT IN REGISTRY, REPLACING WITH DDIM')
    solver = 'DDIM'
  pipe = construct_pipeline(solver, scheduler, model_name, half=True, device=device)
  solver = kwargs['solver']
  
  ########################################
  # GM
  ########################################
  N = len(data.anns)
  gen_imgs = torch.zeros(N, *pipe.img_dims, dtype=torch.uint8, device='cpu')

  seed_everything(42)
  for i in tqdm.tqdm(range(0, N, batch_size), total=N//batch_size, desc='generate+decode...'):
    prompts_batch = data.anns[i:i+batch_size]
    generators = [torch.Generator(device='cpu').manual_seed(i+g) for g in range(len(prompts_batch))]
    imgs = pipe(
      prompts_batch, 
      num_inference_steps=nfe,
      generator=generators,
      output_type='img'
    )
    gen_imgs[i:i+batch_size] = imgs
  
  gc.collect()
  torch.cuda.empty_cache()

  metrics = calc_metrics(
    metric_names=metric_names,
    imgs_gen=gen_imgs, 
    imgs_real=data.imgs, 
    anns=data.anns,
    pipe=pipe, 
    nfe=nfe,
    device=device,
    dataset=dataset,
    path_ddim_200=path_ddim_200
  )
  print(metrics)
  sys.stdout.flush()

  ########################################
  # LOG
  ########################################
  if key is not None:
    wandb.login(key=key, relogin=True)
    run = wandb.init(
      project = project_name,
      entity  = "philurame",
      name = f'{dataset}_{model_name}_{solver}_{scheduler}_{nfe}',
      config = kwargs,
      save_code = True,
      mode='online'
    )
    wandb.log(metrics)
    wandb.finish()
    # # try to sync with wandb online:
    # run_path = [x for x in os.listdir('wandb') if run.id in x][0]
    # run_path = os.path.join('wandb', run_path)
    # os.system(f'wandb sync {run_path}')

  print('__DONE')
  sys.stdout.flush()


if __name__ == '__main__':
  main()