import torch, os, sys, click

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # DIFFUSION_SPEEDUP
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from lib.generate_decode import generate
from lib.registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  model_registry, 
  data_registry, 
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


##########################################################################################
# MAIN
##########################################################################################
@click.command()
@click.option('--dataset', type=str, required=True, help='PARTI|COCO|(CIFAR)')
@click.option('--model_name', type=str, required=True, help='which pipe to use')
@click.option('--solver', type=str, required=True, help='supported methods are in lib/solvers')
@click.option('--scheduler', type=str, required=True, help='supported methods are in lib/schedulers')
@click.option('--nfe', type=int, required=True, help='num inference steps')
@click.option('--max_samples', type=int, default=30_000)
@click.option('--batch_size', type=int, default=128)
@click.option('--device', type=int, required=False, default=-1)
@click.option('--key', type=str, required=False, default='')
@click.option('--metric_names', type=str, required=False, default='')
def main(**kwargs):
  kwargs['device'] = 'cuda' if kwargs['device'] == -1 else f"cuda:{kwargs['device']}"
  max_samples = kwargs['max_samples']
  dataset = kwargs['dataset']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  model_name = kwargs['model_name']
  device = kwargs['device']
  batch_size = kwargs['batch_size']

  data_path = os.path.join(ROOT, 'DATA')
  save_path = os.path.join(data_path, model_name, f'{dataset}_{nfe}', f'{solver}_{scheduler}.pt')

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    f'{save_path=}',
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  if os.path.exists(save_path):
    print(f'{save_path} already exists')
    sys.exit(0)

  assert isinstance(nfe, int) and nfe > 0

  data = data_registry[dataset](data_path, max_samples)
  pipe = construct_pipeline(solver, scheduler, model_name, half=True, device=device)

  ########################################
  # GENERAtE
  ########################################
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  gen_latents = generate(pipe, data.anns, nfe, save_path=save_path, batch_size=batch_size)
  torch.save(gen_latents, save_path)

  print('__DONE')
  sys.stdout.flush()

if __name__ == '__main__':
  main()