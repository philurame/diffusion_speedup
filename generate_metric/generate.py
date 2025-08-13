import click, tqdm, pickle
import torch, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # DIFFUSION_SPEEDUP
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  model_registry, 
  data_registry, 
)
import_dir(os.path.join(ROOT, 'lib'))

def construct_pipeline(solver, scheduler, model_name, half=True, **pipe_kwargs):
  '''construct a pipeline with given model_name, solver and scheduler'''
  PipeClass   = model_registry[model_name]
  SolverClass = solver_registry[solver]
  SchedulerClass = scheduler_registry[scheduler]
  pipe = PipeClass.from_pretrained(half=half, **pipe_kwargs)

  # combine solver and scheduler
  class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
  pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
  return pipe


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
@click.option('--dataset', type=str, required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--solver', type=str, required=True)
@click.option('--scheduler', type=str, required=True)
@click.option('--nfe', type=int, required=True)
@click.option('--path_save', type=str, required=True)
@click.option('--num_samples', type=int, default=100)
@click.option('--batch_size', type=int, default=16)
@click.option('--seed_shift', type=int, default=0)
@click.option('--device', type=int, default=-1)
@click.option('--is_gen_img', type=int, default=0)
def main(**kwargs):
  print(
  '\n'+'#'*50, 
  *[f'{k}={v}' for k, v in kwargs.items()],
  '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  kwargs['device'] = 'cuda' if kwargs['device'] == -1 else f"cuda:{kwargs['device']}"
  dataset = kwargs['dataset']
  nfe = kwargs['nfe']
  solver = kwargs['solver']
  scheduler = kwargs['scheduler'] 
  model_name = kwargs['model_name']
  device = kwargs['device']
  path_save = kwargs['path_save']
  num_samples = kwargs['num_samples']
  batch_size = kwargs['batch_size']
  seed_shift = kwargs['seed_shift']
  is_gen_img = kwargs['is_gen_img']

  device_id = int(device.split(':')[-1]) if ':' in device else 0
  torch.cuda.set_device(device_id)

  data = data_registry[dataset]()
  prompts = data.prompts[:num_samples]
  prompts = [prompts[i%len(prompts)] for i in range(num_samples)] # cycle prompts
  
  pipe = construct_pipeline(solver, scheduler, model_name, device=device)
  
  ########################################
  # Gen
  ########################################
  seed_everything(42)

  noises = torch.zeros(num_samples, *pipe.latent_dims, dtype=torch.float16, device='cpu')
  gen_latents = torch.zeros(num_samples, *pipe.latent_dims, dtype=torch.float16, device='cpu')
  for i in tqdm.tqdm(range(0, num_samples, batch_size), total=num_samples//batch_size, desc='latents...'):
    prompts_batch = prompts[i:i+batch_size]
    generators = [torch.Generator(device='cpu').manual_seed(seed_shift+i+g) for g in range(len(prompts_batch))]
    noise = pipe.prepare_latents(batch_size=len(prompts_batch), generator=generators)
    latents = pipe(
      prompts_batch, 
      num_inference_steps=nfe,
      latents=noise,
      output_type='latent'
    )
    noises[i:i+batch_size] = noise
    gen_latents[i:i+batch_size] = latents
  
  gen_images = None
  try:
    if is_gen_img:
      gen_images = torch.zeros(num_samples, *pipe.img_dims, dtype=torch.uint8, device='cpu')
      for i in tqdm.tqdm(range(0, num_samples, batch_size), total=num_samples//batch_size, desc='decode...'):
        gen_images[i:i+batch_size] = pipe.decode_latents(gen_latents[i:i+batch_size].to(pipe.device)).cpu()
  except Exception as e:
    gen_images = None
    print(f"decode images error, saving as None: {e}")
    
  with open(path_save, 'wb') as f:
    pickle.dump({
      'prompts': prompts,
      'noise': noises,
      'latents': gen_latents,
      'imgs': gen_images
    }, f)
  
  print('__DONE')
  sys.stdout.flush()


if __name__ == '__main__':
  main()