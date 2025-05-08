import torch
import numpy as np
import random
from tqdm import tqdm

def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

@torch.inference_mode()
def generate(pipe, anns, nfe=None, timesteps=None, batch_size=16, save_path=None):
  device = pipe.device
  latents_gen = torch.zeros((len(anns), *pipe.latent_dims), dtype=torch.float16, device=device)
  dataloader  = torch.utils.data.DataLoader(anns, batch_size=batch_size, shuffle=False, num_workers=0)
  
  seed_everything(42)
  for i, prompts in tqdm(enumerate(dataloader), total=len(dataloader), desc='generate...'):
    generators = [torch.Generator(device='cpu').manual_seed(i*batch_size+g) for g in range(len(prompts))]
    latents = pipe(
      prompts, 
      num_inference_steps=nfe,
      timesteps=timesteps,
      generator=generators,
      output_type='latent'
    )
    latents_gen[i*batch_size:(i+1)*batch_size] = latents

    if save_path is not None and nfe >= 50 and (i+1)%65 == 0: # "checkpoint"
      torch.save(latents_gen, save_path)
  return latents_gen

@torch.inference_mode()
def decode_vae(pipe, latents, batch_size=1):
  device = pipe.device
  needs_upcasting = pipe.img_dims[-1]==1024 and pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
  if needs_upcasting:
    pipe.upcast_vae()
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    
  latents = latents / pipe.vae.config.scaling_factor
  imgs = torch.zeros(latents.shape[0], *pipe.img_dims, dtype=torch.uint8, device='cpu')

  for i in tqdm(range(0, latents.shape[0], batch_size), total=len(latents)//batch_size, desc='decode...'):
    decoded_latents = pipe.vae.decode(latents[i:i+batch_size].to(device), return_dict=False)[0]
    imgs[i:i+batch_size] = (pipe.image_processor.postprocess(decoded_latents, output_type='pt')*255).clip(0,255).to(device='cpu', dtype=torch.uint8)
  
  if needs_upcasting:
    pipe.vae.to(dtype=torch.float16)

  return imgs
  



@torch.inference_mode()
def generate_part(pipe, anns, nfe=None, timesteps=None, batch_size=16, save_path=None, i_from=0, i_to=None):
  device = pipe.device
  if i_to is None: i_to = len(anns)

  anns_subset = anns[i_from:i_to]
  subset_size = len(anns_subset)
  latents_gen = torch.zeros((subset_size, *pipe.latent_dims), dtype=torch.float16, device=device)
  dataloader = torch.utils.data.DataLoader(anns_subset, batch_size=batch_size, shuffle=False, num_workers=0)

  seed_everything(42)
  for i, prompts in tqdm(enumerate(dataloader), total=len(dataloader), desc='generate...'):
    global_offset = i_from + i * batch_size
    generators = [torch.Generator(device='cpu').manual_seed(global_offset + g) for g in range(len(prompts))]

    latents = pipe(
      prompts,
      num_inference_steps=nfe,
      timesteps=timesteps,
      guidance_scale=5,
      generator=generators,
      output_type='latent'
    )

    start_idx = i * batch_size
    end_idx = start_idx + latents.shape[0]
    latents_gen[start_idx:end_idx] = latents

    if save_path is not None and nfe >= 50 and (i+1) % 65 == 0:
      torch.save(latents_gen, save_path)

  return latents_gen