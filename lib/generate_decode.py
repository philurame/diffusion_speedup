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
def generate(pipe, anns, nfe, batch_size=16, save_path=None):
  latents_gen = torch.zeros((len(anns), 4, 128, 128), dtype=torch.float16, device='cuda')
  dataloader  = torch.utils.data.DataLoader(anns, batch_size=batch_size, shuffle=False, num_workers=0)
  
  seed_everything(42)
  for i, prompts in tqdm(enumerate(dataloader), total=len(dataloader), desc='generate...'):
    generators = [torch.Generator(device='cpu').manual_seed(i*batch_size+g) for g in range(len(prompts))]
    latents = pipe(
      prompts, 
      num_inference_steps=nfe,
      guidance_scale=5,
      generator=generators,
      output_type='latent'
    )
    latents_gen[i*batch_size:(i+1)*batch_size] = latents

    if save_path is not None and nfe >= 50 and (i+1)%65 == 0: # "checkpoint"
      torch.save(latents_gen, save_path)
  return latents_gen

@torch.inference_mode()
def decode_vae(pipe, latents, batch_size=1, verbose=True):
  needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
  if needs_upcasting:
    pipe.upcast_vae()
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    
  latents = latents / pipe.vae.config.scaling_factor
  imgs = torch.zeros(latents.shape[0], 3, 1024, 1024, dtype=torch.uint8, device='cpu')

  for i in tqdm(range(0, latents.shape[0], batch_size), total=len(latents), desc='decode...'):
    decoded_latents = pipe.vae.decode(latents[i:i+batch_size].to('cuda'), return_dict=False)[0]
    imgs[i:i+batch_size] = (pipe.image_processor.postprocess(decoded_latents, output_type='pt')*255).to(device='cpu', dtype=torch.uint8)
  
  if needs_upcasting:
    pipe.vae.to(dtype=torch.float16)

  return imgs