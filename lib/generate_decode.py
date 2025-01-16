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

def generate(pipe, anns, nfe, batch_size=16):
  latents_gen = torch.zeros((len(anns), 4, 128, 128), dtype=torch.float16, device='cuda')
  dataloader  = torch.utils.data.DataLoader(anns, batch_size=batch_size, shuffle=False, num_workers=0)
  
  seed_everything(42)
  for i, prompts in tqdm(enumerate(dataloader), total=len(dataloader), desc='generate...'):
    generators = [torch.Generator(device='cpu').manual_seed(i*batch_size+g) for g in range(len(prompts))]
    latents = pipe(prompts, 
                   num_inference_steps=nfe, 
                   guidance_scale=5, 
                   generator=generators,
                   return_dict=False, 
                   output_type='latent'
                   )[0]
    latents_gen[i*batch_size:(i+1)*batch_size] = latents
  return latents_gen

def decode_vae(pipe, fake_latents):
  fake_latents = fake_latents / pipe.vae.config.scaling_factor
  fake_imgs = torch.zeros(fake_latents.shape[0], 3, 1024, 1024, dtype=torch.uint8, device='cpu')
  for i in tqdm(range(fake_latents.shape[0]), desc='vae decoding...'):
    decoded_latents = pipe.vae.decode(fake_latents[i:i+1].to(device=pipe.device), return_dict=False)[0]
    fake_imgs[i] = (pipe.image_processor.postprocess(decoded_latents, output_type='pt')*255).to(device='cpu', dtype=torch.uint8)
  return fake_imgs