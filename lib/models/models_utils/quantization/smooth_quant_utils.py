import torch, pickle
import torch.nn as nn

def get_smooth_scale(X_max, W_max, alpha=0.5, clipmin=1e-5):
  X_max = X_max.float().clamp(min=clipmin)
  W_max = W_max.float().clamp(min=clipmin)

  # smooth = (W^α) / (X^(1−α) + eps)
  smooth = ( W_max**alpha / X_max**(1 - alpha) ).clamp(min=clipmin).half()
  return smooth

def aggregate_absmax(**config):
  '''
  - aggregates X_absmax stats of the shape (prompts, timesteps, D) into (D,) based on config (timesteps could be 2 or NFE)
  returns smooth_scale (D, ) for each layer
  '''
  path = config['path_absmax']
  aggr_timesteps = config['aggr_timesteps']
  aggr_prompts   = config['aggr_prompts']
  alpha   = config.get('alpha', 0.5)
  clipmin = config.get('clipmin', 1e-5)
  aggr_weight = config.get('aggr_weight', False)

  with open(path, 'rb') as f:
    stats = pickle.load(f)
  
  smooth_scales = {}

  for k, v in stats.items():
    X_absmax = v['X_absmax']

    if aggr_weight:
      X_absmax = torch.tensor([1.])
      alpha = 1
      aggr_timesteps = None
      aggr_prompts   = None

    if aggr_timesteps == 'max':
      X_absmax = X_absmax.max(dim=1).values
    elif aggr_timesteps == 'median':
      X_absmax = X_absmax.quantile(0.5, dim=1)
    elif aggr_timesteps == 'quantile':
      X_absmax = X_absmax.quantile(0.99, dim=1)
    
    if aggr_prompts == 'max':
      X_absmax = X_absmax.max(dim=0).values
    elif aggr_prompts == 'mean':
      X_absmax = X_absmax.mean(dim=0)
    elif aggr_prompts == 'median':
      X_absmax = X_absmax.quantile(0.5, dim=0)
    elif aggr_prompts == 'quantile':
      X_absmax = X_absmax.quantile(0.99, dim=0)
    
    smooth_scales[k] = get_smooth_scale(X_absmax, v['W_absmax'], alpha, clipmin)
  
  return smooth_scales




def get_absmax(pipe, calib_path, nfe=10):
  with open(calib_path, 'r') as f:
    prompts = [ln.strip() for ln in f if ln.strip()]

  stats = {}
  hooks = []

  def register_layer(layer, name):

    stats[name] = {'X_absmax': [[] for _ in range(len(prompts))], 'W_absmax': None}
    if stats[name]['W_absmax'] is None:
      stats[name]['W_absmax'] = layer.weight.detach().abs().max(dim=0)[0].float()

    def static_hook(module, inp, out):
      X = inp[0].detach()
      X_absmax = X.reshape(-1, X.shape[-1]).abs().max(dim=0)[0].float()    
      stats[name]['X_absmax'][prompt_N].append(X_absmax)

    hooks.append(layer.register_forward_hook(static_hook))

  for mod_name, mod in pipe.text_encoder.named_modules():
    if isinstance(mod, nn.Linear):
      register_layer(mod, f"{mod_name}")
  
  if 'unet' in pipe.__dict__:
    transformer_unet = pipe.unet
  else:
    transformer_unet = pipe.transformer
  for mod_name, mod in transformer_unet.named_modules():
    if isinstance(mod, nn.Linear):
      register_layer(mod, f"{mod_name}")

  with torch.no_grad():
    for prompt_N, prompt in enumerate(prompts):
      pipe(prompt, num_inference_steps=nfe, generator=torch.Generator('cpu').manual_seed(prompt_N), output_type='latent')       

  for h in hooks:
    h.remove()
  
  for k, v in stats.items():
    v['X_absmax'] = torch.stack([torch.stack(i) for i in v['X_absmax']]).cpu()
    v['W_absmax'] = v['W_absmax'].cpu()

  return stats