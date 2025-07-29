import torch, copy, pickle
from quantize_linear import QuantizedLinear
from quarot_utils import random_hadamard_matrix
from smooth_quant_utils import aggregate_absmax

def quantize_model(pipe, config, inplace=False, skip_layers=[], quantize_text_encoder=True):
  # device = pipe.device
  if inplace: 
    pipe_q = pipe
  else:
    pipe_q = copy.deepcopy(pipe)

  group_size = config['group_size']
  skip_activations_timesteps = config['skip_activations_timesteps']
  skip_timesteps = config['skip_timesteps']

  # ================================================================================
  # HADAMARD: construct one rotation_matrix
  # ================================================================================
  bruterots = None
  if config['use_bruterot']:
    with open(config['bruterot']['path'], 'rb') as f:
      bruterots = pickle.load(f)

  # ================================================================================
  # SMOOTH QUANT: construct use_smoothquant params
  # ================================================================================
  smooth_scales = None
  if config['use_smoothquant']:
    if config['smoothquant']['path_smooth_quant']:
      with open(config['smoothquant']['path_smooth_quant'], 'rb') as f:
        smooth_scales = pickle.load(f)
    else:
      smooth_scales = aggregate_absmax(**config['smoothquant'])

  # ================================================================================
  # DuQuant: construct duquant_params from w_duquant_params and/or a_duquant_params
  # ================================================================================
  duquant_params = None
  if config['use_duquant']:
    wd_path = config['duquant']['w_path']
    ad_path = config['duquant']['a_path']
    w_duquant_params = {}
    a_duquant_params = {}
    
    if wd_path:
      with open(wd_path, 'rb') as f:
        w_duquant_params = pickle.load(f)
    if ad_path:
      with open(ad_path, 'rb') as f:
        a_duquant_params = pickle.load(f)

    all_keys = set(w_duquant_params.keys()) | set(a_duquant_params.keys())
    duquant_params = {k: [p for p in [w_duquant_params.get(k), a_duquant_params.get(k)] if p] for k in all_keys}

  # ================================================================================
  # HADAMARD: construct one rotation_matrix
  # ================================================================================
  torch.manual_seed(42)
  rotation_matrix = random_hadamard_matrix(group_size) / torch.sqrt(torch.tensor(group_size))
  rotation_matrix = rotation_matrix.half()

  def replace_linears(root_module, is_text_encoder):
    # 1) collect
    to_replace = []
    for full_name, module in root_module.named_modules():
      if isinstance(module, torch.nn.Linear):
        to_replace.append((full_name, module))

    # 2) replace
    for full_name, orig_mod in to_replace:
      if full_name in skip_layers: continue
      
      # split "foo.bar.baz" into ["foo","bar","baz"]
      path = full_name.split('.')
      # locate parent: everything up to the last
      parent = root_module
      for sub in path[:-1]:
        parent = getattr(parent, sub)
      child_name = path[-1]

      if is_text_encoder: # dont skip "timesteps" in text_encoder:
        config['skip_activations_timesteps'] = []
        config['skip_timesteps'] = []
      else:
        config['skip_activations_timesteps'] = skip_activations_timesteps
        config['skip_timesteps'] = skip_timesteps

      # create the QuantizedLinear with the same dims
      qmod = QuantizedLinear(
        config       = config,
        fp_module    = orig_mod,
        in_features  = orig_mod.in_features,
        out_features = orig_mod.out_features,
        layer_name   = full_name,
        
        bruterots = bruterots,
        rotation_matrix = rotation_matrix,
        smooth_scale    = smooth_scales,
        duquant_params  = duquant_params,
      )
      setattr(parent, child_name, qmod)

  # apply to both submodules
  if quantize_text_encoder:
    replace_linears(pipe_q.text_encoder, is_text_encoder=True)
  if 'unet' in pipe.__dict__:
    replace_linears(pipe_q.unet, is_text_encoder=False)
  else:
    replace_linears(pipe_q.transformer, is_text_encoder=False)
  
  return pipe_q




def fetch_errors(pipe, prompts, nfe=10, seeds=[]):
  hooks = []
  errors = {}

  def register_layer(layer, name):
    errors[name] = []

    def hook(module, inp, out):
      error = module.error.cpu()
      errors[name] = errors.get(name, []) + [error]    
    hooks.append(layer.register_forward_hook(hook))

  # 2) hook every Linear in text_encoder and transformer
  for mod_name, mod in pipe.text_encoder.named_modules():
    if isinstance(mod, QuantizedLinear):
      register_layer(mod, f"text_encoder.{mod_name}")

  if 'unet' in pipe.__dict__:
    transformer_unet = pipe.unet
  else:
    transformer_unet = pipe.transformer

  for mod_name, mod in transformer_unet.named_modules():
    if isinstance(mod, QuantizedLinear):
      register_layer(mod, f"transformer.{mod_name}")

  # 3) single-prompt forward to collect stats
  latents = []
  with torch.no_grad():
    for n, prompt in enumerate(prompts):
      seed = seeds[n] if seeds else n
      latent = pipe(prompt, num_inference_steps=nfe, generator=torch.Generator(device='cpu').manual_seed(seed), output_type='latent')
      latents.append(latent)
  
  werrors ={}
  for mod_name, mod in pipe.text_encoder.named_modules():
    if 'inference_count' in mod.__dict__:
      mod.inference_count = 0
      werrors[mod_name] = mod.werror.item()

  for mod_name, mod in transformer_unet.named_modules():
    if 'inference_count' in mod.__dict__:
      mod.inference_count = 0
      werrors[mod_name] = mod.werror.item()

  # cleanup hooks
  for h in hooks:
    h.remove()
  
  for k, v in errors.items():
    errors[k] = torch.stack(v)

  return errors, werrors, latents
