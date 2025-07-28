import torch, numpy as np
import torch.nn.functional as F

# =============================================================================
# TS PARAMETRIZATIONS
# =============================================================================
class TSModel(torch.nn.Module):
  def __init__(self, nfe, param_method, init_method):
    super().__init__()
    if param_method not in ['cumprod', 'square']: raise NotImplementedError
    if init_method not in ['linear', 'leading']: raise NotImplementedError
    self.param_method = param_method
    self.init_method  = init_method

    if init_method == 'linear':
      timesteps = torch.linspace(0, 999, nfe + 1).round().flip(0)[:-1].float()
      logits = self.get_logits(timesteps)
    if init_method == 'leading':
      ratio = 1000 // nfe
      timesteps = (np.arange(0, nfe) * ratio).round()[::-1].copy()
      timesteps = torch.tensor(timesteps).float()
      logits = self.get_logits(timesteps)
    
    self.timesteps_logits      = torch.nn.Parameter(logits.clone(), requires_grad=True)
    self.unet_timesteps_logits = torch.nn.Parameter(logits.clone(), requires_grad=True)
    self()

  
  def _call(self, logits, **kwargs):
    '''
    converts logits to timesteps

    cumprod: R^d     -> R^d
    others:  R^{d+1} -> R^d
    '''

    if self.param_method == 'cumprod':
      timesteps = 1000*torch.cumprod(F.sigmoid(logits), 0)
    
    if self.param_method == 'square':
      cum_probs = torch.cumsum(logits**2, dim=0)
      cum_probs = cum_probs / max(cum_probs[-1], 1e-8)
      timesteps = (999 - cum_probs * 999)[:-1]

    return timesteps

  def forward(self, timesteps_logits=None, unet_timesteps_logits=None, **kwargs):
    timesteps_logits = timesteps_logits if timesteps_logits is not None else self.timesteps_logits
    unet_timesteps_logits = unet_timesteps_logits if unet_timesteps_logits is not None else self.unet_timesteps_logits

    self.timesteps = self._call(timesteps_logits, **kwargs)
    self.unet_timesteps = self._call(unet_timesteps_logits, **kwargs)
    return self.timesteps, self.unet_timesteps

  def get_logits(self, timesteps, **kwargs):
    if isinstance(timesteps, list): timesteps = torch.tensor(timesteps)

    if self.param_method == 'cumprod':
      logits = timesteps.clone() / 1000.0
      for i in range(1, len(timesteps)):
        logits[i] = max(timesteps[i] / timesteps[i-1], 1e-4)
      logits = torch.log(logits) - torch.log(1 - logits)
      return logits
  
    if self.param_method == 'square':
      probs = torch.cat([(999 - timesteps) / 999, torch.tensor([1.], device=timesteps.device, dtype=timesteps.dtype)]).clone()
      probs[1:] = probs[1:] - probs[:-1]
      return probs ** 0.5