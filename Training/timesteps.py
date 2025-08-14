import torch, numpy as np
import torch.nn.functional as F

# =============================================================================
# TS PARAMETRIZATIONS
# =============================================================================
class TSModel(torch.nn.Module):
  def __init__(self, nfe, param_method, init_method, max_timestep=999.5):
    super().__init__()
    self.max_timestep = max_timestep
    self.param_method = param_method
    self.init_method  = init_method

    if init_method == 'linear':
      timesteps = torch.linspace(0, 999, nfe + 1).round().flip(0)[:-1].float()
    if init_method == 'leading':
      ratio = 1000 // nfe
      timesteps = (np.arange(0, nfe) * ratio).round()[::-1].copy()
      timesteps = torch.tensor(timesteps).float()
    if init_method == 'flow':
      def sd3_time_shift(t, shift):
        return (shift * t) / (1 + (shift - 1) * t)
      sigmas = sd3_time_shift(torch.linspace(1, 0, nfe + 1), 7)
      timesteps = sigmas[:-1] * 1000
    if init_method == 'flow9':
      def sd3_time_shift(t, shift):
        return (shift * t) / (1 + (shift - 1) * t)
      sigmas = sd3_time_shift(torch.linspace(1, 0, nfe + 1), 9)
      timesteps = sigmas[:-1] * 1000

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
      timesteps = self.max_timestep*torch.cumprod(F.sigmoid(logits), 0)
    
    if self.param_method == 'square':
      cum_probs = torch.cumsum(logits**2, dim=0)
      cum_probs = cum_probs / max(cum_probs[-1], 1e-8)
      timesteps = (self.max_timestep - cum_probs * self.max_timestep)[:-1]
    
    if self.param_method == 'module':
      cum_probs = torch.cumsum(logits.abs(), dim=0)
      cum_probs = cum_probs / max(cum_probs[-1], 1e-8)
      timesteps = (self.max_timestep - cum_probs * self.max_timestep)[:-1]

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
      logits = timesteps.clone() / self.max_timestep
      for i in range(1, len(timesteps)):
        logits[i] = max(timesteps[i] / timesteps[i-1], 1e-4)
      logits = torch.log(logits) - torch.log(1 - logits)
      return logits
  
    if self.param_method == 'square':
      probs = torch.cat([(self.max_timestep - timesteps) / self.max_timestep, torch.tensor([1.], device=timesteps.device, dtype=timesteps.dtype)]).clone()
      probs[1:] = probs[1:] - probs[:-1]
      return probs ** 0.5

    if self.param_method == 'module':
      probs = torch.cat([(self.max_timestep - timesteps) / self.max_timestep, torch.tensor([1.], device=timesteps.device, dtype=timesteps.dtype)]).clone()
      probs[1:] = probs[1:] - probs[:-1]
      return probs