import torch

class SchedulerMixin:
  def __init__(self, config, **kwargs):
    for k, v in config.items(): setattr(self, k, v)
    beta_schedule = config['beta_schedule']
    beta_start = config['beta_start']
    beta_end = config['beta_end']
    num_train_timesteps = config['num_train_timesteps']
    self.is_flow_matching = config.get('flow_mathing', False)
    if self.is_flow_matching:
      self.step_index = 0
      return

    if beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
      self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
  
    if kwargs.get("rescale_betas_zero_snr", False):
      self.alphas_cumprod[-1] = 2**(-24)
    
    self.step_index = 0

  def prepare_solver_data(self, timesteps, device, sigmas=None, last_sigma_zero=False):
    '''
    SIGMA_LAST is "SIGMA_MIN" here
    '''
    self.step_index = 0
    self.model_outputs = [None]
    self.num_inference_steps = len(timesteps)

    if not isinstance(timesteps, torch.Tensor): timesteps = torch.tensor([i for i in timesteps])
    self.timesteps = timesteps.float().to(device)

    if self.is_flow_matching:
      if (sigmas is not None) and (len(sigmas) == len(timesteps)):
        sigmas = sigmas if isinstance(sigmas, torch.Tensor) else torch.as_tensor(sigmas, device=timesteps.device, dtype=torch.float32)
        sigmas = sigmas.to(device=timesteps.device, dtype=torch.float32)
      else:
        sigmas = timesteps/1000.0

      self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device, dtype=sigmas.dtype)],dim=0)
      return

    if sigmas is not None:
      if not isinstance(sigmas, torch.Tensor): sigmas = torch.tensor([i for i in sigmas])
      if len(sigmas) == len(timesteps):
        sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]).sqrt().unsqueeze(0)
        sigmas = torch.cat([sigmas, sigma_last])
      self.sigmas = sigmas.to(torch.float32)
    else:
      self.sigmas = self._timesteps_to_sigmas(timesteps, last_sigma_zero)

  
  def sigma_to_alpha_t(self, sigma): # returns sqrt(alpha_t)
    return  1 / ((sigma**2 + 1) ** 0.5)
  
  def eps_pred_from(self, prediction_type, noise_pred, latents):
    if prediction_type == "epsilon":
      return noise_pred
    if prediction_type == "v_prediction":
      sigma_t = self.sigmas[self.step_index]
      alpha_t = self.sigma_to_alpha_t(sigma_t)
      return alpha_t * (latents * sigma_t + noise_pred)

  def _sigma_to_alpha_sigma_t(self, sigma):
    alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
    sigma_t = sigma * alpha_t
    return alpha_t, sigma_t

  def _timesteps_to_sigmas(self, timesteps, last_sigma_zero):
    sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()
    timesteps = timesteps.to(sigmas.device)
    N = sigmas.shape[0]

    idx_lower = timesteps.floor().long()
    idx_upper = idx_lower + 1
    idx_upper = torch.where(idx_upper >= N, idx_lower, idx_upper)
    w = timesteps - idx_lower.to(timesteps.dtype)
    sigma_interp = sigmas[idx_lower] * (1 - w) + sigmas[idx_upper] * w

    if timesteps[-1] < 0.001 or last_sigma_zero: 
      sigma_last = torch.tensor([0.], dtype=torch.float32)
    else:
      sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]).sqrt().unsqueeze(0)
    sigmas = torch.cat([sigma_interp, sigma_last]).to(torch.float32)
    return sigmas