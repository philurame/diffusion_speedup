import torch

class SchedulerMixin:
  def __init__(self, config, **kwargs):
    for k, v in config.items(): setattr(self, k, v)
    beta_schedule = config['beta_schedule']
    beta_start = config['beta_start']
    beta_end = config['beta_end']
    num_train_timesteps = config['num_train_timesteps']

    if beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
      self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
  
    if kwargs.get("rescale_betas_zero_snr", False):
      self.alphas_cumprod[-1] = 2**(-24)

  def prepare_solver_data(self, timesteps, device):
    '''
    SIGMA_LAST is "SIGMA_MIN" here
    '''
    self.num_inference_steps = len(timesteps)
    if not isinstance(timesteps, torch.Tensor): timesteps = torch.tensor([i for i in timesteps])
    self.timesteps = timesteps.float().to(device)

    # set noise schedule according to timesteps
    sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()
    N = sigmas.shape[0]

    idx_lower = timesteps.floor().long() # lower index
    idx_upper = idx_lower + 1            # upper index
    idx_upper = torch.where(idx_upper >= N, idx_lower, idx_upper)
    w = timesteps - idx_lower.to(timesteps.dtype)
    sigma_interp = sigmas[idx_lower] * (1 - w) + sigmas[idx_upper] * w

    sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]).sqrt().unsqueeze(0)
    sigmas_final = torch.cat([sigma_interp, sigma_last]).to(torch.float32)
    self.sigmas = sigmas_final
    self.num_inference_steps = len(timesteps)

    # reset step_index counter
    self.step_index = 0
    self.model_outputs = [None]
  
  def sigma_to_alpha_t(self, sigma): # returns sqrt(alpha_t)
    return  1 / ((sigma**2 + 1) ** 0.5)