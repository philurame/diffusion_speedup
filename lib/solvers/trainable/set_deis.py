import torch, pickle

class SETDEIS2:
  def set_deis_coeffs(self, num_inference_steps=None, timesteps=None):
    '''sets 2 order deis_coeffs according to DEIS; coeffs for x_t, eps_{t}, eps_{t+1}'''
    if timesteps is None:
      timesteps = torch.linspace(0, num_train_timesteps - 1, num_inference_steps + 1).round().flip(0)[:-1]

    if not isinstance(timesteps, torch.Tensor):
      timesteps = torch.tensor([i for i in timesteps])
    num_inference_steps = len(timesteps)
    
    sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()
    idx_lower = timesteps.floor().long()
    idx_upper = idx_lower + 1
    idx_upper = torch.where(idx_upper >= sigmas.shape[0], idx_lower, idx_upper)
    w = timesteps - idx_lower.to(timesteps.dtype)
    sigma_interp = sigmas[idx_lower] * (1 - w) + sigmas[idx_upper] * w
    sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]).sqrt().unsqueeze(0)
    sigmas = torch.cat([sigma_interp, sigma_last]).to(torch.float32)

    self.deis_coeffs = []
    for step_index in range(num_inference_steps):
      sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
      alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)

      if step_index == 0:
        self.deis_coeffs.append([(alpha_t/alpha_s).item(), (-alpha_t * (sigma_s - sigma_t)).item(), 0])
        continue

      def ind_fn(t, b, c):
        return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

      sigma_ss = sigmas[step_index - 1]
      coef1 = ind_fn(sigma_t, sigma_s, sigma_ss) - ind_fn(sigma_s, sigma_s, sigma_ss)
      coef2 = ind_fn(sigma_t, sigma_ss, sigma_s) - ind_fn(sigma_s, sigma_ss, sigma_s)
      self.deis_coeffs.append([(alpha_t/alpha_s).item(), (alpha_t*coef2).item(), (alpha_t*coef1).item()])
    self.deis_coeffs = torch.torch.tensor(self.deis_coeffs, dtype=torch.float32)
  
  def get_deis_coeffs(self):
    '''returns coeffs corresponding to x_t, eps_{t}, eps_{t+1}'''
    sigma_t, sigma_s0 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
    )
    alpha_t, alpha_s0 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
    )
    if self.step_index == 0:
      return alpha_t/alpha_s0, -alpha_t * (sigma_s0 - sigma_t), 0

    def ind_fn(t, b, c):
      return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

    sigma_s1 = self.sigmas[self.step_index - 1]
    coef1 = ind_fn(sigma_t, sigma_s0, sigma_s1) - ind_fn(sigma_s0, sigma_s0, sigma_s1)
    coef2 = ind_fn(sigma_t, sigma_s1, sigma_s0) - ind_fn(sigma_s0, sigma_s1, sigma_s0)

    # x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
    return alpha_t/alpha_s0, alpha_t*coef1, alpha_t*coef2
