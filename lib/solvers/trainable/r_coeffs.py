from lib.registries import solver_registry
import torch

@solver_registry.add_to_registry("COEF")
class COEFSolver:
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    deltas      = self.train_params[self.step_index]
    deis_coeffs = self.deis_coeffs[self.step_index]
    coeffs = deis_coeffs + deltas

    prev_sample = sum([i * j for i, j in zip(coeffs, self.model_outputs)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps):
    '''sets 2 order lin_coeffs according to DEIS LINEAR'''

    timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1).round().flip(0)[:-1]
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
        self.deis_coeffs.append([(alpha_t/alpha_s).item(), 0, 0, (-alpha_t * (sigma_s - sigma_t)).item()])
        continue

      def ind_fn(t, b, c):
        return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

      sigma_ss = sigmas[step_index - 1]
      coef1 = ind_fn(sigma_t, sigma_s, sigma_ss) - ind_fn(sigma_s, sigma_s, sigma_ss)
      coef2 = ind_fn(sigma_t, sigma_ss, sigma_s) - ind_fn(sigma_s, sigma_ss, sigma_s)

      self.deis_coeffs.append([(alpha_t/alpha_s).item(), 0, (alpha_t*coef2).item(), (alpha_t*coef1).item()])
    
    self.deis_coeffs  = torch.torch.tensor(self.deis_coeffs, dtype=torch.float32)
    self.train_params = torch.nn.Parameter(torch.zeros(num_inference_steps, 4, dtype=torch.float32, requires_grad=True))