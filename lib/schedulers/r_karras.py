from registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import numpy as np

@scheduler_registry.add_to_registry("KARRAS")
class KARRASScheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    sigmas = np.flip(sigmas).copy()
    sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
    self.prepare_solver_data(timesteps, device, sigmas=sigmas)
  
  def _convert_to_karras(self, in_sigmas, num_inference_steps):
    sigma_min = in_sigmas[-1].item()
    sigma_max = in_sigmas[0].item()
    rho = 7.0  # 7.0 is the value used in the paper
    ramp = np.linspace(0, 1, num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
  
  def _sigma_to_t(self, sigma, log_sigmas):
    log_sigma = np.log(np.maximum(sigma, 1e-10))
    dists = log_sigma - log_sigmas[:, np.newaxis]
    low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1
    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]
    w = (low - log_sigma) / (low - high)
    w = np.clip(w, 0, 1)
    t = (1 - w) * low_idx + w * high_idx
    t = t.reshape(sigma.shape)
    return t