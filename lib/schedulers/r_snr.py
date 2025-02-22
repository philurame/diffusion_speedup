from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import numpy as np

@scheduler_registry.add_to_registry("SNR")
class SNRScheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    lambdas = np.flip(log_sigmas.copy())
    lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
    sigmas = np.exp(lambdas)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
    self.prepare_solver_data(timesteps, device, sigmas=sigmas)

  def _convert_to_lu(self, in_lambdas, num_inference_steps):
    lambda_min = in_lambdas[-1].item()
    lambda_max = in_lambdas[0].item()
    ramp = np.linspace(0, 1, num_inference_steps)
    lambdas = (lambda_max + ramp * (lambda_min - lambda_max))
    return lambdas
  
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