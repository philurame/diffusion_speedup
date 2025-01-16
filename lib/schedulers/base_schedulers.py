import numpy as np
from registries import scheduler_registry
from lib.schedulers.mixin_scheduler import BaseSchedulerMixin

@scheduler_registry.add_to_registry("LINEAR")
class LINEARScheduler(BaseSchedulerMixin):
   def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    timesteps = (
      np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1) # why prevent ts to be zero?
      .round()[::-1][:-1]
      .copy()
      .astype(np.int64)
      )
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("DDIM")
class DDIMScheduler(BaseSchedulerMixin):
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    step_ratio = 1000 // num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("KARRAS")
class KARRASScheduler(BaseSchedulerMixin):
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    sigmas = np.flip(sigmas).copy()
    sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("SNR")
class SNRScheduler(BaseSchedulerMixin):
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    lambdas = np.flip(log_sigmas.copy())
    lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
    sigmas = np.exp(lambdas)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("AYS")
class AYSScheduler(BaseSchedulerMixin):
  def _loglinear_interp(self, t_steps, num_steps):
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys
  
  def _get_ays_timesteps_ts(self, num_inference_steps):
    ays_ts_10 = np.array([999, 845, 730, 587, 443, 310, 193, 116, 53, 13])
    new_ts = self._loglinear_interp(ays_ts_10, num_inference_steps)
    return new_ts.round().astype(int)
  
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    timesteps = self._get_ays_timesteps_ts(num_inference_steps)
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)