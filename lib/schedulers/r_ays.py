from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import torch
import numpy as np

@scheduler_registry.add_to_registry("AYS")
class AYSScheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
    timesteps = self._get_ays_timesteps_ts(num_inference_steps)
    self.prepare_solver_data(timesteps, device)

  def _get_ays_timesteps_ts(self, num_inference_steps):
    ays_ts_10 = np.array([999, 845, 730, 587, 443, 310, 193, 116, 53, 13])
    new_ts = self._loglinear_interp(ays_ts_10, num_inference_steps)
    return new_ts.round().astype(int)

  def _loglinear_interp(self, t_steps, num_steps):
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys