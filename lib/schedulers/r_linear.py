from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import numpy as np

@scheduler_registry.add_to_registry("LINEAR")
class LINEARScheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):    
    timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1).round()[::-1][:-1]
    self.prepare_solver_data(timesteps, device)