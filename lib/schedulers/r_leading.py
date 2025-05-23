from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import numpy as np

@scheduler_registry.add_to_registry("LEADING")
class LEADINGScheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    ratio = 1000 // num_inference_steps 
    timesteps = (np.arange(0, num_inference_steps) * ratio).round()[::-1].copy().astype(np.float32)
    self.prepare_solver_data(timesteps, device)
