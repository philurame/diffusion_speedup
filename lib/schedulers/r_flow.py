from registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import torch

@scheduler_registry.add_to_registry("FLOW")
class FLOWScheduler(SchedulerMixin):
  shift=7
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):    
    def sd3_time_shift(t):
      return (self.shift * t) / (1 + (self.shift - 1) * t)
    sigmas = sd3_time_shift(torch.linspace(1, 0, num_inference_steps + 1))
    self.prepare_solver_data((sigmas[:-1] * 1000), device)


for i in range(1, 14):
  name = f"FLOW{i}"
  attrs = {"shift": i}
  new_class = type(name + "Scheduler" if name == "FLOW" else name, (FLOWScheduler,), attrs)
  scheduler_registry.add_to_registry(name)(new_class)
