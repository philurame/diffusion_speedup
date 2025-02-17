from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import torch

@scheduler_registry.add_to_registry("CUSTOM")
class CustomScheduler(SchedulerMixin):
   def set_timesteps(self, timesteps, device=None, **kwargs):
    if not isinstance(timesteps, torch.Tensor): timesteps = torch.tensor(timesteps)
    self.prepare_solver_data(timesteps, device) 