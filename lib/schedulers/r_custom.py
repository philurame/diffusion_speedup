from registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import torch

@scheduler_registry.add_to_registry("CUSTOM")
class CustomScheduler(SchedulerMixin):
   def set_timesteps(self, timesteps, sigmas=None, device=None, **kwargs):
    if sigmas is not None and not isinstance(sigmas, torch.Tensor): sigmas = torch.tensor(sigmas)
    self.prepare_solver_data(timesteps, device, sigmas=sigmas) 
    
    