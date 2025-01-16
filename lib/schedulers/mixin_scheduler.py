import numpy as np
import torch

class BaseSchedulerMixin:
  def _set_timesteps_common(self, sigmas, timesteps, device):
    if self.config["final_sigmas_type"] == "sigma_min":
      sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
    elif self.config["final_sigmas_type"] == "zero":
      sigma_last = 0
    else:
      raise ValueError(
        f"`final_sigmas_type` must be one of 'zero' or 'sigma_min', but got {self.config.final_sigmas_type}"
      )
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas).to("cpu")
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
    self.num_inference_steps = len(timesteps)
    self.model_outputs = [None] * self.config.get('solver_order', 1)
    self.lower_order_nums = 0
    self._step_index = None
    self._begin_index = None