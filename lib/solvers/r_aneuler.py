from lib.registries import solver_registry
import torch

@solver_registry.add_to_registry("ANEULER")
class ANEULER:
  order = 1
  is_trainable = False
  def step(self, model_output, sample=None, generator=None, **kwargs):
    # "scale"
    sigma  = self.sigmas[self.step_index]
    sample = sample * ((sigma**2 + 1) ** 0.5)

    sigma_from = self.sigmas[self.step_index]
    sigma_to   = self.sigmas[self.step_index + 1] if self.step_index != len(self.timesteps) - 1 else 0.
    sigma_up   = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator).to(model_output.device)
    prev_sample = sample + model_output * (sigma_down - sigma_from) + noise * sigma_up

    # "descale"
    prev_sample = prev_sample / ((sigma_to**2 + 1) ** 0.5)

    self.step_index += 1
    return prev_sample