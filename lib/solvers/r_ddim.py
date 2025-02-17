from lib.registries import solver_registry

@solver_registry.add_to_registry("DDIM")
class DDIM:
  order = 1
  def step(self, model_output, t=None, sample=None, **kwargs):
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    prev_sample = (alpha_t / alpha_s) * sample - model_output * alpha_t * (sigma_s - sigma_t)
    self.step_index += 1
    return prev_sample