from lib.registries import solver_registry

@solver_registry.add_to_registry("LIN")
class LinSolver:
  order = 2
  def step(self, model_output, timestep=None, sample=None):
    self.model_outputs = self.model_outputs[-1:] + [model_output]
    lin_coeffs = self.lin_coeffs[self.step_index]

    if lin_coeffs[0] is None:
      sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index],
      alpha_t, alpha_s0 = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s0)
      lin_coeffs[0] = (alpha_t/alpha_s0).item()
    lin_coeffs = [i if i is not None else 0 for i in lin_coeffs]

    prev_sample = lin_coeffs[0] * sample + lin_coeffs[1] * self.model_outputs[0] + lin_coeffs[2] * self.model_outputs[1]

    self.step_index += 1
    return prev_sample
