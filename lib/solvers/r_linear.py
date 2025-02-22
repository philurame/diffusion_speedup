from lib.registries import solver_registry

@solver_registry.add_to_registry("LIN")
class LinSolver:
  order = 3
  def step(self, model_output, timestep=None, sample=None):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]
    lin_coeffs = self.lin_coeffs[self.step_index]

    prev_sample = sum([i * j for i, j in zip(lin_coeffs, self.model_outputs)])

    self.step_index += 1
    return prev_sample
