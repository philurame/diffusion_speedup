from lib.registries import solver_registry
from lib.solvers.trainable.set_solver import SETSOLVER
import torch

@solver_registry.add_to_registry("PMATRIX")
class PMATRIX(SETSOLVER):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    train_params = self.train_params[self.step_index].to(model_output.device)
    deltas = train_params[0].view(4, 128, -1) @ train_params[1].view(4, -1, 128)

    deis_coeffs = self.deis_coeffs[self.step_index]

    prev_sample = sum([i * j for i, j in zip(deis_coeffs, self.model_outputs)]) + deltas

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps, device=None):
    self.set_deis_coeffs(num_inference_steps)
    init = torch.randn(num_inference_steps, 2, 4, 4, 128, dtype=torch.float32, generator=torch.Generator('cpu').manual_seed(0)) / 1000
    self.train_params = torch.nn.Parameter(init, requires_grad=True)