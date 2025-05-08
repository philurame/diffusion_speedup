from lib.registries import solver_registry
from lib.solvers.trainable.set_solver import SETSOLVER
import torch

@solver_registry.add_to_registry("NOISE")
class COEFSolver(SETSOLVER):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]
    self.init_noise = self.init_noise or sample

    # /// TODO!!!!!
    deltas      = self.train_params[self.step_index]
    deis_coeffs = self.deis_coeffs[self.step_index]
    coeffs = deis_coeffs + deltas

    prev_sample = sum([i * j for i, j in zip(coeffs, self.model_outputs)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    self.init_noise = None
    self.train_params = torch.nn.Parameter(torch.zeros(num_inference_steps, 4, dtype=torch.float32, requires_grad=True))