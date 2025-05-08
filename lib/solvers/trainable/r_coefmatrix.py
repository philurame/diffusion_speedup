from lib.registries import solver_registry
from lib.solvers.trainable.set_solver import SETSOLVER
import torch

@solver_registry.add_to_registry("COEFMATRIX1")
class COEFMATRIX1(SETSOLVER):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    train_params = self.train_params[self.step_index].to(model_output.device)
    deltas = [i[0].view(4, 128, -1) @ i[1].view(4, -1, 128) for i in train_params]

    deis_coeffs = self.deis_coeffs[self.step_index]
    coeffs = [delta * coeff for delta, coeff in zip(deltas, deis_coeffs)]

    prev_sample = sum([i * j for i, j in zip(coeffs, self.model_outputs)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    rank = 1
    init = torch.full((num_inference_steps, 4, 2, 4, rank, 128), 1/rank**.5)
    init += torch.randn(num_inference_steps, 4, 2, 4, rank, 128, dtype=torch.float32, generator=torch.Generator('cpu').manual_seed(0)) / 1e5
    self.train_params = torch.nn.Parameter(init, requires_grad=True)


@solver_registry.add_to_registry("COEFMATRIX2")
class COEFMATRIX2(COEFMATRIX1):
  order = 3
  is_trainable = True
  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    rank = 2
    init = torch.full((num_inference_steps, 4, 2, 4, rank, 128), 1/rank**.5)
    init += torch.randn(num_inference_steps, 4, 2, 4, rank, 128, dtype=torch.float32, generator=torch.Generator('cpu').manual_seed(0)) / 1e5
    self.train_params = torch.nn.Parameter(init, requires_grad=True)



@solver_registry.add_to_registry("COEFMATRIXSHARED1")
class COEFMATRSHARED1(SETSOLVER):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    matrices, coeffs = self.train_params
    matrices = [(i[0].view(4, 128, -1) @ i[1].view(4, -1, 128)).to(model_output.device) for i in matrices]
    coeffs = coeffs[self.step_index]

    deis_coeffs = self.deis_coeffs[self.step_index]
    coeffs = [delta * coeff for delta, coeff in zip(coeffs, deis_coeffs)]

    prev_sample = sum([i * j * m for i, j, m in zip(coeffs, self.model_outputs, matrices)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    rank = 1
    matrix = torch.full((4, 2, 4, rank, 128), 1/rank**.5)
    matrix += torch.randn(4, 2, 4, rank, 128, dtype=torch.float32, generator=torch.Generator('cpu').manual_seed(0)) / 1e5
    matrix = torch.nn.Parameter(matrix, requires_grad=True)
    coeffs = torch.nn.Parameter(torch.ones(num_inference_steps, 4, dtype=torch.float32, requires_grad=True))
    self.train_params = [matrix, coeffs]


@solver_registry.add_to_registry("COEFMATRIXSHARED2")
class COEFMATRSHARED2(COEFMATRSHARED1):
  order = 3
  is_trainable = True

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    rank = 2
    matrix = torch.full((4, 2, 4, rank, 128), 1/rank**.5)
    matrix += torch.randn(4, 2, 4, rank, 128, dtype=torch.float32, generator=torch.Generator('cpu').manual_seed(0)) / 1e5
    matrix = torch.nn.Parameter(matrix, requires_grad=True)
    coeffs = torch.nn.Parameter(torch.ones(num_inference_steps, 4, dtype=torch.float32, requires_grad=True))
    self.train_params = [matrix, coeffs]