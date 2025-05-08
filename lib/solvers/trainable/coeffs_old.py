from lib.registries import solver_registry
from lib.solvers.trainable.set_solver import SETSOLVER
import torch





@solver_registry.add_to_registry("COEFEXT")
class COEFEXT(SETSOLVER):
  order = 10
  is_trainable = True
  set_sovler = 'DEIS'
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [model_output] + self.model_outputs[:self.step_index]

    c_x, c_eps, c_eps1, c_eps2 = self.get_coeffs(self.set_sovler)
    solv_coeffs = ([c_x, c_eps, c_eps1, c_eps2] + [0.]*self.step_index)[:self.step_index+2]
    deltas = self.train_params[self.step_index]

    coeffs = [i + j for i, j in zip(solv_coeffs, deltas)]
    prev_sample = sum([i * j for i, j in zip(coeffs, [sample] + self.model_outputs)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)
    self.train_params = []
    for i in range(num_inference_steps):
      self.train_params.append(torch.zeros(i+2, dtype=torch.float32, requires_grad=True))


@solver_registry.add_to_registry("COEFEXT2")
class COEFEXT2(SETSOLVER):
  order = 10
  is_trainable = True
  set_sovler = 'DEIS'
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [model_output] + self.model_outputs[:self.step_index] 
    if self.step_index == 0:
      self.prev_xt = []

    c_x, c_eps, c_eps1, c_eps2 = self.get_coeffs(self.set_sovler)
    solv_coeffs = ([c_x, c_eps, c_eps1, c_eps2] + [0.]*self.step_index)[:self.step_index+2]
    deltas = self.train_params[self.step_index][:self.step_index+2]

    coeffs = [i + j for i, j in zip(solv_coeffs, deltas)]
    prev_sample  = sum([i * j for i, j in zip(coeffs, [sample] + self.model_outputs)])

    deltas1 = self.train_params[self.step_index][self.step_index+2:]
    prev_sample += sum([i * j for i, j in zip(deltas1, self.prev_xt)])

    self.prev_xt = [prev_sample] + self.prev_xt

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)

    self.train_params = []
    for i in range(num_inference_steps):
      self.train_params.append(torch.zeros(2*i+2, dtype=torch.float32, requires_grad=True))

