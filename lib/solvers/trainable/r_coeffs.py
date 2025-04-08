from lib.registries import solver_registry
from lib.solvers.trainable.set_deis import SETDEIS2
import torch

@solver_registry.add_to_registry("COEF")
class COEFSolver(SETDEIS2):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [i for i in self.model_outputs if i is not None]
    self.model_outputs = [model_output] + self.model_outputs[:2]

    c_x, c_eps, c_eps1 = self.get_deis_coeffs()
    deltas = self.train_params[self.step_index][:-1 if self.step_index == 0 else None]

    deis_coeffs = [c_x, c_eps, c_eps1, 0][:self.step_index+2]
    coeffs = [i + j for i, j in zip(deis_coeffs, deltas)]
    prev_sample = sum([i * j for i, j in zip(coeffs, [sample] + self.model_outputs)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)
    self.train_params = torch.nn.Parameter(torch.zeros(num_inference_steps, 4, dtype=torch.float32, requires_grad=True))



@solver_registry.add_to_registry("_COEF10")
class COEFSolverTest10(COEFSolver):
  order = 3
  is_trainable = True
  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)
    init = torch.randn(num_inference_steps, 4, dtype=torch.float32, generator=torch.Generator().manual_seed(0)) / 10
    self.train_params = torch.nn.Parameter(init, requires_grad=True)


@solver_registry.add_to_registry("COEFEXT")
class COEFEXT(SETDEIS2):
  order = 10
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [model_output] + self.model_outputs[:self.step_index]

    c_x, c_eps, c_eps1 = self.get_deis_coeffs()
    deis_coeffs = ([c_x, c_eps, c_eps1] + [0.]*self.step_index)[:self.step_index+2]
    deltas = self.train_params[self.step_index]

    coeffs = [i + j for i, j in zip(deis_coeffs, deltas)]
    prev_sample = sum([i * j for i, j in zip(coeffs, [sample] + self.model_outputs)])

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)
    self.train_params = []
    for i in range(num_inference_steps):
      self.train_params.append(torch.zeros(i+2, dtype=torch.float32, requires_grad=True))


@solver_registry.add_to_registry("COEFEXT2")
class COEFEXT2(SETDEIS2):
  order = 10
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [model_output] + self.model_outputs[:self.step_index] 
    if self.step_index == 0:
      self.prev_xt = []

    c_x, c_eps, c_eps1 = self.get_deis_coeffs()
    deis_coeffs = ([c_x, c_eps, c_eps1] + [0.]*self.step_index)[:self.step_index+2]
    deltas = self.train_params[self.step_index][:self.step_index+2]

    coeffs = [i + j for i, j in zip(deis_coeffs, deltas)]
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



@solver_registry.add_to_registry("COEFEXTF")
class COEFEXTF(SETDEIS2):
  order = 10
  is_trainable = True
  fourier_crop = 50
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [model_output] + self.model_outputs[:self.step_index]

    c_x, c_eps, c_eps1 = self.get_deis_coeffs()
    deis_coeffs = ([c_x, c_eps, c_eps1] + [0.]*self.step_index)[:self.step_index+2]
    deltas = self.train_params[self.step_index]

    coeffs = [i + j for i, j in zip(deis_coeffs, deltas[:self.step_index+2])]
    prev_sample = sum([i * j for i, j in zip(coeffs, [sample] + self.model_outputs)])

    fx = self.fourier(sample)
    prev_sample += deltas[-1]*fx

    self.step_index += 1
    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)
    self.train_params = []
    for i in range(num_inference_steps):
      self.train_params.append(torch.zeros(i+3, dtype=torch.float32, requires_grad=True))
  
  def fourier(self, x):
    img_dims = len(x.shape)-2, len(x.shape)-1
    freq = torch.fft.fft2(x)
    freq = torch.fft.fftshift(freq, dim=img_dims)
    c_x, c_y = x.shape[-2] // 2, x.shape[-1] // 2
    
    freq_center = torch.zeros_like(freq)
    d = self.fourier_crop
    freq_center[..., c_x - d: c_x + d, c_y - d:c_y + d] = freq[..., c_x - d: c_x + d, c_y - d:c_y + d]
    freq_resid = freq - freq_center

    freq_resid  = torch.fft.ifftshift(freq_resid,  dim=img_dims)
    x_resid  = torch.real(torch.fft.ifft2(freq_resid))
    return x_resid