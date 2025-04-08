from lib.registries import solver_registry
from lib.solvers.trainable.set_deis import SETDEIS2
import torch, math

@solver_registry.add_to_registry("COEFUP8|1")
class COEFUP81(SETDEIS2):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    A, B = self.train_params
    A = A[self.step_index] # 4 x 4 x dim x rank
    B = B[self.step_index] # 4 x 4 x rank x dim
    deltas = [a @ b for a,b in zip(A, B)] # 4 x 4 x dim x dim

    # upsample deltas to 4 x 4 x 128 x 128:
    identity = torch.eye(128, device=A.device).unsqueeze(0).expand(4, -1, -1)
    deltas = [(identity+self.upsample(d)).to(model_output.device) for d in deltas]

    deis_coeffs = self.deis_coeffs[self.step_index].to(model_output.device)

    coeffs = [delta * coeff for delta, coeff in zip(deltas, deis_coeffs)]

    prev_sample = sum([i @ j if isinstance(j, torch.Tensor) else 0. for i, j in zip(coeffs, self.model_outputs)])

    self.step_index += 1
    return prev_sample
  
  def upsample(self, delta):
    '''input of shape 4 x dim x dim'''
    delta = delta.unsqueeze(1)
    delta_upsample = torch.nn.functional.interpolate(delta, size=(128, 128), mode='bilinear', align_corners=False)
    return delta_upsample.squeeze(1)

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    self.dim = 8
    rank = 1
    
    batch_dims = (num_inference_steps, 4, 4)
    A = torch.empty(*batch_dims, self.dim, rank)
    B = torch.empty(*batch_dims, rank, self.dim)

    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    torch.nn.init.zeros_(B)
    A = torch.nn.Parameter(A, requires_grad=True)
    B = torch.nn.Parameter(B, requires_grad=True)
    self.train_params = [A, B]


@solver_registry.add_to_registry("COEFUP8|2")
class COEFUP82(COEFUP81):
  order = 3
  is_trainable = True
  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    self.dim = 8
    rank = 2
    
    batch_dims = (num_inference_steps, 4, 4)
    A = torch.empty(*batch_dims, self.dim, rank)
    B = torch.empty(*batch_dims, rank, self.dim)
    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    torch.nn.init.zeros_(B)
    A = torch.nn.Parameter(A, requires_grad=True)
    B = torch.nn.Parameter(B, requires_grad=True)
    self.train_params = [A, B]
    


@solver_registry.add_to_registry("COEFUP*8|1")
class COEFUP81_(COEFUP81):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [0., 0.] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    A, B = self.train_params
    A = A[self.step_index] # 4 x 4 x dim x rank
    B = B[self.step_index] # 4 x 4 x rank x dim
    deltas = [a @ b for a,b in zip(A, B)] # 4 x 4 x dim x dim

    # upsample deltas to 4 x (4 x 128 x 128):
    identity = torch.ones(4, 128, 128)
    deltas = [(identity+self.upsample(d)).to(model_output.device) for d in deltas]

    deis_coeffs = self.deis_coeffs[self.step_index].to(model_output.device)

    coeffs = [delta * coeff for delta, coeff in zip(deltas, deis_coeffs)]

    prev_sample = sum([i * j if isinstance(j, torch.Tensor) else 0. for i, j in zip(coeffs, self.model_outputs)])

    self.step_index += 1
    return prev_sample



@solver_registry.add_to_registry("COEFUP*8|2")
class COEFUP82_(COEFUP81_):
  order = 3
  is_trainable = True
  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    self.set_deis_coeffs(num_inference_steps, timesteps)
    num_inference_steps = num_inference_steps or len(timesteps)
    self.dim = 8
    rank = 2
    
    batch_dims = (num_inference_steps, 4, 4)
    A = torch.empty(*batch_dims, self.dim, rank)
    B = torch.empty(*batch_dims, rank, self.dim)
    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    torch.nn.init.zeros_(B)
    A = torch.nn.Parameter(A, requires_grad=True)
    B = torch.nn.Parameter(B, requires_grad=True)
    self.train_params = [A, B]