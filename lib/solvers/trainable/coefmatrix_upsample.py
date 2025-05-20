from lib.registries import solver_registry
from lib.solvers.trainable.set_solver import SETSOLVER
import torch, math

@solver_registry.add_to_registry("UP41")
class COEFUP41:
  order = 10
  def _step(self, model_output, sample, solver_pred, step_index, **kwargs):
    if step_index == 0: 
      self.train_model_outputs = []
      self.prev_xt = []

    self.train_model_outputs = [model_output] + self.train_model_outputs[:step_index] 
    
    deltas = self.train_params[step_index][:step_index+2]
    prev_sample  = sum([i * j for i, j in zip(deltas, [sample] + self.train_model_outputs)])

    deltas1 = self.train_params[step_index][step_index+2:]
    prev_sample += sum([i * j for i, j in zip(deltas1, self.prev_xt)])

    prev_sample = prev_sample + solver_pred

    self.prev_xt = [prev_sample] + self.prev_xt

    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)

    self.train_params = []
    for i in range(num_inference_steps):
      self.train_params.append(torch.zeros(2*i+2, dtype=torch.float32, requires_grad=True))




@solver_registry.add_to_registry("COEF41UP")
class COEFUP41(SETSOLVER):
  order = 3
  is_trainable = True
  rank = 1
  dim_upsample = 4
  multiply_operator = "*"
  set_sovler = 'DEIS'
  def step(self, model_output, sample=None, **kwargs):
    latent_dims = kwargs['latent_dims']
    self.model_outputs = [i for i in self.model_outputs if i is not None]
    self.model_outputs = [model_output] + self.model_outputs[:2]

    A, B = self.train_params
    A = A[self.step_index] # 4 x 4 x dim x rank
    B = B[self.step_index] # 4 x 4 x rank x dim
    deltas = [a @ b for a,b in zip(A, B)] # 4 x 4 x dim x dim

    if self.multiply_operator == "@":
      identity = torch.eye(latent_dims[-1], device=A.device, dtype=model_output.dtype).unsqueeze(0).expand(4, -1, -1)
    elif self.multiply_operator == "*":
      identity = torch.ones(*latent_dims, device=A.device, dtype=model_output.dtype)
    
    # upsample deltas to 4 x 128 x 128:
    deltas = [(identity+self.upsample(d,latent_dims)).to(model_output.device) for d in deltas][:self.step_index+2]

    c_x, c_eps, c_eps1, c_eps2 = self.get_coeffs(self.set_sovler)
    solv_coeffs = [c_x, c_eps, c_eps1, c_eps2][:self.step_index+2]
    coeffs = [i * j for i, j in zip(solv_coeffs, deltas)]

    if self.multiply_operator == "@":
      prev_sample  = sum([i @ j for i, j in zip(coeffs, [sample] + self.model_outputs)])
    elif self.multiply_operator == "*":
      prev_sample  = sum([i * j for i, j in zip(coeffs, [sample] + self.model_outputs)])

    self.step_index += 1
    return prev_sample
  
  def upsample(self, delta, latent_dims):
    '''input of shape 4 x dim x dim'''
    delta = delta.unsqueeze(1)
    delta_upsample = torch.nn.functional.interpolate(delta, size=(latent_dims[-2], latent_dims[-1]), mode='bilinear', align_corners=False)
    return delta_upsample.squeeze(1)

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)
    
    batch_dims = (num_inference_steps, 4, 4)
    A = torch.empty(*batch_dims, self.dim_upsample, self.rank)
    B = torch.empty(*batch_dims, self.rank, self.dim_upsample)

    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    torch.nn.init.zeros_(B)
    A = torch.nn.Parameter(A, requires_grad=True)
    B = torch.nn.Parameter(B, requires_grad=True)
    self.train_params = [A, B]


@solver_registry.add_to_registry("COEF44UP")
class COEFUP44(COEFUP41):
  order = 3
  is_trainable = True
  rank = 4
  multiply_operator = "*"


@solver_registry.add_to_registry("COEF41UP_")
class COEFUP41_(COEFUP41):
  order = 3
  is_trainable = True
  rank = 1
  multiply_operator = "@"
@solver_registry.add_to_registry("COEF44UP_")
class COEFUP44_(COEFUP41):
  order = 3
  is_trainable = True
  rank = 4
  multiply_operator = "@"
