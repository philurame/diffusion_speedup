from registries import solver_registry
import torch

class FLOW:
  order = 1
  int_fn = 'linear'
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = self.model_outputs[-1:] + [model_output]
    sample = sample.to(torch.float32)

    if self.order == 1 or self.step_index == 0: # or self.step_index == len(self.timesteps) - 1:
      prev_sample = self.first_order_update(model_output, sample=sample)
    else:
      prev_sample = self.second_order_update(model_output, sample=sample)

    self.step_index += 1
    return prev_sample.to(model_output.dtype)

  def int_lin(self, x, b, c):
    '''calculates indefinite integral (x-c)/(b-c)dx'''
    return (x-c)**2 / (b-c) / 2
  
  def int_log(self, x, b, c):
    '''calculates indefinite integral (lnx - lnc)/(lnb - lnc)dx'''
    return x * (-torch.log(c) + torch.log(x) - 1) / (torch.log(b) - torch.log(c))

  
  def first_order_update(self, model_output, sample):
    '''Euler step'''
    dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
    prev_sample = sample + dt * model_output
    return prev_sample

  def second_order_update(self, model_output, sample):
    sigma_t, sigma_s, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )

    u_s, u_s1 = self.model_outputs[-1], self.model_outputs[-2]

    if self.int_fn == 'linear' or self.step_index == len(self.timesteps) - 1:
      int_fn = self.int_lin
    elif self.int_fn == 'log':
      int_fn = self.int_log

    coef_s  = int_fn(sigma_t, sigma_s, sigma_s1) - int_fn(sigma_s, sigma_s, sigma_s1)
    coef_s1 = int_fn(sigma_t, sigma_s1, sigma_s) - int_fn(sigma_s, sigma_s1, sigma_s)
    
    x_t = sample + coef_s * u_s + coef_s1 * u_s1
    return x_t
  

@solver_registry.add_to_registry("FLOW")
class FLOW1(FLOW):
  is_trainable = False
  order = 1

@solver_registry.add_to_registry("FLOW2")
class FLOW2(FLOW):
  is_trainable = False
  order = 2
  int_fn = 'linear'

@solver_registry.add_to_registry("FLOW2-LOG")
class FLOW2LOG(FLOW):
  is_trainable = False
  order = 2
  int_fn = 'log'