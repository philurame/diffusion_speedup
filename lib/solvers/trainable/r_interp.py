from lib.registries import solver_registry
import torch.nn.functional as F
import torch

@solver_registry.add_to_registry('1INTERP')
class INTERP:
  order = 2
  is_trainable=True
  def step(self, model_output, t=None, sample=None, **kwargs):
    self.model_outputs = self.model_outputs[-1:] + [model_output]
    
    if self.step_index == 0:
      prev_sample = self.first_order_update(model_output, sample=sample)
    else:
      prev_sample = self.second_order_update(model_output, sample=sample)

    self.step_index += 1
    return prev_sample
    
  def first_order_update(self, model_output, sample):
    '''DDIM step'''
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    prev_sample = (alpha_t / alpha_s) * sample +  alpha_t * (sigma_t - sigma_s) * model_output
    return prev_sample

  def second_order_update(self, model_output, sample):
    '''DDIM and DEIS interpolation'''
    interp_coeff = F.sigmoid(self.train_params[self.step_index-1])
    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )

    alpha_t, alpha_s0 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
    )

    eps_s0, eps_s1 = self.model_outputs[-1], self.model_outputs[-2]

    const_interp = eps_s0

    deis_interp = \
      (torch.log(sigma_t) - torch.log(sigma_s1)) / (torch.log(sigma_s0) - torch.log(sigma_s1)) * eps_s0 + \
      (torch.log(sigma_t) - torch.log(sigma_s0)) / (torch.log(sigma_s1) - torch.log(sigma_s0)) * eps_s1
    
    interp_mv = (1 - interp_coeff)*const_interp + interp_coeff*deis_interp

    x_t = (alpha_t / alpha_s0) * sample + alpha_t * (sigma_t - sigma_s0) * interp_mv
    return x_t
