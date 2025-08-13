from registries import solver_registry
from lib.solvers.mixin_solver import SolverMixin
import torch

@solver_registry.add_to_registry('DPMS')
class DPMS(SolverMixin):
  order = 2
  is_trainable = False
  def step(self, model_output, sample=None, **kwargs):
    prediction_type = kwargs.get("prediction_type", "epsilon")
    model_output = self.epsilon_output(model_output, sample, prediction_type)
    
    model_output = self.convert_to_data_prediction(model_output, sample=sample)
    self.model_outputs = self.model_outputs[-1:] + [model_output]

    sample = sample.to(torch.float32)
    
    if self.step_index == 0 or self.step_index == len(self.timesteps) - 1:
      prev_sample = self.first_order_update(model_output, sample=sample)
    else:
      prev_sample = self.second_order_update(model_output, sample=sample)

    self.step_index += 1
    return prev_sample.to(model_output.dtype)
  
  def convert_to_data_prediction(self, model_output, sample):
    sigma = self.sigmas[self.step_index]
    alpha_t = self.sigma_to_alpha_t(sigma)
    x0_pred = sample / alpha_t - sigma * model_output
    return x0_pred

  def first_order_update(self, model_output, sample):
    '''DDIM step in data prediction'''
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    sigma_t, sigma_s = sigma_t * alpha_t, sigma_s * alpha_s

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    return x_t

  def second_order_update(self, model_output, sample):
    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )
    alpha_t, alpha_s0, alpha_s1 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
      self.sigma_to_alpha_t(sigma_s1),
    )
    sigma_t, sigma_s0, sigma_s1 = sigma_t * alpha_t, sigma_s0 * alpha_s0, sigma_s1 * alpha_s1

    lambda_t, lambda_s0, lambda_s1 = (
      torch.log(alpha_t) - torch.log(sigma_t),
      torch.log(alpha_s0) - torch.log(sigma_s0),
      torch.log(alpha_s1) - torch.log(sigma_s1),
    )
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h

    m0, m1 = self.model_outputs[-1], self.model_outputs[-2]
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)

    x_t = (
      (sigma_t / sigma_s0) * sample
      - (alpha_t * (torch.exp(-h) - 1.0)) * D0
      - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )
    return x_t


@solver_registry.add_to_registry('DPMS2')
class DPMS2(DPMS): pass


@solver_registry.add_to_registry('DPMS3')
class DPMS3:
  order = 3
  is_trainable = False
  def step(self, model_output, sample=None, **kwargs):
    model_output = self.convert_to_data_prediction(model_output, sample=sample)
    self.model_outputs = self.model_outputs[-2:] + [model_output]

    sample = sample.to(torch.float32)
    
    if self.step_index == 0 or self.step_index == len(self.timesteps) - 1:
      prev_sample = self.first_order_update(model_output, sample=sample)
    elif self.step_index == 1:
      prev_sample = self.second_order_update(model_output, sample=sample)
    else:
      prev_sample = self.third_order_update(model_output, sample=sample)

    self.step_index += 1
    return prev_sample.to(model_output.dtype)
  
  def convert_to_data_prediction(self, model_output, sample):
    sigma = self.sigmas[self.step_index]
    alpha_t = self.sigma_to_alpha_t(sigma)
    x0_pred = sample / alpha_t - sigma * model_output
    return x0_pred

  def first_order_update(self, model_output, sample):
    '''DDIM step in data prediction'''
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    sigma_t, sigma_s = sigma_t * alpha_t, sigma_s * alpha_s

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    return x_t

  def second_order_update(self, model_output, sample):
    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )
    alpha_t, alpha_s0, alpha_s1 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
      self.sigma_to_alpha_t(sigma_s1),
    )
    sigma_t, sigma_s0, sigma_s1 = sigma_t * alpha_t, sigma_s0 * alpha_s0, sigma_s1 * alpha_s1

    lambda_t, lambda_s0, lambda_s1 = (
      torch.log(alpha_t) - torch.log(sigma_t),
      torch.log(alpha_s0) - torch.log(sigma_s0),
      torch.log(alpha_s1) - torch.log(sigma_s1),
    )
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h

    m0, m1 = self.model_outputs[-1], self.model_outputs[-2]
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)

    x_t = (
      (sigma_t / sigma_s0) * sample
      - (alpha_t * (torch.exp(-h) - 1.0)) * D0
      - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )
    return x_t
  
  def third_order_update(self, model_output, sample):
    sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
      self.sigmas[self.step_index - 2],
    )
    alpha_t, alpha_s0, alpha_s1, alpha_s2 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
      self.sigma_to_alpha_t(sigma_s1),
      self.sigma_to_alpha_t(sigma_s2),
    )

    sigma_t, sigma_s0, sigma_s1, sigma_s2 = sigma_t * alpha_t, sigma_s0 * alpha_s0, sigma_s1 * alpha_s1, sigma_s2 * alpha_s2

    lambda_t  = torch.log(alpha_t)  - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

    m0, m1, m2 = self.model_outputs[-1], self.model_outputs[-2], self.model_outputs[-3]

    h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
    r0, r1 = h_0 / h, h_1 / h
    D0 = m0
    D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

    x_t = (
      (sigma_t / sigma_s0) * sample
      - (alpha_t * (torch.exp(-h) - 1.0)) * D0
      + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
      - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
    )
    return x_t