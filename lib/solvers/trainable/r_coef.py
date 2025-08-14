import torch
from registries import solver_registry
from lib.solvers.mixin_solver import SolverMixin
from torch.nn.utils import parameters_to_vector, vector_to_parameters


@solver_registry.add_to_registry("COEF")
class COEF(SolverMixin):
  order = 3
  def _step(self, model_output, sample, solver_pred, step_index, **kwargs):
    prediction_type = kwargs.get("prediction_type", "epsilon")
    model_output = self.epsilon_output(model_output, sample, prediction_type)
    if step_index == 0: 
      self.train_model_outputs = []

    self.train_model_outputs = [model_output] + self.train_model_outputs[:self.order-1] 
    
    num_params = min(self.order+1, step_index+2) # sample_coef + \eps coefs = 1+order
    deltas = self.train_params[step_index][:num_params]
    prev_sample  = sum([i * j for i, j in zip(deltas, [sample] + self.train_model_outputs)])

    prev_sample = prev_sample + solver_pred
    return prev_sample


  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)

    self.train_params = []
    for step_index in range(num_inference_steps):
      num_params = min(self.order+1, step_index+2) # sample_coef + \eps coefs = 1+order
      self.train_params.append(torch.zeros(num_params, dtype=torch.float32, requires_grad=True))

  def get_flat_params(self) -> torch.Tensor:
    return parameters_to_vector(self.train_params)
  
  def set_flat_params(self, flat_params: torch.Tensor) -> None:
    vector_to_parameters(flat_params, self.train_params)



solver_registry.add_to_registry("COEF-REG")
class COEFREG(COEF):
  def _step(self, model_output, sample, solver_pred, step_index, **kwargs):
    prediction_type = kwargs.get("prediction_type", "epsilon")
    model_output = self.epsilon_output(model_output, sample, prediction_type)
    if step_index == 0: 
      self.train_model_outputs = []
    
    self.train_model_outputs = [model_output] + self.train_model_outputs[:self.order-1] 
    
    num_params = min(self.order+1, step_index+2) # sample_coef + \eps coefs = 1+order
    deltas = self.train_params[step_index][:num_params]

    # normalized_outputs = [sample / torch.norm(sample)] + [eps/torch.norm(eps) for i, eps in enumerate(self.train_model_outputs)]
    normalized_outputs = [self._rms_norm(outp) for outp in [sample] + self.train_model_outputs]
    prev_sample  = sum([i * j for i, j in zip(deltas, normalized_outputs)])

    prev_sample = prev_sample + solver_pred
    return prev_sample

  def _rms_norm(self, x, smin=1e-4, smax=1e4):
    s = x.detach().to(torch.float32).pow(2).mean().sqrt()
    s = s.clamp(min=smin, max=smax)
    return x / s