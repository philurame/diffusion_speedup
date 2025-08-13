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

    self.train_model_outputs = [model_output] + self.train_model_outputs[:2] 
    
    deltas = self.train_params[step_index][:4]
    prev_sample  = sum([i * j for i, j in zip(deltas, [sample] + self.train_model_outputs)])

    prev_sample = prev_sample + solver_pred

    return prev_sample

  def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
    num_inference_steps = num_inference_steps or len(timesteps)

    self.train_params = []
    for i in range(num_inference_steps):
      self.train_params.append(torch.zeros(min(i+2, 4), dtype=torch.float32, requires_grad=True))


  def get_flat_params(self) -> torch.Tensor:
    return parameters_to_vector(self.train_params)
  
  def set_flat_params(self, flat_params: torch.Tensor) -> None:
    vector_to_parameters(flat_params, self.train_params)


# @solver_registry.add_to_registry("COEFEXT")
# class COEFEXT:
#   order = 10
#   def _step(self, model_output, sample, solver_pred, step_index, **kwargs):
#     if kwargs.get("prediction_type", "epsilon") == "v_prediction":
#       sigma_t = self.sigmas[self.step_index]
#       alpha_t = self.sigma_to_alpha_t(sigma_t)
#       model_output = alpha_t * (sample * sigma_t + model_output)
      
#     if step_index == 0: 
#       self.train_model_outputs = []
#       self.prev_xt = []

#     self.train_model_outputs = [model_output] + self.train_model_outputs[:step_index] 
    
#     deltas = self.train_params[step_index][:step_index+2]
#     prev_sample  = sum([i * j for i, j in zip(deltas, [sample] + self.train_model_outputs)])

#     deltas1 = self.train_params[step_index][step_index+2:]
#     prev_sample += sum([i * j for i, j in zip(deltas1, self.prev_xt)])

#     prev_sample = prev_sample + solver_pred
#     self.prev_xt = [prev_sample] + self.prev_xt
#     return prev_sample

#   def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
#     num_inference_steps = num_inference_steps or len(timesteps)

#     self.train_params = []
#     for i in range(num_inference_steps):
#       self.train_params.append(torch.zeros(2*i+2, dtype=torch.float32, requires_grad=True))



# @solver_registry.add_to_registry("COEF0")
# class COEF0:
#   order = 3
#   def _step(self, model_output, sample, solver_pred, step_index, **kwargs):
#     if kwargs.get("prediction_type", "epsilon") == "v_prediction":
#       sigma_t = self.sigmas[self.step_index]
#       alpha_t = self.sigma_to_alpha_t(sigma_t)
#       model_output = alpha_t * (sample * sigma_t + model_output)
      
#     if step_index == 0: 
#       self.train_model_outputs = []

#     self.train_model_outputs = [model_output] + self.train_model_outputs[:2] 
    
#     deltas = self.train_params[step_index][:4]
#     deltas[0] = 0.
    
#     prev_sample  = sum([i * j for i, j in zip(deltas, [sample] + self.train_model_outputs)])

#     prev_sample = prev_sample + solver_pred

#     return prev_sample

#   def set_train_solver(self, num_inference_steps=None, timesteps=None, device=None):
#     num_inference_steps = num_inference_steps or len(timesteps)

#     self.train_params = []
#     for i in range(num_inference_steps):
#       self.train_params.append(torch.zeros(min(i+2, 4), dtype=torch.float32, requires_grad=True))

