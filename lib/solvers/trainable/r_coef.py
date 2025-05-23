import torch
from lib.registries import solver_registry


@solver_registry.add_to_registry("COEFEXT2")
class COEFEXT2:
  order = 10
  def _step(self, model_output, sample, solver_pred, step_index, **kwargs):
    if kwargs.get("prediction_type", "epsilon") == "v_prediction":
      sigma_t = self.sigmas[self.step_index]
      alpha_t = self.sigma_to_alpha_t(sigma_t)
      model_output = alpha_t * (sample * sigma_t + model_output)
      
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


