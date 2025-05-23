from lib.registries import solver_registry

@solver_registry.add_to_registry("DDIM")
class DDIM:
  order = 1
  def step(self, model_output, sample=None, **kwargs):
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)

    if kwargs.get("prediction_type", "epsilon") == "v_prediction":

      pred_original_sample = alpha_s * sample - (1-alpha_s**2)**0.5 * model_output
      pred_epsilon = alpha_s * model_output + (1-alpha_s**2)**0.5 * sample

      pred_sample_direction = (1 - alpha_t**2)**(0.5) * pred_epsilon
      prev_sample = alpha_t * pred_original_sample + pred_sample_direction


      self.step_index += 1
      return prev_sample

    
    prev_sample = (alpha_t / alpha_s) * sample + model_output * alpha_t * (sigma_t - sigma_s)

    self.step_index += 1
    return prev_sample