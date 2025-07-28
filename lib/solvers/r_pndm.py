from registries import solver_registry

@solver_registry.add_to_registry("IPNDM2")
class IPNDM2:
  order = 2
  def step(self, model_output, sample=None, **kwargs):
    if kwargs.get("prediction_type", "epsilon") == "v_prediction":
      sigma_t = self.sigmas[self.step_index]
      alpha_t = self.sigma_to_alpha_t(sigma_t)
      model_output = alpha_t * (sample * sigma_t + model_output)
      
    self.model_outputs = self.model_outputs[-3:] + [model_output]

    if min(self.order-1, self.step_index) == 0:
      pndm_model_output = self.model_outputs[-1]
    elif min(self.order-1, self.step_index) == 1:
      pndm_model_output = (3 * self.model_outputs[-1] - self.model_outputs[-2]) / 2
    elif min(self.order-1, self.step_index) == 2:
      pndm_model_output = (23 * self.model_outputs[-1] - 16 * self.model_outputs[-2] + 5 * self.model_outputs[-3]) / 12
    else:
      pndm_model_output = (1 / 24) * (55 * self.model_outputs[-1] - 59 * self.model_outputs[-2] + 37 * self.model_outputs[-3] - 9 * self.model_outputs[-4])
    
    prev_sample = self.first_order_update(pndm_model_output, sample=sample)

    self.step_index += 1
    return prev_sample

  def first_order_update(self, model_output, sample):
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    x_t = (alpha_t / alpha_s) * sample - model_output * alpha_t * (sigma_s - sigma_t)
    return x_t


@solver_registry.add_to_registry("IPNDM3")
class IPNDM3(IPNDM2):
  order = 3
  is_trainable = False

@solver_registry.add_to_registry("IPNDM4")
class IPNDM4(IPNDM2):
  order = 4
  is_trainable = False