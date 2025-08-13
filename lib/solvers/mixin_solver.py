class SolverMixin:
  def epsilon_output(self, model_output, sample, prediction_type='epsilon'):
    if prediction_type == 'epsilon':
      return model_output

    if prediction_type == "flow_matching":
      return model_output
      # return ( alpha_t * model_output + sample ) / ( alpha_t + sigma_t )

    sigma_t = self.sigmas[self.step_index]
    alpha_t = self.sigma_to_alpha_t(sigma_t)
    
    if prediction_type == "v_prediction":
      return alpha_t * (sample * sigma_t + model_output)
    
    if prediction_type == "data_prediction":
      return ( sample - alpha_t * model_output ) / sigma_t
    
    
    raise ValueError(f"weird prediction_type: {prediction_type}")
