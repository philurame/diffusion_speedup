from registries import solver_registry
from lib.solvers.mixin_solver import BaseSolverMixin
from diffusers import DEISMultistepScheduler

@solver_registry.add_to_registry("POLS")
class PolSolver(DEISMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      dict(BaseSolverMixin.config, final_sigmas_type = 'sigma_min', algorithm_type = "dpmsolver"),
      **kwargs
    )
    solver.approx_eps = 'lin'
    return solver
  
  def gamma_update(self, model_output_list, sample):
    gamma = self.gammas[self.step_index]
    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )
    alpha_t, sigma_t   = self._sigma_to_alpha_sigma_t(sigma_t) # \sqrt{\alpha_t}, \sqrt{1-\slpha_t}
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
    rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1
    phi_s0, phi_s1 = rho_s0 ** gamma, rho_s1 ** gamma
    
    def integral_fn(a, b, gamma, x, approx='lin'):
      if approx == 'lin':
        # F(x) = inegrate (x - b) / (a - b) * x^(1/gamma-1) dx -> return F(x^gamma)
        return x ** (gamma + 1) * (1/(a - b)/(1 + 1/gamma)) - x * (b * gamma / (a - b))
      else:
        raise NotImplementedError

    C1 = integral_fn(phi_s1,phi_s0,gamma,rho_t, self.approx_eps) - integral_fn(phi_s1,phi_s0,gamma,rho_s0, self.approx_eps)
    C2 = integral_fn(phi_s0,phi_s1,gamma,rho_t, self.approx_eps) - integral_fn(phi_s0,phi_s1,gamma,rho_s0, self.approx_eps)

    eps_s1, eps_s0 = model_output_list[-2], model_output_list[-1] # [prev, curr]
    x_t = alpha_t*(sample/alpha_s0 + eps_s1*C1/gamma + eps_s0*C2/gamma)
    return x_t
  
  def step(self, model_output, timestep, sample, return_dict=True):
    # cringe but i dont care
    if len(self.timesteps) == 5:
      self.gammas = [1, -1.6169717315378749, -1.9681308791303862, -1.236976512804185, 0.18607430114268286] 
    elif len(self.timesteps) == 7:
      self.gammas = [1, -1.3447751160386219, -1.4149097616012618, -1.732297286938051, -1.9850735936914463, -1.197814184081598, 0.4187522417430869]
    elif len(self.timesteps) == 10:
      self.gammas = [1, -1.0030695661710312, -0.77511919242302, 0.5941783361418471, 0.7253813439060255, -2.1192263443890305, -3.4736722692279214, 0.2987171480171791, -1.4293031031946326, 0.945882922615633]
    else:
      raise NotImplementedError

    if self.step_index is None:
      self._init_step_index(timestep)

    lower_order_final = (
      (self.step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
    )
    lower_order_second = (
      (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
    )

    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output

    if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
        prev_sample = self.deis_first_order_update(model_output, sample=sample)
    elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
        prev_sample = self.gamma_update(self.model_outputs, sample=sample)
    else:
      raise NotImplementedError

    if self.lower_order_nums < self.config.solver_order:
      self.lower_order_nums += 1

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
      return (prev_sample,)
    else:
      raise NotImplementedError
    
  def convert_model_output(self, model_output, *args, **kwargs):
    # not a data prediction
    return model_output