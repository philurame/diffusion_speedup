from lib.registries import solver_registry

@solver_registry.add_to_registry("POLS")
class PolSolver:
  order = 2
  def step(self, model_output, timestep=None, sample=None):
    '''
    AFS and/or DDIM at last step could be added here
    '''
    self.model_outputs = self.model_outputs[-1:] + [model_output]

    if self.step_index == 0:
      prev_sample = self.first_order_update(model_output, sample=sample)
    else:
      prev_sample = self.second_order_update(model_output, sample=sample)

    self.step_index += 1
    return prev_sample


  def first_order_update(self, model_output, sample):
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    x_t = (alpha_t / alpha_s) * sample - model_output * alpha_t * (sigma_s - sigma_t)
    return x_t
  
  def second_order_update(self, model_output, sample):
    gamma = self.gammas[self.step_index]
    power = self.powers[self.step_index]

    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )
    alpha_t, alpha_s0 = (
      self.sigma_to_alpha_sigma_t(sigma_t),
      self.sigma_to_alpha_sigma_t(sigma_s0),
    )
    rho_t, rho_s0, rho_s1 = sigma_t, sigma_s0, sigma_s1
    phi_t, phi_s0, phi_s1 = rho_t ** gamma, rho_s0 ** gamma, rho_s1 ** gamma

    def integral_fn(a, b, x, power1, power2):
      # F(x) = inegrate (x^power1 - b) / (a - b) * x^(power2) dx
      return x**(power1+power2+1)/(power1+power2+1)/(a-b) - b*x**(power2+1)/(power2+1)/(a-b)
    
    power1, power2 = power, 1/gamma-1
    C1 = integral_fn(phi_s1**power1,phi_s0**power1,phi_t,power1,power2) - integral_fn(phi_s1**power1,phi_s0**power1,phi_s0,power1,power2)
    C2 = integral_fn(phi_s0**power1,phi_s1**power1,phi_t,power1,power2) - integral_fn(phi_s0**power1,phi_s1**power1,phi_s0,power1,power2)

    eps_s0, eps_s1 = self.model_outputs[-1], self.model_outputs[-2]
    x_t = alpha_t*(sample/alpha_s0 + eps_s1*C1/gamma + eps_s0*C2/gamma)
    return x_t