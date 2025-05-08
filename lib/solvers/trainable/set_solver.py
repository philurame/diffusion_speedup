import torch, pickle

# init_solvers = [
#   'DEIS',
#   'DPMS', 'DPMS3',
#   'IPNDM2', 'IPNDM3',
#   'UNIPC2', 'UNIPC2H', 'UNIPC2_', 'UNIPC2H_', 'UNIPC3', 'UNIPC3H', 'UNIPC3_', 'UNIPC3H_'
# ]

# from lib.registries import import_dir



class SETSOLVER:
  def get_coeffs(self, solver):
    if solver == 'DEIS':
      return self.get_deis_coeffs()
    if solver == 'IPNDM':
      return self.get_ipndm_coeffs()

  def get_deis_coeffs(self):
    '''returns coeffs corresponding to x_t, eps_{t}, eps_{t+1}, eps_{t+2}'''
    sigma_t, sigma_s0 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
    )
    alpha_t, alpha_s0 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
    )
    if self.step_index == 0:
      return alpha_t/alpha_s0, -alpha_t * (sigma_s0 - sigma_t), 0, 0

    def ind_fn(t, b, c):
      return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

    sigma_s1 = self.sigmas[self.step_index - 1]
    coef1 = ind_fn(sigma_t, sigma_s0, sigma_s1) - ind_fn(sigma_s0, sigma_s0, sigma_s1)
    coef2 = ind_fn(sigma_t, sigma_s1, sigma_s0) - ind_fn(sigma_s0, sigma_s1, sigma_s0)
    return alpha_t/alpha_s0, alpha_t*coef1, alpha_t*coef2, 0  # x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)

  def get_ipndm_coeffs(self):
    '''returns coeffs corresponding to x_t, eps_{t}, eps_{t+1}, eps_{t+2}'''

    ct = cs = cs0 = 0
    if self.step_index == 0:
      ct = 1
    elif self.step_index == 1:
      ct = 1.5
      cs = -0.5
    else:
      ct = 23/12
      cs = -4/3
      cs0 = 5/12

    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    scale = alpha_t * (sigma_t - sigma_s)
    return alpha_t / alpha_s, scale * ct, scale * cs, scale * cs0 