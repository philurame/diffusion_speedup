from registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin
import numpy as np

@scheduler_registry.add_to_registry("GITS")
class GITScheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):    
    ts_schedules = {
      3: [998.9994506835938, 516.148193359375, 183.15957641601562],
      4: [998.9994506835938, 682.6539916992188, 366.30621337890625, 116.55728912353516],
      5: [998.9994506835938, 765.9013671875, 482.8583984375, 233.09194946289062, 66.60400390625],
      6: [998.9994506835938, 782.5517578125, 549.4483032226562, 333.01324462890625, 149.83399963378906, 33.29087829589844],
      7: [998.9994506835938, 849.1502685546875, 599.3961791992188, 399.60919189453125, 216.47007751464844, 83.26032257080078, 16.647491455078125],
      8: [998.9994506835938, 849.1502685546875, 649.3515625, 466.19342041015625, 299.7054443359375, 166.52090454101562, 66.60400390625, 16.647491455078125],
      9: [998.9994506835938, 849.1502685546875, 699.30224609375, 549.4483032226562, 399.60919189453125, 266.4013977050781, 149.83399963378906, 66.60400390625, 16.647491455078125],
      10: [999, 783, 632, 483, 350, 233, 133, 67, 33, 17],
      # 10: [998.9994506835938, 832.4999389648438, 699.30224609375, 566.0985107421875, 432.9044189453125, 299.7054443359375, 199.80447387695312, 116.55728912353516, 49.96816635131836, 16.647491455078125],
      11: [998.9994506835938, 849.1502685546875, 732.6011962890625, 599.3961791992188, 482.8583984375, 366.30621337890625, 266.4013977050781, 166.52090454101562, 99.89759063720703, 49.96816635131836, 16.647491455078125]
    }
    if num_inference_steps not in ts_schedules:
      timesteps = self._loglinear_interp(ts_schedules[11], num_inference_steps)
    else:
      timesteps = ts_schedules[num_inference_steps]

    self.prepare_solver_data(timesteps, device)
  

  def _loglinear_interp(self, t_steps, num_steps):
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys




  #   # 0.0292 is actually equal to ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]).sqrt().unsqueeze(0)
  #   sigma_schedule = {
  #     3: [14.6146, 1.7083, 0.532, 0.0292], 
  #     4: [14.6146, 3.1131, 1.0421, 0.3811, 0.0292], 
  #     5: [14.6146, 4.39, 1.5286, 0.6526, 0.2667, 0.0292],
  #     6: [14.6146, 4.7242, 1.9132, 0.9324, 0.4557, 0.1801, 0.0292],
  #     7: [14.6146, 6.4477, 2.2797, 1.1629, 0.6114, 0.3058, 0.1258, 0.0292],
  #     8: [14.6146, 6.4477, 2.7391, 1.4467, 0.8319, 0.4936, 0.2667, 0.1258, 0.0292],
  #     9: [14.6146, 6.4477, 3.3251, 1.9132, 1.1629, 0.7391, 0.4557, 0.2667, 0.1258, 0.0292],
  #     10: [14.6146, 5.9489, 3.3251, 2.0267, 1.2969, 0.8319, 0.5712, 0.3811, 0.2255, 0.1258, 0.0292],
  #     11: [14.6146, 6.4477, 3.8092, 2.2797, 1.5286, 1.0421, 0.7391, 0.4936, 0.3437, 0.2255, 0.1258, 0.0292]   
  #   }

  #   if num_inference_steps not in sigma_schedule: raise NotImplementedError

  #   all_sigmas_log =  np.log(((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt())
  #   sigmas = np.array(sigma_schedule[num_inference_steps])
  #   timesteps = np.array([self._sigma_to_t(sigma, all_sigmas_log) for sigma in sigmas])[:-1]

  #   self.prepare_solver_data(timesteps, device)
  
  # def _sigma_to_t(self, sigma, log_sigmas):
  #   # get log sigma
  #   log_sigma = np.log(np.maximum(sigma, 1e-10))

  #   # get distribution
  #   dists = log_sigma - log_sigmas[:, np.newaxis]

  #   # get sigmas range
  #   low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
  #   high_idx = low_idx + 1

  #   low = log_sigmas[low_idx]
  #   high = log_sigmas[high_idx]

  #   # interpolate sigmas
  #   w = (low - log_sigma) / (low - high)
  #   w = np.clip(w, 0, 1)

  #   # transform interpolation to time range
  #   t = (1 - w) * low_idx + w * high_idx
  #   t = t.reshape(sigma.shape)
  #   return t