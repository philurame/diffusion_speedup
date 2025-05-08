# THIS IS ABSOLUTELY EQUIVALENT TO DDIM !!!

# from lib.registries import solver_registry
# import torch

# @solver_registry.add_to_registry("EULER")
# class EULER:
#   order = 1
#   def step(self, model_output, sample=None, **kwargs):
#     sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]

#     # if self.step_index == 0:
#     sample *= (sigma_s**2 + 1) ** 0.5
    
#     prev_sample = sample + model_output * (sigma_t - sigma_s)

#     # if self.step_index < len(self.timesteps)-1:
#     sigma = self.sigmas[self.step_index]
#     prev_sample /= (sigma_t**2 + 1) ** 0.5

#     self.step_index += 1

#     return prev_sample
  
#   def scale_model_input(self, sample):
#     return sample



# max_sigma = max(self.sigmas) if isinstance(self.sigmas, list) else self.sigmas.max()
#         if self.config.timestep_spacing in ["linspace", "trailing"]:
#             return max_sigma

#         return (max_sigma**2 + 1) ** 0.5



# sample / (sigma**2 + 1) ** 0.5 * (max_sigma**2 + 1) ** 0.5