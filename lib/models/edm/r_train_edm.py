from lib.registries import model_registry
from lib.models.edm.edm import EDM
import torch.utils.checkpoint as cp

@model_registry.add_to_registry('EDM_TRAIN')
class EDMTrain(EDM): 
  @classmethod
  def from_pretrained(self, *args, **kwargs):
    kwargs.pop('is_train', None)
    return super().from_pretrained(is_train=True, *args, **kwargs)

  # def make_unet_solver_step(self, solver, image, t, device):
  #   # predict x0
  #   sigma_t = solver.sigmas[solver.step_index].to(device)
  #   alpha_t = solver.sigma_to_alpha_t(sigma_t)
  #   x_0 = cp.checkpoint(
  #     self.unet, image / alpha_t, sigma_t,
  #     use_reentrant = False,
  #   )

  #   # convert to eps
  #   model_output = (image - x_0 * alpha_t) / (alpha_t * sigma_t)
  #   image = solver.step(model_output, t, image)    
  #   return image