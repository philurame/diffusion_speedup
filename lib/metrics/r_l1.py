from registries import metric_registry
import torch

@metric_registry.add_to_registry('L1')
class L1:
  @torch.inference_mode()
  def __call__(self, imgs_gen, imgs_real, **kwargs):
    '''
    imgs_gen, imgs_real in [0,1]
    '''
    return torch.abs(imgs_gen - imgs_real).mean().item()