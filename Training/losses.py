import torch
import torch.nn.functional as F

class ClassRegistry:
  def __init__(self):
    self._registry = {}
  def add_to_registry(self, name):
    def decorator(cls):
      self._registry[name] = cls
      return cls
    return decorator
  def __iter__(self):
    return iter(self._registry.keys())
  def __getitem__(self, name):
    return self._registry[name]
  def get(self, name, default=None):
    return self._registry.get(name, default)

loss_registry = ClassRegistry()


# ========================================================================================
# L1
# ========================================================================================
@loss_registry.add_to_registry("LATENT-L1")
class LatentL1(torch.nn.Module):
  is_latent = True
  def __init__(self, **kwargs):
    super().__init__()

  def forward(self, gen_latents=None, latents=None, **kwargs):
    loss = F.l1_loss(gen_latents, latents, reduction='none')
    loss = loss.mean(dim=list(range(1,len(loss.shape))))
    return loss # [batch_size]

@loss_registry.add_to_registry("L1")
class L1(torch.nn.Module):
  is_latent = False
  def __init__(self, **kwargs):
    super().__init__()

  def forward(self, gen_imgs=None, imgs=None, **kwargs):
    loss = F.l1_loss(gen_imgs, imgs, reduction='none')
    loss = loss.mean(dim=list(range(1,len(loss.shape))))
    return loss

try:
  from .vbench_utils.utils import *

  # ========================================================================================
  # VBENCH: IQ-AQ-MS
  # ========================================================================================
  @loss_registry.add_to_registry("IQ-AQ-MS")
  class RLIQAQMS(BASE_VBENCH):
    is_latent = False
    def __init__(self, device, coeffs, **kwargs):
      super().__init__(device, init_iq=True, init_aq=True, init_ms=True)
      self.coeffs = coeffs
    
    def forward(self, gen_imgs=None, **kwargs):
      with torch.no_grad():
        aq = aq_loss(self.aq_model, self.clip_model, gen_imgs).cpu()
        iq = iq_loss(self.iq_model, gen_imgs).cpu()
        ms = ms_loss(self.ms_model, gen_imgs)
        combined_loss = ( self.coeffs[0] * iq + self.coeffs[1] * aq + self.coeffs[2] * ms ) / ( self.coeffs[0] + self.coeffs[1] + self.coeffs[2] )
      return combined_loss


  # ========================================================================================
  # VBENCH: IQF-AQF-MS
  # ========================================================================================
  @loss_registry.add_to_registry("IQF-AQF-L1")
  class IQFAQFL1(BASE_VBENCH):
    is_latent = False
    def __init__(self, device, coeffs, **kwargs):
      super().__init__(device, init_iq=True, init_aq=True, init_ms=False)
      self.coeffs = coeffs
    
    def forward(self, gen_imgs=None, imgs=None, **kwargs):
      losses = []
      with torch.no_grad():
        iq_l1 = iq_l1_loss(self.iq_model, gen_imgs, imgs).cpu()
        aq_l1 = aq_l1_loss(self.clip_model, gen_imgs, imgs).cpu()
        l1 = F.l1_loss(gen_imgs, imgs, reduction='none').cpu()
        l1 = l1.mean(dim=list(range(1,len(l1.shape))))
        combined_loss = ( self.coeffs[0] * iq_l1 + self.coeffs[1] * aq_l1 + self.coeffs[2] * l1 ) / ( self.coeffs[0] + self.coeffs[1] + self.coeffs[2] )
      return combined_loss
except Exception as e:
  print(f'Some Losses not loaded: {e}')