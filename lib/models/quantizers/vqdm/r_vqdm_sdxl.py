from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL
import torch, sys, os

VQDM_DIR = os.path.dirname(os.path.abspath(__file__))
if VQDM_DIR not in sys.path:
  sys.path.insert(0, VQDM_DIR)

@model_registry.add_to_registry("VQDM4")
class VQDM4(BaseSDXL):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    unet = torch.load(os.path.join(VQDM_DIR, "quantized_unet.pkl"), map_location="cuda")    
    pipe.unet = unet
    return pipe