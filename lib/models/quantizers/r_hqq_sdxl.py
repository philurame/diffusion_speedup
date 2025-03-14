from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL
from lib.models.quantizers.quant_utils import get_linear_and_conv_layers, seed_everything
import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear


@model_registry.add_to_registry("HQQ")
class HQQ4(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    device = kwargs.get('device', 'cuda')
    nbits = kwargs.get('nbits', 4)
    pipe = super().from_pretrained(*args, **kwargs)
    q_unet(pipe, nbits=nbits, device=device)
    return pipe

  
def q_unet(pipe, nbits, device):
  all_layers = get_linear_and_conv_layers(pipe)
  quant_config = BaseQuantizeConfig(nbits=nbits, group_size=64)

  all_quantized = []
  seed_everything()
  for layer in all_layers:
    hqq_layer = HQQLinear(
      layer, #torch.nn.Linear or None 
      quant_config=quant_config, #quantization configuration
      compute_dtype=torch.float16, #compute dtype
      device=device, #cuda device
      initialize=True, #Use False to quantize later
      del_orig=True #if True, delete the original layer
    )
    all_quantized.append(hqq_layer)

  for orig_layer, quantized_layer in zip(all_layers, all_quantized):
    found_original = False
    modules_to_update = []
    for submodule in pipe.unet.modules():
      for child_name, child_module in submodule.named_children():
        if child_module is orig_layer:
          modules_to_update.append((submodule, child_name))
          found_original = True
    assert found_original, f"could not find {orig_layer}"

    for submodule, child_name in modules_to_update:
      setattr(submodule, child_name, quantized_layer)
  return all_quantized