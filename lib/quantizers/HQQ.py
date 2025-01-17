import torch
from lib.registries import cacher_registry
from lib.quantizers.mixin_quantizer import BasePipeMixin, get_linear_and_conv_layers, seed_everything
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear


def q_unet(pipe, nbits=4):
  all_layers = get_linear_and_conv_layers(pipe)
  quant_config = BaseQuantizeConfig(nbits=nbits, group_size=64)

  all_quantized = []
  seed_everything()
  for layer in all_layers:
    hqq_layer = HQQLinear(layer, #torch.nn.Linear or None 
                          quant_config=quant_config, #quantization configuration
                          compute_dtype=torch.float16, #compute dtype
                          device='cuda', #cuda device
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


@cacher_registry.add_to_registry("HQQ4")
class HQQ4(BasePipeMixin):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    q_unet(pipe, nbits=4)
    return pipe

@cacher_registry.add_to_registry("HQQ3")
class HQQ3(BasePipeMixin):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    q_unet(pipe, nbits=3)
    return pipe