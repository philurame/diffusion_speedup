import torch, sys, os
from lib.registries import cacher_registry
from lib.quantizers.mixin_quantizer import BasePipeMixin

@cacher_registry.add_to_registry("VQDM4")
class VQDM4(BasePipeMixin):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()

    # must provide path to vqdm_4 quantized model!
    vqdm4_dir_path = "/home/mdnikolaev/philurame/Q_LIB/unets/vqdm_4"
    sys.path.append(vqdm4_dir_path)

    unet = torch.load(os.path.join(vqdm4_dir_path, "quantized_unet.pickle"), map_location="cuda")    
    pipe.unet = unet
    return pipe