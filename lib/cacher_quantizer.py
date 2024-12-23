import torch, os
import numpy as np

from diffusers import StableDiffusionXLPipeline
from DeepCache import DeepCacheSDHelper
from tgate import TgateSDXLLoader

import sys
from utils.hqq_q import q_unet

from utils.class_registry import ClassRegistry
cacher_quantizer_registry = ClassRegistry()

#####################################################################################################
# CACHERS
#####################################################################################################

@cacher_quantizer_registry.add_to_registry("NONE")
class BasePipe(StableDiffusionXLPipeline):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", 
      torch_dtype=torch.float16, 
      variant="fp16",
      local_files_only=True
      )
    
    # freeze model
    for module in pipe.components.values():
      if isinstance(module, torch.nn.Module):
        for param in module.parameters():
          param.requires_grad = False
    return pipe

@cacher_quantizer_registry.add_to_registry("FREEU")
class FREEU(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    return pipe

@cacher_quantizer_registry.add_to_registry("DEEPCACHE")
class DEEPCACHE(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=3,
      cache_branch_id=0,
    )
    helper.enable()
    return pipe

@cacher_quantizer_registry.add_to_registry("TGATE")
class TGATE(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    pipe = TgateSDXLLoader(pipe)
    return pipe
  def __call__(self, *args, **kwargs):
    num_inference_steps = kwargs["num_inference_steps"]
    gate_step = num_inference_steps//2.5
    return self.tgate(*args, **kwargs, gate_step=gate_step)

#####################################################################################################
# QUANTIZERS
#####################################################################################################

@cacher_quantizer_registry.add_to_registry("HQQ4")
class HQQ4(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    q_unet(pipe, nbits=4)
    return pipe

@cacher_quantizer_registry.add_to_registry("HQQ3")
class HQQ3(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    q_unet(pipe, nbits=3)
    return pipe

@cacher_quantizer_registry.add_to_registry("VQDM4")
class VQDM4(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    vqdm4_dir_path = "/home/mdnikolaev/philurame/Q_LIB/unets/vqdm_4"
    sys.path.append(vqdm4_dir_path) # add "src" to path
    unet = torch.load(os.path.join(vqdm4_dir_path, "quantized_unet.pickle"), map_location="cuda")    
    pipe.unet = unet
    return pipe
