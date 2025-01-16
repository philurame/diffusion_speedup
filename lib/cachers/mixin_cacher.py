import torch
from diffusers import StableDiffusionXLPipeline

class BasePipeMixin(StableDiffusionXLPipeline):
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