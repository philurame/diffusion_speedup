
from registries import model_registry
from lib.models.sora.sora import BaseSora
@model_registry.add_to_registry("SORA_BASE")
class BasePipeline(BaseSora):
  @classmethod
  def from_pretrained(self, *args, **kwargs):
    kwargs.pop('is_train', None)
    pipe = super().from_pretrained(is_train=False, *args, **kwargs)
    for p in pipe.transformer.parameters():
      p.requires_grad_(False)
    for p in pipe.vae.parameters():
      p.requires_grad_(False)
    return pipe