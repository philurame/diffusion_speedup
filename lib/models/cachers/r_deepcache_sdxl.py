from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL
from DeepCache import DeepCacheSDHelper


@model_registry.add_to_registry("DEEPCACHE")
class DEEPCACHE(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = super().from_pretrained()
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=kwargs.get("cache_interval", 3),
      cache_branch_id=kwargs.get("cache_branch_id", 0),
    )
    helper.enable()
    return pipe
