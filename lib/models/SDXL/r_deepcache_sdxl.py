from registries import model_registry
from lib.models.SDXL.sdxl import BaseSDXL
from DeepCache import DeepCacheSDHelper

@model_registry.add_to_registry("DEEPCACHE2")
class DEEPCACHE3(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = super().from_pretrained(**kwargs)
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=kwargs.get("cache_interval", 2),
      cache_branch_id=kwargs.get("cache_branch_id", 0),
    )
    helper.enable()
    return pipe

@model_registry.add_to_registry("DEEPCACHE3")
class DEEPCACHE3(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = super().from_pretrained(**kwargs)
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=kwargs.get("cache_interval", 3),
      cache_branch_id=kwargs.get("cache_branch_id", 0),
    )
    helper.enable()
    return pipe

@model_registry.add_to_registry("DEEPCACHE4")
class DEEPCACHE4(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = super().from_pretrained(**kwargs)
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=kwargs.get("cache_interval", 4),
      cache_branch_id=kwargs.get("cache_branch_id", 0),
    )
    helper.enable()
    return pipe

@model_registry.add_to_registry("DEEPCACHE5")
class DEEPCACHE5(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = super().from_pretrained(**kwargs)
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=kwargs.get("cache_interval", 5),
      cache_branch_id=kwargs.get("cache_branch_id", 0),
    )
    helper.enable()
    return pipe
