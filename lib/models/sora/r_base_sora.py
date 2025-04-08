try:
  from lib.registries import model_registry
  from lib.models.sora.sora import BaseSora
  @model_registry.add_to_registry("SORA_BASE")
  class BasePipeline(BaseSora): pass
except:
  pass
