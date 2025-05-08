from lib.registries import model_registry
from lib.models.SD15.sd15 import BaseSD15

@model_registry.add_to_registry("SD15_BASE")
class BasePipeline(BaseSD15): pass
