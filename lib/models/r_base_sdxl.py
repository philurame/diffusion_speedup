from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL

@model_registry.add_to_registry("BASE")
class BasePipeline(BaseSDXL): pass
