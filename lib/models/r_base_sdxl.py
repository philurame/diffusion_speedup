from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL

@model_registry.add_to_registry("SDXL_BASE")
class BasePipeline(BaseSDXL): pass
