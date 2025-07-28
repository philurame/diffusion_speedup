from registries import model_registry
from lib.models.SDXL.sdxl import BaseSDXL

@model_registry.add_to_registry("SDXL_BASE")
class BasePipeline(BaseSDXL): pass
