from lib.registries import model_registry
from lib.models.SD14.sd14 import BaseSD14

@model_registry.add_to_registry("SD14_BASE")
class BasePipeline(BaseSD14): pass
