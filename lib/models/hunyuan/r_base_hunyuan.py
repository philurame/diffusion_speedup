
from registries import model_registry
from lib.models.hunyuan.hunyuan import BaseHunyuan
@model_registry.add_to_registry("HUNYUAN_BASE")
class BasePipeline(BaseHunyuan): pass

