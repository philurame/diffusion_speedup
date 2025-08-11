from registries import model_registry
from lib.models.SDXL.sdxl import BaseSDXL
from lib.models.models_utils.caching_timestep_helper import CachingTimestepHelper

@model_registry.add_to_registry("REINFORCE_CACHER")
class ReinforceCacherSDXL(BaseSDXL):
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        
        pipe = super().from_pretrained(*args, **kwargs)
        helper = CachingTimestepHelper(pipe)
        pipe.cacher = helper
        
        return pipe