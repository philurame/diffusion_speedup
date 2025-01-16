from registries import cacher_registry
from lib.cachers.mixin_cacher import BasePipeMixin

from DeepCache import DeepCacheSDHelper
from tgate import TgateSDXLLoader

@cacher_registry.add_to_registry("NONE")
class BasePipe(BasePipeMixin):
  pass

@cacher_registry.add_to_registry("DEEPCACHE")
class DEEPCACHE(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
      cache_interval=3,
      cache_branch_id=0,
    )
    helper.enable()
    return pipe

@cacher_registry.add_to_registry("TGATE")
class TGATE(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    pipe = TgateSDXLLoader(pipe)
    return pipe
  def __call__(self, *args, **kwargs):
    num_inference_steps = kwargs["num_inference_steps"]
    gate_step = num_inference_steps//2.5
    return self.tgate(*args, **kwargs, gate_step=gate_step)
  
@cacher_registry.add_to_registry("FREEU")
class FREEU(BasePipe):
  @classmethod
  def from_pretrained(cls):
    pipe = super().from_pretrained()
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    return pipe
