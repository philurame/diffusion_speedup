from lib.registries import model_registry
from lib.models.edm.edm import EDM

@model_registry.add_to_registry('EDM_BASE')
class EDMBase(EDM): 
  pass