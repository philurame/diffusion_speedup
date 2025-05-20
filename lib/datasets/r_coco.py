from lib.registries import data_registry
import pickle, os

@data_registry.add_to_registry('COCO30k')
class COCO30:
  def __init__(self, data_path, max_samples):
    with open('/workspace-SR008.fs2/philurame/coco2014/captions/coco30k_prompts.txt', 'r') as f:
      self.anns = [x.strip() for x in f.readlines()][:max_samples]
    self.imgs = None

@data_registry.add_to_registry('COCOLD3')
class COCOLD3:
  def __init__(self, data_path, max_samples):
    with open('/workspace-SR008.fs2/philurame/coco2014/captions/cocold3_prompts.txt', 'r') as f:
      self.anns = [x.strip() for x in f.readlines()][:max_samples]
    self.imgs = None