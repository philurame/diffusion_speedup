from lib.registries import data_registry
import os

@data_registry.add_to_registry('HUAWEI')
class HUAWEI:
  def __init__(self, data_path, max_samples):
    with open(os.path.join(data_path, 'huawei_prompts.txt.txt'), 'r') as f:
      self.anns = [x.strip() for x in f.readlines()]

    self.imgs = None