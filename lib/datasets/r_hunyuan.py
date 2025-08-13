from registries import data_registry
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # DIFFUSION_SPEEDUP

@data_registry.add_to_registry('HUNYUAN')
class HUAWEI_TRAIN:
  def __init__(self, data_path=os.path.join(ROOT, 'DATA')):
    path_prompts = os.path.join(data_path, 'hunyuan_prompts.txt')
    path_images  = None
    
    self.prompts = None
    self.imgs    = None
    if os.path.exists(path_prompts):
      with open(path_prompts, 'r') as f:
        self.prompts = [line.strip() for line in f]