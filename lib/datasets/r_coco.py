from registries import data_registry
import os, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # DIFFUSION_SPEEDUP

@data_registry.add_to_registry('COCO-LONG')
class COCO_LONG:
  def __init__(self, data_path=os.path.join(ROOT, 'DATA')):
    self.prompts = None
    self.imgs    = None

    path_prompts = os.path.join(data_path, 'coco_long_10k.txt')
    path_imgs    = os.path.join(data_path, 'coco_imgs299_10k.pt')

    if os.path.exists(path_prompts):
      with open(path_prompts, 'r') as f:
        self.prompts = [line.strip() for line in f]
    
    if os.path.exists(path_imgs):
      self.imgs = torch.load(path_imgs)
  

@data_registry.add_to_registry('COCO-SHORT')
class COCO_SHORT:
  def __init__(self, data_path=os.path.join(ROOT, 'DATA')):
    self.prompts = None
    self.imgs    = None

    path_prompts = os.path.join(data_path, 'coco_short_10k.txt')
    path_imgs    = os.path.join(data_path, 'coco_imgs299_10k.pt')

    if os.path.exists(path_prompts):
      with open(path_prompts, 'r') as f:
        self.prompts = [line.strip() for line in f]
    
    if os.path.exists(path_imgs):
      self.imgs = torch.load(path_imgs)
