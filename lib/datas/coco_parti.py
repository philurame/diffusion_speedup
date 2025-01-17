import pickle, os
from lib.registries import data_registry

@data_registry.add_to_registry('COCO')
class COCO:
  def __init__(self, data_path, max_samples):
    with open(os.path.join(data_path, 'datasets_coco_parti.pkl'), 'rb') as f:
      data = pickle.load(f)['COCO']

    self.anns = data['anns'][:max_samples]
    self.imgs = data['imgs'][:max_samples]
  
@data_registry.add_to_registry('PARTI')
class PARTI:
  def __init__(self, data_path, max_samples):
    with open(os.path.join(data_path, 'datasets_coco_parti.pkl'), 'rb') as f:
      data = pickle.load(f)['PARTI']

    self.anns = data['anns'][:max_samples]
    self.imgs = None