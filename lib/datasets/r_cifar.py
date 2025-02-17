from lib.registries import data_registry

@data_registry.add_to_registry('CIFAR')
class CIFAR:
  def __init__(self, **kwargs):
    self.anns = None
    self.imgs = None