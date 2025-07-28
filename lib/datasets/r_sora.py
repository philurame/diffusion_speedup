from registries import data_registry
import os

@data_registry.add_to_registry('SORA-HUAWEI-TRAIN')
class HUAWEI_TRAIN:
  def __init__(self, data_path):
    with open(os.path.join(data_path, 'sora_huawei_prompts_train.txt'), 'r') as f:
      self.prompts = [x.strip() for x in f.readlines()]
    self.imgs = None

@data_registry.add_to_registry('SORA-HUAWEI-TEST')
class HUAWEI_TEST:
  def __init__(self, data_path):
    with open(os.path.join(data_path, 'sora_huawei_prompts_test.txt'), 'r') as f:
      self.prompts = [x.strip() for x in f.readlines()]
    self.imgs = None

@data_registry.add_to_registry('SORA-CHRONO')
class CHRONO:
  def __init__(self, data_path):
    with open(os.path.join(data_path, 'sora_chrono_prompts_test.txt'), 'r') as f:
      self.prompts = [x.strip() for x in f.readlines()]
    self.imgs = None