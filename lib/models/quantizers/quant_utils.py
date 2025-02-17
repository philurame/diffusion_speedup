import re
import torch.nn as nn

import random, torch
import numpy as np


def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True


def get_linear_and_conv_layers(pipe, convClass=nn.Conv2d, linClass=nn.Linear, refilter=None, min_channels=16):
  if refilter is None:
    LINEAR_LAYER_ONLY_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|proj_(in|out))$"
    # DEFAULT_LAYER_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|conv(\d+|_shortcut)?|proj_(in|out))$"
    refilter = LINEAR_LAYER_ONLY_REGEX

  def layer_filter_fn(layer: nn.Module, layer_name: str) -> bool:
    if convClass is not None and isinstance(layer, (convClass,)):
      if min(layer.in_channels, layer.out_channels) < min_channels:
        return
    elif isinstance(layer, linClass):
      if min(layer.in_features, layer.out_features) < min_channels:
        return
    else:
      return
    return re.search(refilter, layer_name)

  # collect groups
  down_group = []
  # collect from down blocks
  for i, block in enumerate(pipe.unet.down_blocks):
    group = {}
    for module_name, module in block.named_modules():
      full_module_name = f"down_blocks.{i}.{module_name}"
      if layer_filter_fn(module, full_module_name):
        group[full_module_name] = module
    down_group.append(list(group.items()))

  # collect from mid block
  group = {}
  block = pipe.unet.mid_block
  for module_name, module in block.named_modules():
    full_module_name = f"mid_block.{module_name}"
    if layer_filter_fn(module, full_module_name):
      group['full_module_name'] = module
  mid_group = [list(group.items())]

  up_group = []
  # collect from up blocks
  for i, block in enumerate(pipe.unet.up_blocks):
    group = {}
    for module_name, module in block.named_modules():
      full_module_name = f"up_blocks.{i}.{module_name}"
      if layer_filter_fn(module, full_module_name):
        group['full_module_name'] = module
    up_group.append(list(group.items()))
  all_layers = down_group+mid_group+up_group
  all_layers = [i[1] for j in all_layers for i in j]
  return all_layers