import imageio, gc, json, os, torch, subprocess, tempfile
import torch.nn.functional as F

class ClassRegistry:
  def __init__(self):
    self._registry = {}
  def add_to_registry(self, name):
    def decorator(cls):
      self._registry[name] = cls
      return cls
    return decorator
  def __iter__(self):
    return iter(self._registry.keys())
  def __getitem__(self, name):
    return self._registry[name]
  def get(self, name, default=None):
    return self._registry.get(name, default)

metric_registry = ClassRegistry()

# ========================================================================================
# Metrics could be added here
# ========================================================================================