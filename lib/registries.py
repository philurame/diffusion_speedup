import os, importlib

class ClassRegistry:
  """
  A class to act as a registry for storing classes with associated names.
  """
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

solver_registry    = ClassRegistry()
scheduler_registry = ClassRegistry()
model_registry     = ClassRegistry()
data_registry      = ClassRegistry()
metric_registry    = ClassRegistry()


def import_dir(path_dir):
  """
  Recursively load all .py files in the given folder (and subfolders).
  This ensures that any @add_to_registry decorators in those files are run,
  so the classes become registered in solver_registry or scheduler_registry.
  """
  # assuming that this file is in the root/utils dir
  PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  for root, dirs, files in os.walk(path_dir):
    for file in files:
      if file.endswith(".py") and file.startswith("r_") and not file.startswith("__"):
        # Convert the file path into a module path e.g., root/custom/solvers/euler_solver.py -> custom.solvers.euler_solver
        module_name = os.path.splitext(file)[0]
        # Build the relative package path from PROJECT_ROOT
        rel_path = os.path.relpath(root, PROJECT_ROOT)  # Change here
        package = rel_path.replace(os.sep, ".")
        full_module_name = f"{package}.{module_name}"
        importlib.import_module(full_module_name)
        # print(f"Loaded {full_module_name}")