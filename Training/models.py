import torch, os, random, sys
import numpy as np
import torch.nn.functional as F

def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# =============================================================================
# PIPE
# =============================================================================
ROOT = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP'
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  model_registry, 
  data_registry, 
  metric_registry
)
import_dir(os.path.join(ROOT, 'lib'))


def construct_pipeline(solver, scheduler, model_name=None, half=True, init_solver=None, pipe=None, **pipe_kwargs):
  '''construct a pipeline with given model_name, solver and scheduler'''
  
  if init_solver is not None:
    init_solver  = solver_registry[init_solver]
    train_solver = solver_registry[solver]
    class SolverClass(init_solver, train_solver):
      def step(self, model_output, sample=None, **kwargs):
        step_index = self.step_index
        solver_pred = super().step(model_output, sample, **kwargs)
        res_pred = super()._step(model_output, sample, solver_pred, step_index, **kwargs)
        return res_pred
  else:
    SolverClass = solver_registry[solver]

  if pipe is None:
    PipeClass = model_registry[model_name]
    pipe = PipeClass.from_pretrained(half=half, **pipe_kwargs)
  
  SchedulerClass = scheduler_registry[scheduler]
  class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
  pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
  return pipe