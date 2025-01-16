from registries import solver_registry
from lib.solvers.mixin_solver import BaseSolverMixin

from diffusers import (
  EulerDiscreteScheduler, 
  DPMSolverMultistepScheduler, 
  DEISMultistepScheduler, 
  UniPCMultistepScheduler,
  DDIMScheduler,
  PNDMScheduler,
  IPNDMScheduler
  )

@solver_registry.add_to_registry("EULER")
class EULERBaseSolver(EulerDiscreteScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      solver_order = 1,
      **kwargs
      )
    return solver

@solver_registry.add_to_registry("DDIM")
class DDIMBaseSolver(DPMSolverMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      solver_order = 1, 
      algorithm_type = 'dpmsolver++', 
      final_sigmas_type = 'zero',
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("DDIMHF")
class DDIMHFBaseSolver(DDIMScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("PNDM")
class PNDMBaseSolver(PNDMScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    raise NotImplementedError # TODO explicitly, PNDMScheduler-HF realization is very bad
  
@solver_registry.add_to_registry("IPNDM")
class IPNDMBaseSolver(IPNDMScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("DPMS")
class DPMSBaseSolver(DPMSolverMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("DPMS3")
class DPMS3BaseSolver(DPMSolverMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      solver_order = 3,
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("DEIS")
class DEISBaseSolver(DEISMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      dict(BaseSolverMixin.config, final_sigmas_type = 'sigma_min'),
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("DEIS3")
class DEIS3BaseSolver(DEISMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      dict(BaseSolverMixin.config, final_sigmas_type = 'sigma_min'),
      solver_order = 3,
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("UNIPC")
class UNIPCBaseSolver(UniPCMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      **kwargs
    )
    return solver

@solver_registry.add_to_registry("UNIPC3")
class UNIPC3BaseSolver(UniPCMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      solver_order = 3,
      **kwargs
    )
    return solver