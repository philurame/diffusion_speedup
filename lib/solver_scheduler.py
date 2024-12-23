import torch
import numpy as np
from diffusers import (
  EulerDiscreteScheduler, 
  DPMSolverMultistepScheduler, 
  DEISMultistepScheduler, 
  UniPCMultistepScheduler,
  DDIMScheduler,
  PNDMScheduler,
  IPNDMScheduler
  )

from utils.class_registry import ClassRegistry
solver_registry = ClassRegistry()
scheduler_registry = ClassRegistry()

#####################################################################################################
# SOLVERS
#####################################################################################################

class BaseSolverMixin:
  config = {
    'num_train_timesteps': 1000,
    'beta_start': 0.00085,
    'beta_end': 0.012,
    'beta_schedule': 'scaled_linear',
    'trained_betas': None,
    'prediction_type': 'epsilon',
    'interpolation_type': 'linear',
    'use_karras_sigmas': False,
    'use_exponential_sigmas': False,
    'use_beta_sigmas': False,
    'sigma_min': None,
    'sigma_max': None,
    'timestep_spacing': 'leading',
    'timestep_type': 'discrete',
    'steps_offset': 1,
    'rescale_betas_zero_snr': False,
    'final_sigmas_type': 'zero', # or 'sigma_min'?
    'clip_sample': False,
    'sample_max_value': 1.0,
    'set_alpha_to_one': False,
    'skip_prk_steps': True,
  }

  def _convert_to_lu(self, in_lambdas: torch.Tensor, num_inference_steps) -> torch.Tensor:
    lambda_min: float = in_lambdas[-1].item()
    lambda_max: float = in_lambdas[0].item()
    ramp = np.linspace(0, 1, num_inference_steps)
    lambdas = (lambda_max + ramp * (lambda_min - lambda_max))
    return lambdas

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

@solver_registry.add_to_registry("DPMSORIG")
class DPMSORIGBaseSolver(DPMSolverMultistepScheduler, BaseSolverMixin):
  @classmethod
  def from_config(cls, **kwargs):
    solver = super().from_config(
      BaseSolverMixin.config,
      lower_order_final = False,
      final_sigmas_type = "sigma_min",
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

#####################################################################################################
# SCHEDULERTS (MIXINS)
#####################################################################################################

class BaseScheduler:
  def _set_timesteps_common(self, sigmas, timesteps, device):
    if self.config["final_sigmas_type"] == "sigma_min":
      sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
    elif self.config["final_sigmas_type"] == "zero":
      sigma_last = 0
    else:
      raise ValueError(
        f"`final_sigmas_type` must be one of 'zero' or 'sigma_min', but got {self.config.final_sigmas_type}"
      )
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas).to("cpu")
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
    self.num_inference_steps = len(timesteps)
    self.model_outputs = [None] * self.config.get('solver_order', 1)
    self.lower_order_nums = 0
    self._step_index = None
    self._begin_index = None

@scheduler_registry.add_to_registry("LINEAR")
class LINEARScheduler(BaseScheduler):
   def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    timesteps = (
      np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1) # why prevent ts to be zero?
      .round()[::-1][:-1]
      .copy()
      .astype(np.int64)
      )
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("DDIM")
class DDIMScheduler(BaseScheduler):
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    step_ratio = 1000 // num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("KARRAS")
class KARRASScheduler(BaseScheduler):
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    sigmas = np.flip(sigmas).copy()
    sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("SNR")
class SNRScheduler(BaseScheduler):
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    lambdas = np.flip(log_sigmas.copy())
    lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
    sigmas = np.exp(lambdas)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("AYS")
class AYSScheduler(BaseScheduler):
  def _loglinear_interp(self, t_steps, num_steps):
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys
  
  def _get_ays_timesteps_ts(self, num_inference_steps):
    ays_ts_10 = np.array([999, 845, 730, 587, 443, 310, 193, 116, 53, 13])
    new_ts = self._loglinear_interp(ays_ts_10, num_inference_steps)
    return new_ts.round().astype(int)
  
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    timesteps = self._get_ays_timesteps_ts(num_inference_steps)
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)

@scheduler_registry.add_to_registry("AYSSD15")
class AYSScheduler(BaseScheduler):
  def _loglinear_interp(self, t_steps, num_steps):
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys
  
  def _get_ays_timesteps_ts(self, num_inference_steps):
    ays_ts_10 = np.array([999, 850, 736, 645, 545, 455, 343, 233, 124, 24])
    new_ts = self._loglinear_interp(ays_ts_10, num_inference_steps)
    return new_ts.round().astype(int)
  
  def set_timesteps(self, num_inference_steps = None, device = None, timesteps = None):
    timesteps = self._get_ays_timesteps_ts(num_inference_steps)
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    self._set_timesteps_common(sigmas, timesteps, device)