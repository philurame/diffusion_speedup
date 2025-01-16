import numpy as np
import torch

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