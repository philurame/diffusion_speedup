# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
import numpy as np
from typing import Union

# Shrinking operator
def shrink_lp_op(x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
  if lp_norm == 1: 
    #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    out = torch.abs(x)
    out.sub_(1.0 / beta).clamp_min_(0.0)
    out.mul_(torch.sign(x))
    return out
  else:
    #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1))
    out = torch.abs(x)
    out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
    out.mul_(torch.sign(x))
    return out

# Proximal solver || W - dequantize(quantize(W))||_p^p
#@torch.compile(fullgraph=True)
def optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis):
  W_q = torch.round(W_f * scale + zero).clamp_(min_max[0], min_max[1])
  W_r = (W_q - zero) / scale
  W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
  zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
  return W_r, W_q, zero, scale
    
@torch.inference_mode()
def optimize_weights_hqq(
  tensor: torch.Tensor,
  scale: torch.Tensor,
  zero: torch.Tensor,
  min_max: list,
  axis: int = 1,
  device: Union[str, None] = None,
  opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
  verbose: bool = False,
  **kwargs
) -> tuple:
  lp_norm, beta, kappa, iters = (
    opt_params["lp_norm"],
    opt_params["beta"],
    opt_params["kappa"],
    opt_params["iters"],
)

  if device is None:
    device = tensor.device
  else:
    device = torch.device(device)

  dtype = torch.float32 if (device.type == "cpu") else torch.float16
  W_f   = tensor.to(dtype=dtype, device=device)
  scale = scale.to(dtype=dtype, device=device)
  zero  = zero.to(dtype=dtype, device=device)

  best_error = torch.tensor(torch.inf, dtype=torch.float32, device=device)
  for i in range(iters):
    W_r, W_q, zero, scale = optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis)
    current_error = torch.abs(W_f - W_r).mean().float()

    if verbose: 
      print(i, np.round(current_error, 6))
    
    if current_error < best_error:
      best_error = current_error
    else:
      break

  scale = scale.to(tensor.device)
  zero = zero.to(tensor.device)
  del W_f, W_q, W_r
  torch.cuda.empty_cache()

  W_q = torch.round(tensor * scale + zero).clamp_(min_max[0], min_max[1])
  return W_q, scale, zero