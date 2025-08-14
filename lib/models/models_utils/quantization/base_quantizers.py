#####################################################

'''
Quantization applied to 2-dim tensor of shape (Group, -1),
where each (1, -1) subtensor is quantized separately with its own scale and zero_point scalars
'''

#####################################################

'''
Xq = [Xf * S + Z]
S = (q_max - q_min)/(Xmax - Xmin)
Z = q_min - Xmin * S

Xf = (Xq - Z) / S

Xmax, Xmin = max(Xf), min(Xf) for symmetrical
Xmax, Xmin = max(|Xf|), -max(|Xf|) for non-symmetrical

[q_min, q_max] = [-N-1, N] for symmetrical
[q_min, q_max] = [0, N] for non-symmetrical
'''

#####################################################

'''
[-N-1, N] and [0 ,N]:
signed intB: [-2**(B-1), 2**(B-1)-1] 
unsign intB: [0, 2**(B)-1]
where B is either 4 or 8.
'''

#####################################################

import torch
import torch.nn as nn

# import logging
# logger = logging.getLogger(__name__)

class BaseQuantizer(nn.Module):

  def __init__(self, nbits, round_zero=False):
    super().__init__()
    
    self.nbits = nbits
    if self.nbits is not None:
      self.q_min, self.q_max = 0, 2**(self.nbits)-1
      self.round_zero = round_zero
      
      self.x_max = None
      self.x_min = None

      self.register_buffer('scale', None)
      self.register_buffer('zero_point', None)
  
  def dequantize(self, x_quant: torch.Tensor):
    if self.nbits is None: return x_quant
    x_dequant = (x_quant - self.zero_point) / self.scale
    return x_dequant

class StaticQuantizer(BaseQuantizer):

  def __init__(self, nbits, round_zero=False):
    super().__init__(nbits, round_zero=round_zero)
  
  def forward(self, x: torch.Tensor):
    if self.nbits is None: return x
    x_quant = self.quantize(x)
    x_dequant = (x_quant - self.zero_point) / self.scale
    return x_dequant
    
  def quantize(self, x: torch.Tensor):
    if self.nbits is None: return x
    self.init_quant_params(x)
    x_int = torch.round(x * self.scale + self.zero_point)
    x_quant = torch.clamp(x_int, self.q_min, self.q_max)
    return x_quant
    
  def init_quant_params(self, x):
    assert len(x.shape) == 2  # [N_group, -1]

    x_max = x.max(dim=1)[0]
    x_min = x.min(dim=1)[0]

    q_range = self.q_max - self.q_min
    x_range = x_max - x_min
    scale = q_range / x_range

    # assert torch.all(scale < self.tresh), "unexpected large scale exists"
    scale = torch.where(x_range.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
    scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems

    zero_point = self.q_min - x_min * scale

    if self.round_zero:
      zero_point = torch.round(zero_point)

    self.scale = scale.unsqueeze(-1)  # [G] -> [G,1]
    self.zero_point = zero_point.unsqueeze(-1)

class DynamicQuantizer(BaseQuantizer):

  def __init__(self, nbits, round_zero=False):
    super().__init__(nbits, round_zero=round_zero)
  
  def forward(self, x: torch.Tensor, device=None):
    if self.nbits is None: return x
    x_quant = self.quantize(x, device)
    x_dequant = (x_quant - self.zero_point) / self.scale
    return x_dequant

  def quantize(self, x:torch.Tensor, device=None):
    if self.nbits is None: return x
    device = x.device if device is not None else device

    # get the quant_params online
    assert len(x.shape) == 2  # [N_group, -1]
    assert torch.isnan(x).sum() == 0  # no nan exists
            
    x_max = x.max(dim=1)[0]
    x_min = x.min(dim=1)[0]
    self.x_max = torch.max(self.x_max.to(device), x_max) if self.x_max is not None else x_max
    self.x_min = torch.min(self.x_min.to(device), x_min) if self.x_min is not None else x_min

    q_range = self.q_max - self.q_min
    x_range = self.x_max - self.x_min
    scale = q_range / x_range
    
    scale = torch.where(x_range.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
    scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems
    
    zero_point = self.q_min - self.x_min * scale

    if self.round_zero:
      zero_point = torch.round(zero_point)

    self.scale = scale.unsqueeze(-1)  # [G] -> [G,1]
    self.zero_point = zero_point.unsqueeze(-1)

    # quantize model with quant params
    x_int = torch.round(x * self.scale + self.zero_point)
    x_quant = torch.clamp(x_int, self.q_min, self.q_max)
    return x_quant