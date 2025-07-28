import torch
import torch.nn as nn
import torch.nn.functional as F

from base_quantizers import StaticQuantizer #, DynamicQuantizer
from hqq_utils import optimize_weights_hqq
# from bitpack_utils import pack_u8, unpack_u8

class QuantizedLinear(torch.nn.Linear):
  def __init__(
    self,
    in_features: int,
    out_features: int,
    config: dict,
    fp_module: torch.nn.Linear,
    layer_name: str = None,
    bruterots: dict = None,
    rotation_matrix: torch.Tensor = None,
    smooth_scale: torch.Tensor = None,
    duquant_params: list = None,
  ) -> None:
    is_bias = fp_module.bias is not None
    device = fp_module.weight.device
    super().__init__(in_features, out_features, is_bias, device)
    if is_bias:
      self.bias.data = fp_module.bias.data

    self.layer_name = layer_name
    self.rotation_matrix = rotation_matrix

    # layer-specific params:
    self.duquant_params = duquant_params[layer_name] if duquant_params is not None else None
    self.smooth_scale = smooth_scale[layer_name] if smooth_scale is not None else None

    # ========================================================================
    # parse config
    # ========================================================================
    self.calc_error = config['calc_error']
    self.group_size = config['group_size'] # there is also fp_module.weight.shape[-1] group_size actually!
    assert in_features % self.group_size == 0

    self.w_quantizer = StaticQuantizer(nbits=config['w_nbits'], round_zero=False)
    self.a_quantizer = StaticQuantizer(nbits=config['a_nbits'], round_zero=False)
    self.use_smooth_quant = smooth_scale is not None
    self.use_rotation = config['use_rotation']
    self.use_duquant  = config['use_duquant']
    self.use_hqq      = config['use_hqq']
    self.use_bruterot = config['use_bruterot']
    self.duquant_R1   = config.get('duquant').get('R1')
    self.skip_activations_timesteps = config.get('skip_activations_timesteps', [])
    self.skip_timesteps = config.get('skip_timesteps', [])
    

    self.bruterot = None
    if self.use_bruterot:
      if isinstance(bruterots[layer_name], tuple) and isinstance(bruterots[layer_name][-1], bool):
        self.bruterot, self.use_smooth_quant, self.use_hqq = bruterots[layer_name]
      else:
        self.bruterot = bruterots[layer_name]
    
    
    # debug/experimental params
    self.inference_count = 0
    if self.calc_error:
      self.fp_module = fp_module
    
    if config['w_nbits'] is None:
      self.weight.data = fp_module.weight.data
      return

    # ========================================================================
    # process weights
    # ========================================================================
    fp_module.weight.data = fp_module.weight.data.half() # SORA WHY?
    W = fp_module.weight.data
    W_shape = W.shape
    assert torch.device(W.device) != torch.device('cpu')
    assert W.dtype == torch.float16

    # SMOOTH QUANT
    if self.use_smooth_quant:
      self.smooth_scale = self.smooth_scale.to(W.device)
      W = W / self.smooth_scale.view([1, W.shape[-1]])

    W = W.reshape([-1, self.group_size])

    if self.use_bruterot:
      if self.bruterot is not None:
        if isinstance(self.bruterot, torch.Tensor):
          self.bruterot = self.bruterot.to(dtype=W.dtype, device=W.device)
          W = torch.matmul(W, self.bruterot)
        else: # duquant
          (R1, perm, R2) = self.bruterot
          R1 = R1.to(dtype=W.dtype, device=W.device)
          R2 = R2.to(dtype=W.dtype, device=W.device)
          self.bruterot = (R1, perm, R2)
          W = torch.matmul(W, R1)
          W = W.reshape(W_shape)[..., perm]
          W = torch.matmul(W.reshape([-1, self.group_size]), R2)

    # HADAMARD
    if self.use_rotation:
      self.rotation_matrix = self.rotation_matrix.to(W.device)
      W = torch.matmul(W, self.rotation_matrix)

    # DUQUANT
    if self.use_duquant:
      for i, (R1, perm, R2) in enumerate(self.duquant_params):
        R1 = R1.to(dtype=W.dtype, device=W.device)
        R2 = R2.to(dtype=W.dtype, device=W.device)
        self.duquant_params[i] = (R1, perm, R2)
        W = torch.matmul(W, R1)
        if not self.duquant_R1:
          W = W.reshape(W_shape)[..., perm]
          W = torch.matmul(W.reshape([-1, self.group_size]), R2)
    
    assert W.dtype == torch.float16

    W_q = self.w_quantizer.quantize(W)

    # HQQ
    if self.use_hqq:
      min_max = [self.w_quantizer.q_min, self.w_quantizer.q_max]
      W_q, self.w_quantizer.scale, self.w_quantizer.zero_point = optimize_weights_hqq(
        W, self.w_quantizer.scale, self.w_quantizer.zero_point, axis=1, min_max=min_max,
      )

    # instantly dequantize for faster testing
    self.weight.data = self.w_quantizer.dequantize(W_q).reshape(W_shape)

    if self.calc_error:
      self.werror = self._calc_error(W.reshape(W_shape), self.weight.data, kind=self.calc_error)
      

  def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Forward pass with quantization.
    Input shape: [B, N_token, C]
    """
    # if self.layer_name == 'transformer_blocks.11.attn1.to_k':
    #   print(self.inference_count)

    if self.calc_error:
      y_fp = self.fp_module(X)
      if self.inference_count in self.skip_timesteps:
        self.inference_count += 1
        self.error = torch.tensor(0.)
        return y_fp

    
    # ========================================================================
    # process activations
    # ========================================================================
    X_shape = X.shape

    # SMOOTH QUANT
    if self.use_smooth_quant:
      X = X * self.smooth_scale.view([1, X.shape[-1]])

    X = X.reshape([-1, self.group_size])

    if self.use_bruterot and self.bruterot is not None:
      if isinstance(self.bruterot, torch.Tensor):
        X = torch.matmul(X, self.bruterot)
      else: # duquant
        (R1, perm, R2) = self.bruterot
        X = torch.matmul(X, R1)
        X = X.reshape(X_shape)[..., perm]
        X = torch.matmul(X.reshape([-1, self.group_size]), R2)

    # HADAMARD
    if self.use_rotation:
      X = torch.matmul(X, self.rotation_matrix)

    # DUQUANT
    if self.use_duquant:
      for (R1, perm, R2) in self.duquant_params:
        X = torch.matmul(X, R1)
        if not self.duquant_R1:
          X = X.reshape(X_shape)[..., perm]
          X = torch.matmul(X.reshape([-1, self.group_size]), R2)
    
    # ========================================================================
    # Quantize and compute output & error
    # ========================================================================
    # Quantize for regular timesteps, skip for skip_activations_timesteps
    if self.inference_count not in self.skip_activations_timesteps or True:
      X = self.a_quantizer(X) # it is valid even if self.a_quantizer.nbits is None

    X = X.reshape(X_shape)

    y = F.linear(X, self.weight, self.bias, *args, **kwargs)

    if self.calc_error:
      self.error = self._calc_error(y_fp, y, kind=self.calc_error)
    
    self.inference_count += 1

    return y

  def _calc_error(self, y_fp, y, kind='mean', cutmin=1e-4, cutmax=1e5):
    signal = torch.sqrt(torch.mean(y_fp.float() ** 2))
    if kind=='mean':
      diff = torch.sqrt(torch.mean((y_fp - y).float() ** 2))
    elif kind=='max':
      diff = torch.sqrt(torch.max((y_fp - y).float() ** 2))
    
    if   signal < cutmin: return diff
    if signal > cutmax: return torch.tensor(0.)
    return diff / signal