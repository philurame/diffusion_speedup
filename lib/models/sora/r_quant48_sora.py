from registries import model_registry
from lib.models.sora.sora import BaseSora
from lib.models.models_utils.quantization.quantize_model import quantize_model

import torch, gc

CONFIG = {
  'w_nbits': 4,
  'a_nbits': 8,
  'calc_error': '',
  'use_bruterot': 0,
  'bruterot': {
    'path': '/workspace-SR008.fs2/philurame/QUANTIZATION/broterot_SQ_HQQ.pkl',
    },
  'use_rotation': 1,
  'use_hqq': 1,
  'use_smoothquant': 1,
  'smoothquant': {
    'path_absmax': '/workspace-SR008.fs2/philurame/QUANTIZATION/absmax_10.pkl',
    'aggr_weight': False,
    'aggr_timesteps': 'quantile', # best: quantile - median or max - median?
    'aggr_prompts': 'median',
    'alpha': 0.5,
  },
  'skip_timesteps': [],
  'skip_activations_timesteps': [],
  'group_size': 64,
  'nfe': 10,

  'use_duquant': 0,
  'duquant': {},
}

SKIP_LAYERS = ['proj_out',
 'transformer_blocks.11.attn2.to_v',
 'transformer_blocks.23.attn2.to_v',
 'transformer_blocks.12.attn2.to_v',
 'transformer_blocks.22.attn2.to_v',
 'transformer_blocks.19.attn2.to_v',
 'transformer_blocks.25.attn2.to_v',
 'transformer_blocks.12.attn1.to_out.0',
 'transformer_blocks.24.attn2.to_v',
 'transformer_blocks.10.ff.net.2',
 'transformer_blocks.18.attn2.to_v',
 'transformer_blocks.11.attn1.to_q',
 'transformer_blocks.21.attn2.to_v',
 'transformer_blocks.17.attn2.to_v']

# SKIP_LAYERS = ['proj_out',
# 'transformer_blocks.11.attn2.to_v',
# 'transformer_blocks.12.attn2.to_v',
# 'transformer_blocks.10.ff.net.2',
# 'transformer_blocks.10.attn2.to_q',
# 'transformer_blocks.22.attn2.to_v',
# 'transformer_blocks.11.attn1.to_q',
# 'transformer_blocks.23.attn2.to_v',
# 'transformer_blocks.25.attn2.to_v',
# 'transformer_blocks.12.attn1.to_out.0',
# 'transformer_blocks.10.ff.net.0.proj',
# 'transformer_blocks.19.attn2.to_v',
# 'transformer_blocks.12.ff.net.2',
# 'transformer_blocks.24.attn2.to_v',
# 'transformer_blocks.22.ff.net.0.proj']


@model_registry.add_to_registry("SORA_QUANT48")
class SoraQuant(BaseSora):    
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    device = torch.device(kwargs["device"])
    pipe = super().from_pretrained(*args, **kwargs)

    torch.cuda.empty_cache()
    pipe = quantize_model(pipe, CONFIG, inplace=True, skip_layers=SKIP_LAYERS, quantize_text_encoder=True)
    gc.collect()
    torch.cuda.empty_cache()
    return pipe



CONFIG1 = {
  'w_nbits': 4,
  'a_nbits': 8,
  'calc_error': 'mean',
  'use_bruterot': 1,
  'bruterot': {
    'path': '/workspace-SR008.fs2/philurame/QUANTIZATION/broterot_SQ_HQQ_wtf200.pkl',
    },

  'use_rotation': 0,
  'use_hqq': 1,

  'use_duquant': 0,
  'duquant': {
    # 'w_path': '/workspace-SR008.fs2/philurame/QUANTIZATION/duquant/du_weight.pkl',
    # 'w_path': '/workspace-SR008.fs2/philurame/QUANTIZATION/duquant/du_weight_smooth.pkl',
    'w_path': '/workspace-SR008.fs2/philurame/QUANTIZATION/duquant/du_weight_smooth(q-med).pkl',
    'a_path': '',
    'R1': False,
  },

  'use_smoothquant': 1,
  'smoothquant': {
    'path_absmax': '/workspace-SR008.fs2/philurame/QUANTIZATION/absmax_10.pkl',
    'aggr_weight': False,
    'aggr_timesteps': 'quantile', # best: quantile - median or max - median?
    'aggr_prompts': 'median',
    'alpha': 0.5,
  },

  'skip_timesteps': [],
  'skip_activations_timesteps': [],

  'group_size': 64,
  'nfe': 10,
}

SKIP_LAYERS1 = ['proj_out',
 'transformer_blocks.11.attn2.to_v',
 'transformer_blocks.23.attn2.to_v',
 'transformer_blocks.12.attn2.to_v',
 'transformer_blocks.22.attn2.to_v',
 'transformer_blocks.19.attn2.to_v',
 'transformer_blocks.25.attn2.to_v',
 'transformer_blocks.12.attn1.to_out.0',
 'transformer_blocks.24.attn2.to_v',
 'transformer_blocks.10.ff.net.2',
 'transformer_blocks.18.attn2.to_v',
 'transformer_blocks.11.attn1.to_q',
 'transformer_blocks.21.attn2.to_v',
 'transformer_blocks.17.attn2.to_v']


@model_registry.add_to_registry("SORA_QUANT48R")
class SoraQuant(BaseSora):    
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    device = torch.device(kwargs["device"])
    pipe = super().from_pretrained(*args, **kwargs)

    torch.cuda.empty_cache()
    pipe = quantize_model(pipe, CONFIG1, inplace=True, skip_layers=SKIP_LAYERS1, quantize_text_encoder=True)
    gc.collect()
    torch.cuda.empty_cache()
    return pipe