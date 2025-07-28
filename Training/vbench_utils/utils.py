from .iq_utils import iq_loss, iq_l1_loss
from .aq_utils import aq_loss, aq_l1_loss
from .ms_utils import ms_loss
from .ms_utils import MotionSmoothness

from pyiqa.archs.musiq_arch import MUSIQ
import clip, torch

class BASE_VBENCH(torch.nn.Module):
  '''
  initialize IQ, AQ, MS models
  '''
  def __init__(self, device, init_iq=True, init_aq=True, init_ms=True):
    super().__init__()
    if init_iq:
      model_path = '/home/jovyan/.cache/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
      self.iq_model = MUSIQ(pretrained_model_path=model_path).to(device)
      self.iq_model.training = False

    if init_aq:
      self.clip_model, _ = clip.load('ViT-L/14', device=device)

      model_path = '/home/jovyan/.cache/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth'
      self.aq_model = torch.nn.Linear(768, 1).to(device)
      self.aq_model.load_state_dict(torch.load(model_path))
      self.aq_model.eval()
    
    if init_ms:
      ms_config = '/workspace-SR008.fs2/philurame/VMODEL/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'
      ms_ckpt   = '/home/jovyan/.cache/vbench/amt_model/amt-s.pth'

      self.ms_model = MotionSmoothness(config=ms_config, ckpt=ms_ckpt, device=device)