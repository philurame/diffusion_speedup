from registries import metric_registry

from pyiqa.archs.musiq_arch import MUSIQ
from torchvision import transforms
import torch.nn.functional as F
import os, torch, tqdm

@metric_registry.add_to_registry('IQ')
class IQMetric:
  model_name_or_path='/home/jovyan/.cache/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
  @torch.inference_mode()
  def __call__(self, imgs_gen, batch_size=64, device=None, verbose=False, **kwargs):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = device or imgs_gen.device
    iq_model = MUSIQ(pretrained_model_path=self.model_name_or_path).to(device)
    iq_model.training = False

    images = self.iq_transform(imgs_gen)
    score = 0
    for i in tqdm.tqdm(range(0, len(images), batch_size), disable=not verbose):
      frames = images[i:i + batch_size].to(device)
      score += iq_model(frames).sum().item()
    return score/len(images)

  def iq_transform(self,images):
    _, _, h, w = images.size()
    if max(h, w) > 512:
      scale = 512.0 / max(h, w)
      images = transforms.Resize(size=(int(h * scale), int(w * scale)), antialias=False)(images)
    return images