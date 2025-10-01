from registries import metric_registry

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torch.nn.functional as F
import clip, torch, tqdm

@metric_registry.add_to_registry('AQ')
class AQMetric:
  '''adapted from https://github.com/LAION-AI/aesthetic-predictor'''
  # model_name_or_path='/home/jovyan/.cache/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth'
  model_name_or_path='/home/jovyan/maliev/DIFFUSION_SPEEDUP/DATA/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth'
  @torch.inference_mode()
  def __call__(self, imgs_gen, batch_size=64, device=None, verbose=False, **kwargs):
    device = device or imgs_gen.device
    clip_model, _ = clip.load('ViT-L/14', device=device)
    aq_model = torch.nn.Linear(768, 1).to(device)
    aq_model.load_state_dict(torch.load(self.model_name_or_path))
    aq_model.eval()
    image_transform = self.clip_transform(224)
    
    aq_score = 0
    c = 0
    for i in tqdm.tqdm(range(0, len(imgs_gen), batch_size), disable=not verbose):
      image_batch = imgs_gen[i:i + batch_size]
      image_batch = image_transform(image_batch)      
      image_batch = image_batch.to(device)

      with torch.no_grad():
        image_feats = clip_model.encode_image(image_batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        aq_score += aq_model(image_feats).squeeze(dim=-1).sum().item()
        c += image_batch.shape[0]
    
    return aq_score / c

  def clip_transform(self, n_px):
    return Compose([
      Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=False),
      CenterCrop(n_px),
      Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711)
      ),
    ])