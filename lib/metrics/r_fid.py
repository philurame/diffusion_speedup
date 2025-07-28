from registries import metric_registry
from lib.metrics.fid_utils import FIDClass

from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPProcessor

import os
import numpy as np
import torch

@metric_registry.add_to_registry('FID')
class FIDMetric(FIDClass):
  img_size = 299
  @torch.inference_mode()
  def __call__(self, imgs_gen, imgs_real, batch_size=64, device=None, verbose=False, **kwargs):
    torch.manual_seed(0)
    device = device or imgs_gen.device
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    self.model = InceptionV3([block_idx]).to(device)
    self.model.eval()

    mu_gen, sigma_gen   = self._stats_from_tensor(imgs_gen,  batch_size, verbose=verbose)
    mu_real, sigma_real = self._stats_from_tensor(imgs_real, batch_size, verbose=verbose)
    return float(self._frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real))
    
  def _extract_features(self, loader):
    device = next(self.model.parameters()).device
    activs = []
    for batch in loader:
      # batch could be tuple from TensorDataset
      imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
      imgs = imgs.to(device)
      out = self.model(imgs)[0]
      if out.size(2) != 1 or out.size(3) != 1:
        out = adaptive_avg_pool2d(out, (1, 1))
      feats = out.squeeze(-1).squeeze(-1).cpu().numpy()
      activs.append(feats)
    return np.vstack(activs)


@metric_registry.add_to_registry('FID-DINO')
class DINOv2FIDMetric(FIDClass):
  img_size = 224
  model_name_or_path = "facebook/dinov2-base"
  @torch.inference_mode()
  def __call__(self, imgs_gen, imgs_real, batch_size=64, device=None, verbose=False, **kwargs):
    torch.manual_seed(0)
    device = device or imgs_gen.device
    self.processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
    self.model = AutoModel.from_pretrained(self.model_name_or_path).to(device)
    self.model.eval()

    mu_gen, sigma_gen   = self._stats_from_tensor(imgs_gen,  batch_size, verbose=verbose)
    mu_real, sigma_real = self._stats_from_tensor(imgs_real, batch_size, verbose=verbose)
    return float(self._frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real))

  def _extract_features(self, loader):
    device = self.model.device
    all_feats = []
    for batch in loader:
      imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
      inputs = self.processor(
        images=imgs, return_tensors="pt", do_normalize=True, do_rescale=False
      ).to(device)
      outputs = self.model(**inputs)
      # use [CLS] token embedding
      cls_feats = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
      all_feats.append(cls_feats)
    return np.vstack(all_feats)


@metric_registry.add_to_registry('FID-CLIP')
class CLIPFIDMetric(FIDClass):
  img_size = 224
  model_name_or_path = 'openai/clip-vit-large-patch14'
  @torch.inference_mode()
  def __call__(self, imgs_gen, imgs_real, batch_size=64, device=None, verbose=False, **kwargs):
    torch.manual_seed(0)
    device = device or imgs_gen.device
    self.processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
    self.clip_model = CLIPModel.from_pretrained(self.model_name_or_path).vision_model.to(device)
    self.clip_model.eval()

    mu_gen, sigma_gen   = self._stats_from_tensor(imgs_gen,  batch_size, verbose=verbose)
    mu_real, sigma_real = self._stats_from_tensor(imgs_real, batch_size, verbose=verbose)
    return float(self._frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real))

  def _extract_features(self, loader):
    device = next(self.clip_model.parameters()).device
    all_feats = []
    for batch in loader:
      imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
      inputs = self.processor(images=imgs, return_tensors="pt", do_rescale=False).to(device)
      outputs = self.clip_model(**inputs)
      # pooled_output shape: (B, hidden_dim)
      feats = outputs.pooler_output.detach().cpu().numpy()
      all_feats.append(feats)
    return np.vstack(all_feats)


# ---------------------------------------------------------
# ALTERNATIVE: torchmetrics FID
# ---------------------------------------------------------
# from torchmetrics.image.fid import FrechetInceptionDistance
# import tqdm
# @metric_registry.add_to_registry('FID')
# class FIDMetric:
#   @torch.inference_mode()
#   def __call__(self, imgs_gen, imgs_real, batch_size=512, device=None, verbose=False, **kwargs):
#     device = kwargs.get('device', None)
#     device = device or imgs_gen.device
#     fid_model = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
#     for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), disable=not verbose):
#       fid_model.update(imgs_real[i:i+batch_size].to(device), real=True)
#       fid_model.update(imgs_gen[i:i+batch_size].to(device), real=False)
#     return fid_model.compute().item()