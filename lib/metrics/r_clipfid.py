from lib.registries import metric_registry
import os
import numpy as np
import tqdm
import torch
import pathlib
from scipy import linalg
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMAGE_EXTS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

class ImageFolder(torch.utils.data.Dataset):
  def __init__(self, files, transform=None):
    self.files = files
    self.transform = transform
  def __len__(self): return len(self.files)
  def __getitem__(self, idx):
    img = Image.open(self.files[idx]).convert("RGB")
    return self.transform(img) if self.transform else img


class CLIPFIDMetric:
  def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch32", num_workers: int = 0, device = None):
    # Load CLIP vision encoder + processor
    self.clip = CLIPModel.from_pretrained(model_name_or_path).vision_model
    self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
    self.num_workers = num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)
    if device!=None: 
      self.device = device
      self.clip = self.clip.to(self.device).eval()

  @torch.inference_mode()
  def __call__(self, **kwargs):
    # Batch / device
    batch_size = kwargs.get('batch_size', 64)
    device_arg = kwargs.get('device', None)
    self.device = torch.device(device_arg if device_arg else ("cuda" if torch.cuda.is_available() else "cpu"))
    self.clip = self.clip.to(self.device).eval()

    # Generated stats
    if isinstance(kwargs['imgs_gen'], torch.Tensor):
      verbose = kwargs
      mu_gen, sigma_gen = self._stats_from_tensor(kwargs['imgs_gen'], batch_size)
    else:
      mu_gen, sigma_gen = self._load_stats(kwargs['imgs_gen'], batch_size)

    # Real stats
    if 'reference_real' in kwargs and kwargs['reference_real'] is not None:
      mu_real, sigma_real = self._load_stats(kwargs['reference_real'], batch_size)
    elif 'imgs_real' in kwargs and kwargs['imgs_real'] is not None:
      mu_real, sigma_real = self._stats_from_tensor(kwargs['imgs_real'], batch_size)
    else:
      raise ValueError("Provide either imgs_real or reference_real for CLIP-FID.")

    # Return Frechet distance
    return float(self._frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real))

  def _stats_from_tensor(self, imgs: torch.Tensor, batch_size: int, verbose=False):
    # Ensure in CPU and normalized to [0,1]
    x = imgs.cpu()
    if x.dtype == torch.uint8:
      x = x.float().div(255.0)
    # If in [-1,1]
    if x.min() < -0.1:
      x = (x + 1) / 2

    # Build loader
    ds = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(
      ds, batch_size=batch_size, shuffle=False,
      drop_last=False, num_workers=self.num_workers
    )
    if verbose:
      loader = tqdm.tqdm(loader)
    feats = self._extract_features(loader)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

  def _load_stats(self, path: str, batch_size: int, verbose=False):
    path = pathlib.Path(path)
    if path.suffix.lower() == '.npz':
      with np.load(path) as f:
        return f['mu'], f['sigma']
    # Otherwise treat as folder of images
    files = sorted([str(f) for ext in IMAGE_EXTS for f in path.glob(f"*.{ext}")])
    return self._stats_from_files(files, batch_size, verbose)

  def _stats_from_files(self, files, batch_size, verbose):
    transform = transforms.Compose([
      transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(224),
      transforms.ToTensor(), 
    ])
    ds = ImageFolder(files, transform=transform)
    loader = torch.utils.data.DataLoader(
      ds, batch_size=batch_size, shuffle=False,
      drop_last=False, num_workers=self.num_workers
    )
    if verbose:
      loader = tqdm.tqdm(loader)
    feats = self._extract_features(loader)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)

  def _extract_features(self, loader):
    all_feats = []
    for batch in loader:
      imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
      inputs = self.processor(images=imgs, return_tensors="pt", do_rescale=False).to(self.device)
      outputs = self.clip(**inputs)
      # pooled_output shape: (B, hidden_dim)
      feats = outputs.pooler_output.detach().cpu().numpy()
      all_feats.append(feats)
    return np.vstack(all_feats)

  def _frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
      covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


@metric_registry.add_to_registry('CLIP-FID')
class CLIPFID(CLIPFIDMetric):
  @torch.inference_mode()
  def __call__(self, **kwargs):
    if 'reference_real' not in kwargs or kwargs['reference_real'] is None:
      kwargs['reference_real'] = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/coco-fid_coco30k.npz'
    return super().__call__(**kwargs)