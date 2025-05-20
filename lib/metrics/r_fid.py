from lib.registries import metric_registry

import os
import pathlib
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3

IMAGE_EXTS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

class ImageFolder(torch.utils.data.Dataset):
  def __init__(self, files, transform=None):
    self.files = files
    self.transform = transform
  def __len__(self): return len(self.files)
  def __getitem__(self, idx):
    img = Image.open(self.files[idx]).convert("RGB")
    return self.transform(img) if self.transform else img

class FIDMetric:
  def __init__(self, dims: int = 2048, num_workers: int = None):
    self.dims = dims
    self.num_workers = num_workers if num_workers is not None else min(os.cpu_count() or 0, 8)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    self.model = InceptionV3([block_idx])

  @torch.inference_mode()
  def __call__(self, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    device_arg = kwargs.get('device', None)
    self.device = torch.device(device_arg if device_arg else ("cuda" if torch.cuda.is_available() else "cpu"))
    self.model = self.model.to(self.device)

    # Get stats for generated
    if isinstance(kwargs['imgs_gen'], torch.Tensor):
      mu_gen, sigma_gen = self._stats_from_tensor(kwargs['imgs_gen'], batch_size)
    else:
      mu_gen, sigma_gen = self._load_stats(kwargs['imgs_gen'], batch_size)

    # Get stats for real
    if 'reference_real' in kwargs and kwargs['reference_real'] is not None:
      mu_real, sigma_real = self._load_stats(kwargs['reference_real'], batch_size)
    elif 'imgs_real' in kwargs and kwargs['imgs_real'] is not None:
      mu_real, sigma_real = self._stats_from_tensor(kwargs['imgs_real'], batch_size)

    return float(self.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real))

  def _stats_from_tensor(self, tensor: torch.Tensor, batch_size: int):
    # Normalize uint8 and [-1,1] to [0,1]
    x = tensor.cpu()
    if x.dtype == torch.uint8:
      x = x.float().div(255.0)
    if x.min() < -0.1:
      x = x.add(1).div(2)
    # Build loader
    ds = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                          drop_last=False, num_workers=self.num_workers)
    activs = self._activations_from_loader(loader)
    mu = np.mean(activs, axis=0)
    sigma = np.cov(activs, rowvar=False)
    return mu, sigma

  def _get_activations(self, files, batch_size):
    files = list(files)
    if batch_size > len(files): batch_size = len(files)
    ds = ImageFolder(files, transform=TF.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    return self._activations_from_loader(loader)

  def _activations_from_loader(self, loader):
    self.model.eval()
    activs = []
    for batch in loader:
      # batch could be tuple from TensorDataset
      imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
      imgs = imgs.to(self.device)
      out = self.model(imgs)[0]
      if out.size(2) != 1 or out.size(3) != 1:
          out = adaptive_avg_pool2d(out, (1, 1))
      feats = out.squeeze(-1).squeeze(-1).cpu().numpy()
      activs.append(feats)
    return np.vstack(activs)

  def calculate_activation_statistics(self, files, batch_size):
    act = self._get_activations(files, batch_size)
    return np.mean(act, axis=0), np.cov(act, rowvar=False)

  def _load_stats(self, path: str, batch_size: int):
    if path.lower().endswith('.npz'):
      with np.load(path) as f:
        return f['mu'], f['sigma']
    p = pathlib.Path(path)
    files = [str(f) for ext in IMAGE_EXTS for f in sorted(p.glob(f"*.{ext}"))]
    return self.calculate_activation_statistics(files, batch_size)

  def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
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
  

@metric_registry.add_to_registry('FID-Img')
class FIDImg(FIDMetric):
  @torch.inference_mode()
  def __call__(self, **kwargs):
    kwargs.pop('reference_real', None)
    assert 'imgs_gen' in kwargs and kwargs['imgs_gen'] is not None
    return super().__call__(**kwargs)

@metric_registry.add_to_registry('FID-Ref')
class FIDRef(FIDMetric):
  @torch.inference_mode()
  def __call__(self, **kwargs):
    kwargs.pop('imgs_real', None)
    kwargs['reference_real'] = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/val2014_fid_refs.npz'
    return super().__call__(**kwargs)