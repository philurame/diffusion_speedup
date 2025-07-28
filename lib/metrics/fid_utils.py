import os
import numpy as np
import tqdm
import torch
import pathlib
from scipy import linalg
from torchvision import transforms
from PIL import Image

IMAGE_EXTS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

class ImageFolder(torch.utils.data.Dataset):
  def __init__(self, files, transform=None):
    self.files = files
    self.transform = transform
  def __len__(self): return len(self.files)
  def __getitem__(self, idx):
    img = Image.open(self.files[idx]).convert("RGB")
    return self.transform(img) if self.transform else img

class FIDClass:
  def _extract_features(self, loader):
    raise NotImplementedError

  def _stats_from_tensor(self, imgs: torch.Tensor, batch_size: int, verbose=False):
    num_workers = min(os.cpu_count() or 0, 8)

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
      drop_last=False, num_workers=num_workers
    )
    if verbose:
      loader = tqdm.tqdm(loader)
    feats = self._extract_features(loader)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

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

  # ---------------------------------------
  # FOR REFERENCE USE THESE
  # ---------------------------------------
  def _stats_from_files(self, files, batch_size, verbose=False):
    num_workers = min(os.cpu_count() or 0, 8)
    transform = transforms.Compose([
      transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(self.img_size),
      transforms.ToTensor(), 
    ])
    ds = ImageFolder(files, transform=transform)
    loader = torch.utils.data.DataLoader(
      ds, batch_size=batch_size, shuffle=False,
      drop_last=False, num_workers=num_workers
    )
    if verbose:
      loader = tqdm.tqdm(loader)
    feats = self._extract_features(loader)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)

  def _load_stats(self, path: str, batch_size: int):
    path = pathlib.Path(path)
    if path.suffix.lower() == '.npz':
      with np.load(path) as f:
        return f['mu'], f['sigma']
    # Otherwise treat as folder of images
    files = sorted([str(f) for ext in IMAGE_EXTS for f in path.glob(f"*.{ext}")])
    return self._stats_from_files(files, batch_size)