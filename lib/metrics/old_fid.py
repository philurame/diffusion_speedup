from lib.registries import metric_registry
import torch, tqdm
from torchmetrics.image.fid import FrechetInceptionDistance


@metric_registry.add_to_registry('FID')
class FIDMetric:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    imgs_gen  = kwargs['imgs_gen']
    imgs_real = kwargs['imgs_real']
    device = kwargs['device']

    fid_model = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), desc='FID...'):
      fid_model.update(imgs_real[i:i+batch_size].to(device), real=True)
      fid_model.update(imgs_gen[i:i+batch_size].to(device), real=False)

    return fid_model.compute().item()



import pickle
import numpy as np
import scipy.linalg
from lib.metrics.fid_utils import open_url



      
@metric_registry.add_to_registry('FID1')
class FID1Metric:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    imgs_real = kwargs['imgs_real']
    imgs_gen  = kwargs['imgs_gen']
    device    = kwargs.get('device', torch.device('cuda'))
    batch_size = kwargs.get('batch_size', 512)

    # Load Inception-v3 model (features-only)
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    with open_url(detector_url, verbose=True) as f:
      inception_net = pickle.load(f).to(device).eval()

    def _compute_stats(imgs):
      N = imgs.shape[0]

      if imgs.dtype != torch.uint8:
        if imgs.min() >= -0.1: # [0, 1]
          imgs = (imgs * 255).clip(0,255).to(torch.uint8)
        else: # [-1, 1]
          imgs = ((imgs+1) * 127.5).clip(0,255).to(torch.uint8)

      # 3) Resize to 299×299
      if imgs.shape[-2:] != (299, 299):
        imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
      
      if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)

      # accumulators
      mu = torch.zeros(feature_dim, dtype=torch.float64, device=device)
      sigma = torch.zeros((feature_dim, feature_dim), dtype=torch.float64, device=device)

      for start in range(0, N, batch_size):
        batch = imgs[start:start+batch_size]
        feats = inception_net(batch.to(device), **detector_kwargs).to(torch.float64)
        mu += feats.sum(dim=0)
        sigma += feats.t() @ feats

      mu /= N
      # unbiased covariance
      sigma = (sigma - N * torch.outer(mu, mu)) / (N - 1)
      return mu, sigma


    # compute stats for real and generated
    mu_real, sigma_real = _compute_stats(imgs_real)
    mu_gen,  sigma_gen  = _compute_stats(imgs_gen)

    # compute Frechet distance
    diff = mu_real - mu_gen
    m = (diff * diff).sum().item()

    # scipy expects numpy arrays
    covmean, _ = scipy.linalg.sqrtm(
      (sigma_gen @ sigma_real).cpu().numpy(), disp=False
    )
    # # numerical stability: drop imaginary part
    # if np.iscomplexobj(covmean):
    #   covmean = covmean.real

    trace_term = (
      sigma_gen.trace().item()
      + sigma_real.trace().item()
      - 2.0 * np.trace(covmean)
    )
    fid_value = m + trace_term
    return float(fid_value)