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

# def calculate_inception_stats(
#     image_path, num_expected=None, seed=0, max_batch_size=64,
#     num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
# ):
#     # Rank 0 goes first.
#     if dist.get_rank() != 0:
#         torch.distributed.barrier()

#     # Load Inception-v3 model.
#     # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
#     dist.print0('Loading Inception-v3 model...')
#     detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
#     detector_kwargs = dict(return_features=True)
#     feature_dim = 2048
#     with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
#         detector_net = pickle.load(f).to(device)

#     # List images.
#     dist.print0(f'Loading images from "{image_path}"...')
#     dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
#     assert len(dataset_obj) in [10000, 30000, 50000]
#     if num_expected is not None and len(dataset_obj) < num_expected:
#         raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
#     if len(dataset_obj) < 2:
#         raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

#     # Other ranks follow.
#     if dist.get_rank() == 0:
#         torch.distributed.barrier()

#     # Divide images into batches.
#     num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
#     all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
#     rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
#     data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

#     # Accumulate statistics.
#     dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
#     mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
#     sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
#     for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
#         torch.distributed.barrier()
#         if images.shape[0] == 0:
#             continue
#         if images.shape[1] == 1:
#             images = images.repeat([1, 3, 1, 1])
#         features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
#         mu += features.sum(0)
#         sigma += features.T @ features

#     # Calculate grand totals.
#     torch.distributed.all_reduce(mu)
#     torch.distributed.all_reduce(sigma)
#     mu /= len(dataset_obj)
#     sigma -= mu.ger(mu) * len(dataset_obj)
#     sigma /= len(dataset_obj) - 1
#     return mu.cpu().numpy(), sigma.cpu().numpy()

# #----------------------------------------------------------------------------

# def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
#     m = np.square(mu - mu_ref).sum()
#     s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
#     fid = m + np.trace(sigma + sigma_ref - s * 2)
#     return float(np.real(fid))



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