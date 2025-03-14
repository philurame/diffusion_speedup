from lib.registries import metric_registry
import os, pickle, torch, tqdm, scipy
import numpy as np

DATA_DIR = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # lib
  'DATA'
  )

@metric_registry.add_to_registry('CIFAR_FID')
class CIFAR_FID:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    N = kwargs.get('N', 50000)
    pipe = kwargs['pipe']
    nfe  = kwargs['nfe']
    device = pipe.device

    with open(os.path.join(DATA_DIR, 'cifar_reference.pkl'), 'rb') as f:
      ref = pickle.load(f)
      cifar_mu, cifar_sigma = ref['mu'], ref['sigma']

    with open(os.path.join(DATA_DIR, 'inception-2015-12-05.pkl'), 'rb') as f:
      detector_net = pickle.load(f).to(device)
    
    imgs_gen = torch.zeros((N,3,32,32), device='cpu', dtype=torch.uint8)
    for i in tqdm.tqdm(range(0, N, 1000), desc='CIFAR_FID...'):
      generators = [torch.Generator(device='cpu').manual_seed(i*1000+g) for g in range(1000)]
      outp = pipe(num_inference_steps=nfe, generator=generators)
      imgs_gen[i:i+1000] = (outp*127.5+128).clip(0,255).to(torch.uint8)
    
    fid = self._calculate_fid(imgs_gen, cifar_mu, cifar_sigma, detector_net, device=device)
    return fid

  def _calculate_fid(self, imgs_gen, mu_ref, sigma_ref, detector_net, device=torch.device('cuda')):
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    dataloader = torch.utils.data.DataLoader(imgs_gen, batch_size=1000, shuffle=False, num_workers=0)
    for images in tqdm.tqdm(dataloader):
      if images.shape[0] == 0: continue
      if images.shape[1] == 1: images = images.repeat([1, 3, 1, 1])

      features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
      mu += features.sum(0)
      sigma += features.T @ features

    # Calculate grand totals.
    mu /= len(imgs_gen)
    sigma -= mu.ger(mu) * len(imgs_gen)
    sigma /= len(imgs_gen) - 1
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()

    # calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))
