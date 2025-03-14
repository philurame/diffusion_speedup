import torch, os, random, sys, wandb
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# =============================================================================
# PIPE
# =============================================================================
ROOT = os.path.dirname( os.path.dirname(os.path.abspath(__file__)) )
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from lib.registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  model_registry, 
  data_registry, 
)
import_dir(os.path.join(ROOT, 'lib'))

ANNS = data_registry['COCO'](os.path.join(ROOT, 'DATA'), max_samples=10_000).anns

def construct_pipeline(solver, scheduler, model_name, half=True, **pipe_kwargs):
  '''construct a pipeline with given model_name, solver and scheduler'''
  PipeClass   = model_registry[model_name]
  SolverClass = solver_registry[solver]
  SchedulerClass = scheduler_registry[scheduler]
  pipe = PipeClass.from_pretrained(half=half, **pipe_kwargs)
  class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
  pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
  return pipe

# =============================================================================
# TS PARAMETRIZATIONS
# =============================================================================
class TSParam:
  def __init__(self, method):
    if method not in ['cumprod', 'softmax', 'softplus', 'square']: raise NotImplementedError
    self.method = method
  
  def __call__(self, logits, **kwargs):
    '''
    cumprod: R^d -> R^d
    others:  R^d -> R^{d-1}
    '''
    if isinstance(logits, list): ts = torch.tensor(logits)
    if self.method == 'cumprod':
      if kwargs.get('use_sigmoid', True):
        logits = F.sigmoid(logits)
      ts = 1000*torch.cumprod(logits, 0)

    elif self.method == 'softmax':
      cum_probs = torch.cumsum(F.softmax(logits, dim=0), dim=0)
      ts = (999 - cum_probs * 999)[:-1]

    elif self.method == 'softplus':
      pos_logits = F.softplus(logits)
      cum_probs = torch.cumsum(pos_logits, dim=0)
      cum_probs = cum_probs / cum_probs[-1]
      ts = (999 - cum_probs * 999)[:-1]

    elif self.method == 'square':
      pos_logits = logits**2
      cum_probs = torch.cumsum(pos_logits, dim=0)
      cum_probs = cum_probs / cum_probs[-1]
      ts = (999 - cum_probs * 999)[:-1]
    
    return ts

  def get_logits(self, ts, **kwargs):
    if isinstance(ts, list): ts = torch.tensor(ts)
    if self.method == 'cumprod':
      logits = ts.clone() / 1000.0
      for i in range(1, len(ts)):
        logits[i] = ts[i] / ts[i-1]
      if kwargs.get('use_sigmoid', True):
        logits = torch.log(logits) - torch.log(1 - logits)
      return logits
    
    elif self.method == 'softmax':
      probs = torch.cat([(999 - ts) / 999., torch.tensor([1.], device=ts.device, dtype=ts.dtype)]).clone()
      probs[1:] = probs[1:] - probs[:-1] 
      return torch.log(probs)
    
    elif self.method == 'softplus':
      probs = torch.cat([(999 - ts) / 999, torch.tensor([1.], device=ts.device, dtype=ts.dtype)]).clone()
      probs[1:] = probs[1:] - probs[:-1]
      return torch.log(torch.exp(probs) - 1)
    
    elif self.method == 'square':
      probs = torch.cat([(999 - ts) / 999, torch.tensor([1.], device=ts.device, dtype=ts.dtype)]).clone()
      probs[1:] = probs[1:] - probs[:-1]
      return probs**.5

def probs_to_ts(probs, do_round=False):
  '''for optuna'''
  if isinstance(probs, list): probs = np.array(probs)
  ts = 1000*np.cumprod(probs, 0)
  if do_round: ts = [round(t) for t in ts]
  for i in range(1, len(ts)): # fix repeating timesteps
    if abs(ts[i]-ts[i-1]) < 1: ts[i] = ts[i-1] - (1 if do_round else 1e-1)
  return ts
def ts_to_probs(ts):
  '''for optuna'''
  if isinstance(ts, list): ts = np.array(ts)
  ts = ts.copy() / 1000.0
  probs = np.empty_like(ts)
  probs[0] = ts[0]
  for i in range(1, len(ts)):
    probs[i] = ts[i] / ts[i - 1]
  return probs


# =============================================================================
# LPIPS
# =============================================================================
def lpips(imgs1, imgs2, lpips_net):
  m, M = min(imgs1.min(), imgs2.min()), max(imgs1.max(), imgs2.max())
  if m >= -0.01 and M < 2: # [0, 1]
    imgs1 = imgs1 * 2 - 1
    imgs2 = imgs2 * 2 - 1
  elif m >= -0.01: # uint8, [0, 255]
    imgs1 = imgs1 / 127.5 - 1
    imgs2 = imgs2 / 127.5 - 1
  feats1 = get_features(imgs1, lpips_net)
  feats2 = get_features(imgs2, lpips_net)
  return get_lpips(feats1, feats2, lpips_net)

def get_features(imgs, lpips_net):
  device = next(lpips_net.parameters()).device
  outs_net = lpips_net.net.forward(lpips_net.scaling_layer(imgs.to(device)))

  feats = tuple(_normalize_tensor(feat) for feat in outs_net)
  return feats

def get_lpips(feats1, feats2, lpips_net, reduction='mean'):
  device = next(lpips_net.parameters()).device
  feats1 = [f.to(device) for f in feats1]
  feats2 = [f.to(device) for f in feats2]

  total_loss = torch.tensor(0.0, device=device)
  for f1, f2, lin in zip(feats1, feats2, lpips_net.lins):
    diff = (f1 - f2)**2
    total_loss += lin(diff).mean(dim=[2, 3], keepdim=True).sum()
    
  if reduction == 'mean':
    return total_loss / feats1[0].shape[0]
  elif reduction == 'sum':
    return total_loss

def _normalize_tensor(in_feat, eps=1e-8):
  return in_feat / torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))



# =============================================================================
# WANDB LOGGING SPECIAL
# =============================================================================

def wandb_log_fig(fig, key, global_step):
  fig.tight_layout()
  wandb.log({key: wandb.Image(fig)}, step=global_step)
  plt.close('all')

def wandb_log_ts(t_steps, global_step=None, key=None, xlabel="NFE", ylabel="TS"):
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.plot(t_steps)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid()
  if global_step is None: return
  wandb_log_fig(fig=fig, key=key, global_step=global_step)

def wandb_log_imgs(imgs_student, imgs_teacher, global_step=None, key=None):
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  vis_grid(imgs_student, ax=ax[0])
  ax[0].axis('off')
  ax[0].set_title("Student")

  vis_grid(imgs_teacher, ax=ax[1])
  ax[1].axis('off')
  ax[1].set_title("Teacher")
  if global_step is None: return

  wandb_log_fig(fig=fig, key=key, global_step=global_step)

def vis_grid(imgs_row, ax=None):
  imgs_row = imgs_row.detach().cpu()
  nrow = int(np.around(np.sqrt(imgs_row.shape[0])))
  imgs_grid = make_grid(imgs_row, nrow=nrow).permute(1, 2, 0).numpy()

  imgs_grid = imgs_grid / 2 + 0.5
  imgs_grid = np.clip(imgs_grid, 0, 1)
  if ax is None:
    plt.imshow(imgs_grid)
  else:
    ax.imshow(imgs_grid)

# =============================================================================
# EDM
# =============================================================================
import pickle, tqdm, scipy
DATA_DIR = os.path.join(ROOT, 'DATA')

class CIFAR_FID:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    N = kwargs.get('N', 50000)
    pipe = kwargs['pipe']
    timesteps = kwargs['timesteps']
    device = pipe.device

    with open(os.path.join(DATA_DIR, 'cifar_reference.pkl'), 'rb') as f:
      ref = pickle.load(f)
      cifar_mu, cifar_sigma = ref['mu'], ref['sigma']

    with open(os.path.join(DATA_DIR, 'inception-2015-12-05.pkl'), 'rb') as f:
      detector_net = pickle.load(f).to(device)
    
    imgs_gen = torch.zeros((N,3,32,32), device='cpu', dtype=torch.uint8)
    for i in tqdm.tqdm(range(0, N, 1000), desc='CIFAR_FID...'):
      generators = [torch.Generator(device='cpu').manual_seed(i*1000+g) for g in range(1000)]
      outp = pipe(timesteps=timesteps, generator=generators)
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
