import warnings

import torch
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# RLLoss wrapper
# ---------------------------------------------------------------------------
class RLLoss(torch.nn.Module):
  '''
  wraps any simple loss class into RLLoss
  '''
  def __init__(self, metric):
    super().__init__()
    self.metric = metric

  def forward(self, logprob, gen_latents_samples, gen_imgs_samples, **kwargs):

    if self.metric.is_latent:
      gen_samples = gen_latents_samples
    else:
      gen_samples = gen_imgs_samples

    losses = [self.metric(s, **kwargs).detach() for s in gen_samples]

    losses = torch.stack(losses, dim=0).to(logprob.device) # [num_samples, batch_size]
    losses_sample_mean  = losses.mean(dim=0) # [batch_size]
    losses_corrected = (losses - losses_sample_mean[None, :]).detach() # [num_samples, batch_size]

    num_samples = losses.shape[0]
    rl_loss = (losses_corrected * logprob[:, None]) * (num_samples) / (num_samples - 1)
    rl_loss = rl_loss.mean(dim=0).sum()

    return rl_loss, losses_sample_mean.sum()

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_MIN_TIMESTEP_GAP = 0.15
_TIMESTEP_RANGE = (0.0, 1000.0)

def _valid_timestep_sequence(seq: torch.Tensor) -> bool:
  """Return *True* if `seq` is within range and strictly increasing by ≥ `_MIN_TIMESTEP_GAP`."""
  if seq is None or seq.numel() == 0: 
    return True
  if (seq < _TIMESTEP_RANGE[0]).any() or (seq > _TIMESTEP_RANGE[1]).any():
    return False
  return (seq[1:] - seq[:-1] >= _MIN_TIMESTEP_GAP).all()


def _valid_all(seqs) -> bool:
  return all(_valid_timestep_sequence(s) for s in seqs)


# ---------------------------------------------------------------------------
# RL sampling
# ---------------------------------------------------------------------------

_MAX_RESAMPLE = 1_000


def rl_sample(config, pipe, timesteps_model, rl_logits, prompts, noise, **kwargs):
  """
  Draw ``num_samples`` sets of RL parameters and run ``pipe`` to produce latents.

  Returns
  -------
  gen_latents
    A list with one latent per sample.
  log_probs
    The log-probability of each sample under the Normal proposal distribution.
  """
  num_samples = config.rl.num_samples

  mean = _build_mean_vector(config, timesteps_model, pipe)
  std = torch.exp(rl_logits)
  dist = Normal(mean, std)

  for _ in range(_MAX_RESAMPLE):
    flat_samples = dist.sample((num_samples,))
    timesteps, unet_ts, solver = _split_samples(config, flat_samples, timesteps_model)

    if _valid_all(timesteps) and _valid_all(unet_ts):
      break
  else:
    warnings.warn("rl_sample: giving up after MAX_RESAMPLE attempts")

  if config.solver.train:
    # original_solver_params = list(pipe.scheduler.train_params)
    original_solver_params = pipe.scheduler.get_flat_params()

  gen_latents = []
  with torch.no_grad():
    for i in range(num_samples):
      if config.solver.train:
        # _apply_solver_params(pipe, solver[i])
        pipe.scheduler.set_flat_params(solver[i])

      latent = pipe(
        prompt=prompts, 
        timesteps=timesteps[i],
        unet_timesteps=unet_ts[i],
        latents=noise.to(pipe.device, dtype=torch.float16),
        output_type="latent",
      )
      gen_latents.append(latent)

  # restore scheduler state
  if config.solver.train:
    # pipe.scheduler.train_params = original_solver_params
    pipe.scheduler.set_flat_params(original_solver_params)

  log_probs = dist.log_prob(flat_samples).sum(dim=1).to(pipe.device)

  return gen_latents, log_probs

# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _build_mean_vector(config, timesteps_model, pipe) -> torch.Tensor:
  pieces = []
  if config.timesteps.train:
    pieces.extend([timesteps_model.timesteps_logits, timesteps_model.unet_timesteps_logits])
  if config.solver.train:
    pieces.extend(pipe.scheduler.train_params)
  return torch.cat([p.to(pipe.device) for p in pieces])

def _split_samples(config, sample_logits, timesteps_model):
  """Convert a batch of flat samples into structured parameters."""
  nfe = config.nfe
  timesteps_logits, unet_timesteps_logits, solver_params = [], [], []

  for row in sample_logits:
    idx = 0
    if config.timesteps.train:
      ts_logits  = row[idx : idx + nfe]; idx += nfe
      uts_logits = row[idx : idx + nfe]; idx += nfe
      ts, uts = timesteps_model(ts_logits, uts_logits)
    else:
      ts = uts = timesteps_model.timesteps

    if config.solver.train:
      solver = row[idx:]
    else:
      solver = None

    timesteps_logits.append(ts)
    unet_timesteps_logits.append(uts)
    solver_params.append(solver)

  return timesteps_logits, unet_timesteps_logits, solver_params

# def _apply_solver_params(pipe, flat_params: torch.Tensor) -> None:
#   """Copy `flat_params` into `pipe.scheduler.train_params`"""
#   idx=0
#   for n, p in enumerate(pipe.scheduler.train_params):
#     pipe.scheduler.train_params[n] = flat_params[idx:idx+p.numel()].view_as(p)
#     idx += p.numel()
