import click, optuna, torch, pickle, sys, os, numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

TS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TS_ROOT)
from ts_utils import *


@torch.inference_mode()
def objective(trial, kwargs):
  n = kwargs['n']
  ddim_features = kwargs['ddim_features']
  pipe = kwargs['pipe']
  nfe  = kwargs['nfe']
  lpips_model = kwargs['lpips_model']
  do_round = kwargs['do_round']

  probs = [trial.suggest_float(f'prob[{i}]', 0, 0.999 if i==0 else 1) for i in range(nfe)]
  timesteps = probs_to_ts(probs, do_round=do_round)
  if timesteps[-1]<0: return float('nan')

  imgs = []
  for i in range(n):
    prompt = ANNS[i]
    seed = torch.Generator(device='cpu').manual_seed(i)
    img_norm = pipe(prompt, timesteps=timesteps, generator=seed, output_type='pt')
    img_norm = torch.nn.functional.interpolate(img_norm, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    imgs.append(img_norm)
  
  imgs = torch.stack(imgs)
  features  = get_features(imgs, lpips_model)

  lpips_score = get_lpips(ddim_features, features, lpips_model).item()
  return lpips_score

def _save_study_callback(study, trial, save_path):
  if (trial.number + 1) % 199 == 0:
    with open(save_path, "wb") as f:
      pickle.dump(study, f)
    print(f"Checkpoint saved: {save_path}")

@click.command()
@click.option('--n', type=int)
@click.option('--solver', type=str)
@click.option('--nfe', type=int)
@click.option('--do_round', type=bool)
def main(**kwargs):
  seed_everything(42)

  DP = os.path.join(TS_ROOT, 'DATA')
  save_path = os.path.join(DP, f"study_n[{kwargs['n']}]_NFE[{kwargs['nfe']}]_SOLVER[{kwargs['solver']}]_ROUND[{kwargs['do_round']}].pkl")
  print(f"{save_path=}\n\n")
  sys.stdout.flush()
  
  kwargs['lpips_model'] = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to('cuda')
  ddim_imgs = torch.load(os.path.join(DP, 'imgs_ddim1000_200.pt'), weights_only=False)[:kwargs['n']]
  ddim_imgs = torch.nn.functional.interpolate(ddim_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
  kwargs['ddim_features'] = get_features(ddim_imgs, kwargs['lpips_model'])
  
  
  kwargs['pipe'] = construct_pipeline(kwargs['solver'], 'CUSTOM', 'BASE')
    
  study = optuna.create_study(direction='minimize')

  base_probs = ts_to_probs(torch.linspace(0, 999, kwargs['nfe'] + 1).round().flip(0)[:-1])
  base_params = {f'prob[{i}]':base_probs[i] for i in range(len(base_probs))}
  study.enqueue_trial(base_params)

  obj = lambda trial: objective(trial, kwargs)
  callb = lambda study, trial: _save_study_callback(study, trial, save_path)
  study.optimize(obj, n_trials=15000, callbacks=[callb], n_jobs=1)

  with open(save_path, 'wb') as f:
    pickle.dump(study, f)


if __name__ == '__main__':
  main()