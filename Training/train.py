import sys, os, wandb, click
from train_class import Trainer
import warnings
warnings.filterwarnings('ignore')
ROOT = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP'
sys.path.append(ROOT)
with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
  WANDB_KEY = f.read().strip()


def get_params(kwargs):
  p = {
    'p_dataset': None,
    'val_size': None,
    }
  
  if kwargs['model'] == 'SDXL_TRAIN':
    p['val_size'] = 16
    p['p_dataset'] = os.path.join(ROOT, 'DATA', 'TRAIN_DATA', f'{kwargs["teacher_name"]}.pkl')

  elif kwargs['model'] == 'EDM_TRAIN':
    p['val_size'] = 50
    p['p_dataset'] = os.path.join(ROOT, 'DATA', 'TRAIN_DATA', f'EDM_{kwargs["teacher_name"]}.pkl')
  
  else:
    raise NotImplementedError
    
  return p


@click.command()
@click.option("--device", type=str)
@click.option("--loss", type=str, default="l2")
@click.option("--train_timesteps", type=bool, default=True)
@click.option("--train_timesteps_unet", type=bool, default=False)
@click.option("--solver", type=str, default="DEIS")
@click.option("--model", type=str, default='SDXL_TRAIN')
@click.option("--teacher_name", type=str, default="DDIM200")
@click.option("--lr_ts", type=float, default=0.005)
@click.option("--lr_solv", type=float, default=0.0005)
@click.option("--lr_uts", type=float, default=0.0005)
@click.option("--nfe", type=int, default=10)
@click.option("--ts_param_method", type=str, default='square')
@click.option("--ts_start_method", type=str, default='linear')
@click.option("--epochs", type=int, default=150)
@click.option("--train_size", type=int, default=20)
@click.option("--batch_size", type=int, default=20)
@click.option("--mini_batch_size", type=int, default=1)
@click.option("--val_batch_size", type=int, default=10)
@click.option("--early_stop", type=int, default=25)
@click.option("--early_stop_metric", type=str, default='lpips')
@click.option("--img_log_interval", type=int, default=5)
@click.option("--log_jacobian", type=bool, default=False)
def main(**kwargs):

  kwargs.update(get_params(kwargs))
  kwargs.update({"save_code": True, "mode": 'online'})

  # proj_name = f"{kwargs['model']}_{kwargs['teacher_name']}"
  
  if kwargs['train_timesteps']:
    run_name  = f"TS_train:{kwargs['train_size']}_nfe:{kwargs['nfe']}_slr:{1000*kwargs['lr_ts']}_L:{kwargs['loss']}"

    if kwargs['model'] == 'SDXL_TRAIN':
      proj_name = f"SDXL_TRAIN_DEIS40"
    elif kwargs['model'] == 'EDM_TRAIN':
      proj_name = f"EDM_TRAIN_TIMESTEPS"
    else:
      raise NotImplementedError

  elif kwargs['train_timesteps_unet']:
    run_name  = f"UTS_train:{kwargs['train_size']}_nfe:{kwargs['nfe']}_slr:{1000*kwargs['lr_uts']}_L:{kwargs['loss']}"
  else:
    run_name  = f"S_{kwargs['solver']}_{kwargs['teacher_name']}_train:{kwargs['train_size']}_nfe:{kwargs['nfe']}_slr:{1000*kwargs['lr_solv']}_L:{kwargs['loss']}"

    if kwargs['model'] == 'SDXL_TRAIN':
      proj_name = f"SOLVER_TEST"
    elif kwargs['model'] == 'EDM_TRAIN':
      proj_name = f"EDM_SOLVER_TEST"
    else:
      raise NotImplementedError


  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  wandb.login(key=WANDB_KEY, relogin=True)
  wandb.init(project=proj_name, config=kwargs, name=run_name)

  trainer = Trainer(wandb.config)
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()