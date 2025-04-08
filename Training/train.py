import sys, os, wandb, click
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

  elif kwargs['model'] == 'SORA_TRAIN':
    p['val_size'] = 16
    p['p_dataset'] = os.path.join(ROOT, 'DATA', 'TRAIN_DATA', f'{kwargs["teacher_name"]}.pkl')

  elif kwargs['model'] == 'EDM_TRAIN':
    p['val_size'] = 50
    p['p_dataset'] = os.path.join(ROOT, 'DATA', 'TRAIN_DATA', f'EDM_{kwargs["teacher_name"]}.pkl')
  
  else:
    raise NotImplementedError
    
  return p


@click.command()
@click.option("--device", type=str, required=True)
@click.option("--loss", type=str, default="l2lpips")
@click.option("--train_timesteps", type=bool, default=True)
@click.option("--train_timesteps_unet", type=bool, default=False)
@click.option("--solver", type=str, default="COEFMATRIX1")
@click.option("--model", type=str, default='SDXL_TRAIN')
@click.option("--teacher_name", type=str, default="DEIS40")
@click.option("--lr_ts", type=float, default=0.005)
@click.option("--lr_solv", type=float, default=0.001)
@click.option("--lr_uts", type=float, default=0.001)
@click.option("--nfe", type=int, default=10)
@click.option("--ts_param_method", type=str, default='square')
@click.option("--ts_start_method", type=str, default='linear')
@click.option("--epochs", type=int, default=400)
@click.option("--train_size", type=int, default=50)
@click.option("--batch_size", type=int, default=1)
@click.option("--mini_batch_size", type=int, default=1)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--early_stop", type=int, default=100)
@click.option("--early_stop_metric", type=str, default='lpips')
@click.option("--img_log_interval", type=int, default=5)
@click.option("--log_jacobian", type=bool, default=False)
@click.option("--save_solv_params", type=str, default='')
def main(**kwargs):
  # os.environ["CUDA_VISIBLE_DEVICES"] = kwargs['device']
  kwargs['device'] = f"cuda:{kwargs['device']}"

  kwargs.update(get_params(kwargs))
  kwargs.update({"save_code": True, "mode": 'online'})

  if kwargs['solver'] not in ['DEIS', 'DPMS']:
    run_name = f"{kwargs['solver']}_{kwargs['teacher_name']}_train:{kwargs['train_size']},{kwargs['batch_size']}_nfe:{kwargs['nfe']}_slr:{int(1000*kwargs['lr_solv'])}_L:{kwargs['loss']}"
    proj_name = f"SOLVER_TEST"
  
  if kwargs['train_timesteps']:
    run_name = f"{kwargs['solver']}_{kwargs['teacher_name']}_train:{kwargs['train_size']},{kwargs['batch_size']}_nfe:{kwargs['nfe']}_slr:{int(1000*kwargs['lr_ts'])},{int(1000*kwargs['lr_solv'])}_L:{kwargs['loss']}_ts:{kwargs['ts_start_method']}"
    proj_name = "SDXL_TRAIN_TS_SOLVER"

  if kwargs['train_timesteps_unet']:
    run_name  = f"{kwargs['solver']}_{kwargs['teacher_name']}_train:{kwargs['train_size']},{kwargs['batch_size']}_nfe:{kwargs['nfe']}_slr:{int(1000*kwargs['lr_uts'])},{int(1000*kwargs['lr_ts'])},{int(1000*kwargs['lr_solv'])}_L:{kwargs['loss']}_ts:{kwargs['ts_start_method']}"
    proj_name = "SDXL_TRAIN_TS_SOLVER"

  if kwargs['model'] == 'EDM_TRAIN':
    proj_name = f"EDM_SOLVER_TEST"
  
  if kwargs['model'] == 'SORA_TRAIN':
    proj_name = f"SORA_TRAIN"

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  wandb.login(key=WANDB_KEY, relogin=True)
  wandb.init(project=proj_name, config=kwargs, name=run_name)

  from train_class import Trainer
  trainer = Trainer(wandb.config)
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()