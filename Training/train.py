import sys, os, wandb, click
import warnings
warnings.filterwarnings('ignore')
ROOT = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP'
sys.path.append(ROOT)
with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
  WANDB_KEY = f.read().strip()


def get_params(kwargs):
  p = {}
  p['p_dataset'] = os.path.join(ROOT, 'DATA', 'TRAIN_DATA', f'{kwargs["teacher_name"]}.pkl')
  
  if kwargs['model'] == 'SD15_TRAIN':
    val_size = 100
  elif kwargs['model'] == 'SD14_TRAIN':
    val_size = 100
  elif kwargs['model'] == 'SDXL_TRAIN':
    val_size = 16
  elif kwargs['model'] == 'SORA_TRAIN':
    val_size = 16
  elif kwargs['model'] == 'EDM_TRAIN':
    val_size = 50
  else:
    raise NotImplementedError
  
  p['val_size'] = val_size
    
  return p


@click.command()
@click.option("--device", type=str, required=True)
@click.option("--loss", type=str, default="L1")
@click.option("--train_timesteps", type=bool, default=True)
@click.option("--train_timesteps_unet", type=bool, default=True)
@click.option("--train_solver", type=bool, default=True)
@click.option("--solver", type=str, default="COEFEXT2")
@click.option("--init_solver", type=str, default='DEIS')
@click.option("--model", type=str, default='SDXL_TRAIN')
@click.option("--teacher_name", type=str, default="DEIS40")
@click.option("--lr_ts", type=float, default=0.001)
@click.option("--lr_solv", type=float, default=0.001)
@click.option("--lr_uts", type=float, default=0.001)
@click.option("--lr_ladd", type=float, default=0.0001)
@click.option("--adv_lambda", type=float, default=1)
@click.option("--recon_loss_type", type=str, default='LATENT-L1')
@click.option("--nfe", type=int, default=6)
@click.option("--ts_param_method", type=str, default='square')
@click.option("--ts_start_method", type=str, default='linear')
@click.option("--epochs", type=int, default=400)
@click.option("--train_size", type=int, default=400)
@click.option("--batch_size", type=int, default=10)
@click.option("--mini_batch_size", type=int, default=1)
@click.option("--val_batch_size", type=int, default=1)
@click.option("--calc_metrics_epoch", type=int, default=0)
@click.option("--img_log_interval", type=int, default=5)
@click.option("--log_jacobian", type=bool, default=False)
@click.option("--save_solv_params", type=str, default='')
@click.option("--decode_imgs", type=bool, default=True)
@click.option("--relax_radius", type=float, default=-1)
@click.option("--lr_r", type=float, default=0.01)
def main(**kwargs):
  # os.environ["CUDA_VISIBLE_DEVICES"] = kwargs['device']
  kwargs['device'] = f"cuda:{kwargs['device']}"

  kwargs.update(get_params(kwargs))
  kwargs.update({"save_code": True, "mode": 'online'})

  proj_name = kwargs['model']

  run_name = f"nfe:{kwargs['nfe']}_L:{kwargs['loss']}_S:{kwargs['solver']}_T:{kwargs['teacher_name']}_trS:{kwargs['train_size']},{kwargs['batch_size']}"

  if kwargs['ts_start_method'] != 'linear':
    run_name += f"_tsInit:{kwargs['ts_param_method']}"
  
  if kwargs['train_solver']:
    run_name += f"_init-{kwargs['init_solver']}"
    
  if kwargs['relax_radius'] > 0:
    run_name += f"_R:{kwargs['relax_radius']}"
  
  if kwargs['loss'] == 'LATENT-ADD':
    run_name += f"_LADD:[{round(1e4*kwargs['lr_ladd'], 1)}|{kwargs['adv_lambda']}]({kwargs['recon_loss_type']})"
  
  run_name += f"_({int(kwargs['train_timesteps'])}{int(kwargs['train_timesteps_unet'])}{int(kwargs['train_solver'])}{int(kwargs['relax_radius']>0)})"

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  wandb.login(key=WANDB_KEY, relogin=True)
  wandb.init(project=proj_name, config=kwargs, name=run_name, dir="/workspace-SR008.fs2/philurame/wandb")

  from train_class import Trainer
  trainer = Trainer(wandb.config)
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()