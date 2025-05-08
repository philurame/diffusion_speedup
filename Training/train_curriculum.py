import sys, os, wandb, click
from train_curriculum_class import TrainerCurriculum
import warnings
warnings.filterwarnings('ignore')
ROOT = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP'
sys.path.append(ROOT)
with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
  WANDB_KEY = f.read().strip()

@click.command()
@click.option("--device", type=str)
@click.option("--solver", type=str, default="DEIS")
@click.option("--teacher_schedule", type=str, default="LINEAR")
@click.option("--teacher_start", type=int, default=10)
@click.option("--teacher_step", type=int, default=1)
@click.option("--teacher_epochs", type=int, default=50)
@click.option("--lr_solv", type=float, default=0.001)
@click.option("--lr_decay", type=float, default=1.)
@click.option("--nfe", type=int, default=10)
@click.option("--epochs", type=int, default=400)
@click.option("--train_size", type=int, default=20)
@click.option("--batch_size", type=int, default=20)
@click.option("--mini_batch_size", type=int, default=1)
@click.option("--val_batch_size", type=int, default=10)
@click.option("--img_log_interval", type=int, default=1)
@click.option("--ts_start_method", type=str, default='linear')
@click.option("--loss", type=str, default="l2")
@click.option("--log_jacobian", type=bool, default=False)
@click.option("--train_timesteps", type=bool, default=False)
@click.option("--train_timesteps_unet", type=bool, default=False)
def main(**kwargs):

  kwargs['val_size'] = min(16, kwargs['train_size'])

  kwargs.update({"save_code": True, "mode": 'online'})

  proj_name = f"SDXL_CURRICULUM_TEST"
  
  run_name  = f"{kwargs['solver']}-{kwargs['nfe']}_teacher:{kwargs['teacher_schedule']}{kwargs['teacher_start']}_train:{kwargs['train_size']}_slr:{1000*kwargs['lr_solv']}*{kwargs['lr_decay']}"

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in kwargs.items()],
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  wandb.login(key=WANDB_KEY, relogin=True)
  wandb.init(project=proj_name, config=kwargs, name=run_name)

  trainer = TrainerCurriculum(wandb.config)
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()