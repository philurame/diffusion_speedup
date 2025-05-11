import sys, os, wandb, click, yaml
import warnings
warnings.filterwarnings('ignore')
ROOT = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP'
sys.path.append(ROOT)
with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
  WANDB_KEY = f.read().strip()


@click.command()
@click.option("--device", type=str, required=True, help="CUDA device name")
@click.option("--config", type=str, required=True, help="YAML config path")
def main(**kwargs):
  with open(kwargs['config'], "r") as f:
    config = yaml.safe_load(f)

  # os.environ["CUDA_VISIBLE_DEVICES"] = kwargs['device'] # preventing leaks
  config['device'] = f"cuda:{kwargs['device']}"
  config.update({"save_code": True, "mode": 'online'})
  proj_name = config['model']
  run_name = f"nfe:{config['nfe']}_L:{config['loss']}_S:{config['solver']['solver']}_T:{config['solver']['teacher']}"

  if config['dataset']['train_size']!=1000 and config['dataset']['batch_size']!=10:
    run_name += f"_B:{config['dataset']['train_size']},{config['dataset']['batch_size']}"
  
  if config['solver']['train']:
    run_name += f"_I-{config['solver']['init']}"
  
  if config['loss'] == 'LATENT-ADV':
    run_name += f"_ADV:[{round(1e3*config['adv']['lr'], 1)}|{config['adv']['lambda']}]"
  
  if config['loss'] == 'LATENT-ADD':
    run_name += f"_ADD:[{round(1e3*config['adv']['lr'], 1)}|{config['adv']['lambda']}|{int(config['adv']['freeze'])}]"
  
  lrs = [config[i]['lr'] for i in ['timesteps', 'unet_timesteps', 'solver', 'adv']]
  if lrs != [0.001]*len(lrs):
    run_name += '_lr['+'|'.join([f'{i/0.001}' for i in lrs])+']'

  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in config.items()],
    '#'*50+'\n', sep='\n'
  )
  sys.stdout.flush()

  wandb.login(key=WANDB_KEY, relogin=True)
  wandb.init(project=proj_name, config=config, name=run_name, dir="/workspace-SR008.fs2/philurame/wandb")

  from train_class import Trainer
  trainer = Trainer(wandb.config)
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()