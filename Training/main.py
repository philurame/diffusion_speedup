from setproctitle import setproctitle
setproctitle("@philurame (telegram)") 

import os, uuid, yaml, click, mlflow
from omegaconf import OmegaConf
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from log_utils_mlflow import log_config, install_mlflow_guards

def flatten_cfg(d, prefix=""):
  for k, v in d.items():
    key = f"{prefix}.{k}" if prefix else k
    if isinstance(v, dict):
      yield from flatten_cfg(v, key) # recursive
    else:
      yield key, v

@click.command()
@click.option("--config",  "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--device",  "-d", required=True, help="CUDA device ordinal, e.g. 0")
def main(config: str, device: str):
  # ───────────────────── 2)  basic setup  ─────────────────────
  # os.environ["CUDA_VISIBLE_DEVICES"] = device # this throws, somehow

  with open(config) as f:
    cfg_dict: dict = yaml.safe_load(f) 

  cfg_dict["device"] = f"cuda:{device}"

  if 'rl' not in cfg_dict or 'train' not in cfg_dict['rl']:
    cfg_dict['rl'] = {'train': False}

  exp_name = cfg_dict['model']
  if cfg_dict['rl']['train']:
    cfg_dict['model'] += '_BASE'
  else:
    cfg_dict['model'] += '_TRAIN'
  
  run_name = f"{device}_{uuid.uuid4().hex[:6]}"
  cfg_dict['run_id'] = run_name

  cfg = OmegaConf.create(cfg_dict) 

  cfg_dict = dict(flatten_cfg(cfg_dict))
  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in cfg_dict.items()],
    '#'*50+'\n', sep='\n', flush=True
  )
  tags = {
    "nfe": cfg.nfe,
    "use_rl": cfg.rl.train == True,
    "loss": cfg.loss.name,

    "ts_train": cfg.timesteps.train,
    "sol_train": cfg.solver.train,
        
    "solver": cfg.solver.solver,
    "solver_guide": cfg.solver.guide,

    "device": f"cuda:{device}"
  }
  tags = {k:str(v) for k,v in tags.items()}

  mlflow.set_tracking_uri("file:/workspace-SR008.fs2/philurame/MLFLOW")
  mlflow.set_experiment(f"{exp_name}")

  from trainer import Trainer
  
  with mlflow.start_run(run_name=run_name, tags=tags, log_system_metrics=True) as run:
    install_mlflow_guards() # ends run if killed
    log_config(cfg_dict)
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
  import warnings
  warnings.filterwarnings('ignore')
  main()


