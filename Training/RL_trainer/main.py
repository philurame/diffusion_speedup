import os, uuid, yaml, click, logging
from pathlib import Path
from omegaconf import OmegaConf
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

  # RL-only config:
  cfg_dict['rl']['train'] = True
  cfg_dict['model'] += '_BASE'

  cfg = OmegaConf.create(cfg_dict) 

  cfg_dict = dict(flatten_cfg(cfg_dict))
  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in cfg_dict.items()],
    '#'*50+'\n', sep='\n', flush=True
  )

  # ───────────────────── 3)  WANDB init  ──────────────────────
  import wandb
  run_name = f"{device}_{uuid.uuid4().hex[:6]}"

  tags = [
    f"nfe:{cfg.nfe}",
    "rl" if cfg.rl.train else "gradient",
    f"loss:{cfg.loss.name}",

    f"ts_train:{int(cfg.timesteps.train)}",
    f"sol_train:{int(cfg.solver.train)}",
        
    f"solver:{cfg.solver.solver}",
    f"solver_guide:{cfg.solver.guide}",
  ]
  
  with open('/workspace-SR008.fs2/philurame/wandb_key.txt', 'r') as f:
    wandb.login(key=f.read().strip(), relogin=True)

  os.environ['WANDB_MODE'] = 'offline'
  wandb.init(project=cfg.model, name=run_name, config=cfg_dict, tags=tags, mode="offline")
  # try:
  #   wandb.init(
  #     project=cfg.model.replace("BASE", "TRAIN"),
  #     name=run_name,
  #     config=cfg_dict,
  #     tags=tags,
  #     dir=str(Path.home() / "wandb"),
  #     mode="online",
  #   )
  # except Exception as e:
  #   wandb.finish()
  #   logging.warning("WANDB init failed, switching to offline: %s", e)
  #   wandb.init(project=cfg.model.replace("BASE", "TRAIN"), name=run_name, config=cfg_dict, tags=tags, mode="offline")

  # ───────────────────── 4)  kick off training  ───────────────
  from trainer import Trainer
  trainer = Trainer(cfg)
  trainer.train()
  wandb.finish()


if __name__ == "__main__":
  import warnings
  warnings.filterwarnings('ignore')
  main()