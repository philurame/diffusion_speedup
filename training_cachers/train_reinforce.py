import click
import yaml
import wandb
import datetime
from munch import Munch
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.registries import import_dir, model_registry, metric_registry, solver_registry, scheduler_registry
import_dir(os.path.join(ROOT, 'lib'))

from lib.generate_decode import seed_everything
from lib.datasets.r_coco import COCO
from torch.utils.data import DataLoader

from training_cachers.reinforce_logic import reinforce_training_loop


@click.command()
@click.option("--config", type=str, required=True, help="Path to YAML config file")
@click.option("--device", type=str, default="cuda")
def main(config, device):
    with open(config, 'r') as f:
        args = yaml.safe_load(f)
    args = Munch(args)
    args.device = device
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y.%m.%d-%H:%M:%S")
    run_name = f"{args.name}_{timestamp}"
    
    wandb.login(key="ab888ce1f2f170f823dedfc2ee7cae69cdd33ee0") # hardcoded
    wandb.init(project=args.project, config=args, name=run_name) #, mode="disabled")

    pipe = model_registry[args.model_name].from_pretrained(device=device)
    SolverClass = solver_registry[args.solver]
    SchedulerClass = scheduler_registry[args.scheduler]    
    class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
    pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)

    coco = COCO(data_path=os.path.join(ROOT, 'DATA'), max_samples=10000)
    train_dataloader = DataLoader(coco.anns[:args.max_samples], batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(coco.anns[-args.max_samples:], batch_size=args.batch_size, shuffle=False)

    metric_name = args.metric
    reinforce_training_loop(pipe, train_dataloader, val_dataloader, metric_name, args)

    wandb.finish()

if __name__ == '__main__':
    main()