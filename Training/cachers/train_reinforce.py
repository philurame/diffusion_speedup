import click
import yaml
# import wandb
import neptune
import datetime
from munch import Munch
import os, sys, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registries import import_dir, model_registry, metric_registry, solver_registry, scheduler_registry
import_dir(os.path.join(ROOT, 'lib'))

from Training.models import seed_everything
from lib.datasets.r_coco import COCO_SHORT, COCO_LONG
from torch.utils.data import DataLoader

from Training.Cachers_trainer.reinforce_logic import reinforce_training_loop

@click.command()
@click.option("--config", type=str, required=True, help="Path tмне o YAML config file")
@click.option("--device", type=str, default="cuda")
def main(config, device):
    with open(config, 'r') as f:
        args = yaml.safe_load(f)
    args = Munch(args)
    args.device = device
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    if args.name is None:
        key_params = [
            f"steps={args.num_steps}",
            f"ns={args.num_samples}", 
            f"ms={args.max_samples}",
            f"init={args.init_logits}",
            f"metric={args.metric}",
            f"lr={args.lr}",
            f"gs={args.gs}",
        ]
        args.name = "_".join(key_params)
    run_name = f"{args.name}_{timestamp}"
    print(f"Generated run name: {run_name}")
    
    # wandb.login(key="ab888ce1f2f170f823dedfc2ee7cae69cdd33ee0") # hardcoded
    # wandb.init(
    #     project=args.project, 
    #     config=args, 
    #     name=run_name,
    #     id=run_name,
    #     mode="offline"
    # )

    my_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MmRiMzUyNy0xMmIwLTQ1NDUtODQyYS1iNTMxMDk0ZmNkMzEifQ=='
    run = neptune.init_run(project=args.project, api_token=my_token, name=run_name)
    run["parameters"] = dict(args)
    run["artifacts/config.yaml"].upload(config)

    pipe = model_registry[args.model_name].from_pretrained(device=device)
    SolverClass = solver_registry[args.solver]
    SchedulerClass = scheduler_registry[args.scheduler]    
    class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
    pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
    
    # # pipe_baseline: deepcache, чисто для сравнения
    pipe_baseline = model_registry[args.baseline_name].from_pretrained(device=device)
    pipe_baseline.scheduler = SolverSchedulerConstructor(config=pipe_baseline.scheduler_config)

    coco = COCO_SHORT()
    train_dataloader = DataLoader(coco.prompts[:args.max_samples], batch_size=args.train_batch_size, shuffle=False)
    # train_dataloader = DataLoader(coco.prompts[:3], batch_size=args.train_batch_size, shuffle=False) # ПОПРАВИТЬ
    # train_dataloader = DataLoader([coco.prompts[2]], batch_size=1, shuffle=False)
    # print(f"\n\nWORK WITH ONLY PROMPT:\n{coco.anns[2]}\n\n")
    val_dataloader = DataLoader(coco.prompts[-args.max_samples:], batch_size=args.val_batch_size, shuffle=False)

    metric_name = args.metric
    reinforce_training_loop(pipe, pipe_baseline, train_dataloader, val_dataloader, metric_name, args, run)

    # wandb.finish()
    run.stop()

if __name__ == '__main__':
    main()