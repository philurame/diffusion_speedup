import click
import yaml
import neptune
import datetime
from munch import munchify
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registries import import_dir, model_registry, solver_registry, scheduler_registry
import_dir(os.path.join(ROOT, 'lib'))

from lib.datasets.r_coco import COCO_SHORT
from torch.utils.data import DataLoader

from Training.cachers.train_reinforce_utils import reinforce_training_loop

@click.command()
@click.option("--config", type=str, required=True, help="Path to YAML config file")
@click.option("--project_name", type=str, required=True, help="neptune project name in format {workspace}/{project}")
@click.option("--device", type=str, default="cuda")
def main(config, workspace_name, device):
    with open(config, 'r') as f:
        args = yaml.safe_load(f)
    args = munchify(args)
    
    args.device = device
    args.workspace_name = workspace_name
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    args.checkpoint_path = os.path.join('checkpoints/cachers', timestamp)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    if args.name is None:
        key_params = [
            f"steps={args.num_steps}",
            f"ns={args.num_samples}", 
            f"ms={args.max_samples}",
            f"metric={args.metric}",
            f"lr={args.lr}",
            f"gs={args.gs}",
            f"alpha={args.alpha}",
            f"model_variant={args.logit_predictor.model_variant}",
            f"train_logits={args.logit_predictor.train_logits}",
            f"init_logits={args.logit_predictor.init_logits}",
            # f"prompt_extraction_variant={args.logit_predictor.prompt_extraction_variant}"
        ]
        args.name = "_".join(key_params)
    run_name = f"{args.name}_{timestamp}"
    print(f"Generated run name: {run_name}")

    run = neptune.init_run(project=project_name, name=run_name)
    run["parameters"] = dict(args)
    run["artifacts/config.yaml"].upload(config)

    pipe = model_registry[args.model_name].from_pretrained(device=device)
    SolverClass = solver_registry[args.solver]
    SchedulerClass = scheduler_registry[args.scheduler]    
    class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
    pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)

    coco = COCO_SHORT()
    train_dataloader = DataLoader(coco.prompts[:args.max_samples], batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(coco.prompts[-args.max_samples:], batch_size=args.batch_size, shuffle=False)
    test_prompts = coco.prompts[-args.test_size:]

    metric_name = args.metric
    reinforce_training_loop(pipe, train_dataloader, val_dataloader, test_prompts, metric_name, args, run)

    run.stop()

if __name__ == '__main__':
    main()