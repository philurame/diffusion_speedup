
import click
import yaml
from munch import Munch, munchify
import os
import sys
import torch
import gc
import neptune

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registries import import_dir, metric_registry, model_registry, solver_registry, scheduler_registry
import_dir(os.path.join(ROOT, 'lib'))

from lib.datasets.r_coco import COCO_SHORT
from torch.utils.data import DataLoader

from Training.cachers.train_reinforce_utils import generate_data     
from Training.models import seed_everything

# ROOT = '/home/jovyan/maliev/DIFFUSION_SPEEDUP'
    
    
def calc_metrics(metric_names, **data):
        '''calculate metrics by looping over all metrics in metrics_registry''' 
        res_metrics = {}
        for metric_name in metric_names:
            metricInstance = metric_registry[metric_name]()
            res_metrics[metric_name] = metricInstance(**data)

            gc.collect()
            torch.cuda.empty_cache()

            print(f'{metric_name}: {res_metrics[metric_name]}', flush=True)
        return res_metrics
    
    
def log_teacher(pipe, helper, test_noise, args, test_prompts):
    run = neptune.init_run(project=args.project_name)
    run["parameters"] = dict(args)
    
    metric_names = ['AQ', 'IQ', 'HPS', 'CLIP', 'ImageReward']
    test_dataloader = DataLoader(test_prompts, batch_size=args.batch_size, shuffle=False)
    
    # TEACHER MODEL
    teacher_path = os.path.join(
        ROOT, 'DATA', 'cachers', 'eval_data', 
        f"teacher_test_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.test_size}.pt"
    )
    imgs_teacher = generate_data(
        pipe, helper, [], test_dataloader, test_noise, teacher_path, args, 
        args.teacher_nfe, "teacher test data", is_test=True
    )
    
    metrics = calc_metrics(
        metric_names=metric_names,
        imgs_gen=imgs_teacher,
        prompts=test_prompts,
        device=args.device,
    )
    run["metrics"].append(metrics, step=0)
    
    run.stop()
    
    return imgs_teacher
    
    
def log_deepcache(pipe, helper, test_noise, args, test_prompts, imgs_teacher):
    run = neptune.init_run(project=args.project_name, custom_run_id='DEEPCACHE3')
    run["parameters"] = dict(args)
    
    metric_names = ['LPIPS', 'PLPIPS', 'L1', 'AQ', 'IQ', 'HPS', 'CLIP', 'ImageReward']
    test_dataloader = DataLoader(test_prompts, batch_size=args.batch_size, shuffle=False)
    
    # DEEPCACHE3 TEST DATA
    baseline_path = os.path.join(
        ROOT, "DATA", "cachers", "eval_data", 
        f"deepcache_3_test_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.test_size}.pt"
    )
    not_cached_steps = list(range(0, int(args.teacher_nfe), 3))
    ts_to_skip = sorted(set(list(range(args.student_nfe))) - set(not_cached_steps))
    
    imgs_baseline = generate_data(
        pipe, helper, ts_to_skip, test_dataloader, test_noise, baseline_path, args,
        args.student_nfe, dataset_description="DEEPCACHE3 test data", is_test=True
    )
    
    metrics = calc_metrics(
        metric_names=metric_names,
        imgs_gen=imgs_baseline,
        imgs_real=imgs_teacher,
        prompts=test_prompts,
        device=args.device,
    )
    run["metrics"].append(metrics, step=0)
    run.stop()
    
    
@click.command()
@click.option("--config", type=str, required=True, help="Path to YAML config file")
@click.option("--project_name", type=str, required=True, help="neptune project name in format {workspace}/{project}")
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=42)
def main(config, project_name, device, seed):
    seed_everything(seed)
    
    with open(config, 'r') as f:
        args = yaml.safe_load(f)
    args = munchify(args)
    
    args.device = device
    args.seed = seed
    args.project_name = project_name

    pipe = model_registry[args.model_name].from_pretrained(device=device)
    SolverClass = solver_registry[args.solver]
    SchedulerClass = scheduler_registry[args.scheduler]    
    class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
    pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
    
    helper = pipe.cacher

    coco = COCO_SHORT()
    test_prompts = coco.prompts[-args.test_size:]
    
    test_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"test_noise_{args.test_size}.pt")
    test_noise = torch.load(test_noise_path, weights_only=True, map_location='cpu').to(args.device)
    
    imgs_teacher = log_teacher(pipe, helper, test_noise, args, test_prompts)
    log_deepcache(pipe, helper, test_noise, args, test_prompts, imgs_teacher)

if __name__ == '__main__':
    main()
