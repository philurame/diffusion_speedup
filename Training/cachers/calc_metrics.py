import os
import sys
import gc
import click
import torch
import random
import numpy as np
import neptune
from datetime import datetime 
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registries import (
  import_dir, 
  solver_registry, 
  scheduler_registry, 
  model_registry, 
  metric_registry
)

import_dir(os.path.join(ROOT, 'lib'))
from lib.datasets.r_coco import COCO_SHORT

METRIC_COLUMNS = ['LPIPS', 'PLPIPS', 'L1', 'AQ', 'IQ', 'HPS', 'CLIP', 'ImageReward']

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_steps(spec: str):
    return [int(s.strip()) for s in spec.split(',') if s.strip() != ""]

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

def generate_data(pipe, helper, ts_to_skip, test_dataloader, test_noise, data_path, nfe, gs, batch_size):

    seed_everything()

    not_cached = sorted(set(range(nfe)) - set(ts_to_skip))

    print("\n\tNOT CACHED:", not_cached)
    print("\tCACHED:", ts_to_skip)

    if not os.path.exists(data_path):

        outputs = []
        with torch.inference_mode():
            for batch_idx, anns in enumerate(tqdm(test_dataloader)):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                current_noise = test_noise[batch_start:batch_end]
                
                with helper.inference(timesteps=ts_to_skip):
                    gen = pipe(
                        anns,
                        num_inference_steps=nfe,
                        guidance_scale=gs,
                        latents=current_noise,
                        output_type="pt"
                    )
                outputs.append(gen)

        generated = torch.cat(outputs, dim=0).cpu()
        generated = (generated + 1) * 0.5           # [-1,1] -> [0,1]
        torch.save(generated, data_path)
        print(f"\tDATA SAVED TO: {data_path}")

    else:
        generated = torch.load(data_path, weights_only=True)
        assert len(test_noise) == len(generated)
        print(f"\tDATA LOADED FROM: {data_path}")

    return generated

@click.command()
@click.option('--model_name',   type=str,   required=True, default="REINFORCE_CACHER",                            help='which pipe to use')
@click.option('--solver',       type=str,   required=True, default="DDIM",                                        help='supported methods are in lib/solvers')
@click.option('--scheduler',    type=str,   required=True, default="LINEAR",                                      help='supported methods are in lib/schedulers')
@click.option('--nfe',          type=int,   required=True, default=25,                                            help='num inference steps')
@click.option('--gs',           type=float, required=True, default=5,                                             help='guidance scale')
@click.option('--test_size',    type=int,   required=True, default=1000,                                          help='size of test data')
@click.option('--batch_size',   type=int,   required=True, default=8,                                             help='size of test data batch')
@click.option('--metric_names', type=str,   required=True, default="LPIPS,PLPIPS,L1,AQ,IQ,HPS,CLIP,ImageReward",  help='list of metrics separated by comma')
@click.option("--not_cached",   type=str,   required=True, default="",)
@click.option("--device",       type=str,                  default="cuda")
def main(**kwargs):

    model_name          = kwargs['model_name']
    solver              = kwargs['solver']
    scheduler           = kwargs['scheduler']
    nfe                 = kwargs['nfe']
    gs                  = kwargs['gs']
    test_size           = kwargs['test_size']
    batch_size          = kwargs['batch_size']
    metric_names        = kwargs['metric_names'].split(',')
    not_cached_steps    = parse_steps(kwargs['not_cached'])
    device              = kwargs['device']
    

    ### MODEL INITIALIZATION

    seed_everything()

    pipe = model_registry[model_name].from_pretrained(device=device)
    SolverClass = solver_registry[solver]
    SchedulerClass = scheduler_registry[scheduler]
    class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
    pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
    helper = pipe.cacher

    coco = COCO_SHORT()
    prompts = coco.prompts[-test_size:]
    test_dataloader = DataLoader(coco.prompts[-test_size:], batch_size=batch_size, shuffle=False)

    latent_size = pipe.unet.config.sample_size
    test_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises",  f"test_noise_{test_size}.pt")
    if os.path.exists(test_noise_path):
        test_noise = torch.load(test_noise_path, weights_only=True).to(device)
        print(f"\n\tLOADED NOISE FROM: {test_noise_path}")
    else:
        test_noise = torch.randn(
            (test_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=device)
        torch.save(
            test_noise, 
            test_noise_path
        )
        print(f"\n\tCREATED NOISE IN: {test_noise_path}")


    ### TEACHER MODEL
    teacher_data_path = os.path.join(ROOT, 'DATA', 'cachers', 'eval_data', f"teacher_test_{model_name}_{solver}_{scheduler}_{nfe}_{test_size}.pt")
    imgs_teacher = generate_data(pipe, helper, [], test_dataloader, test_noise, teacher_data_path, nfe, gs, batch_size)


    ### CACHED MODEL
    all_steps_zero = list(range(nfe))
    ts_to_skip = sorted(set(all_steps_zero) - set(not_cached_steps))
    student_data_path = os.path.join(ROOT, 'DATA', 'cachers', 'eval_data', f'student_test_{model_name}_{solver}_{scheduler}_{nfe}_{test_size}_{str(not_cached_steps)}.pt')
    imgs_student = generate_data(pipe, helper, ts_to_skip, test_dataloader, test_noise, student_data_path, nfe, gs, batch_size)

    del pipe, helper, test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    metrics = calc_metrics(
        metric_names=metric_names,
        imgs_gen=imgs_student, 
        imgs_real=imgs_teacher, 
        prompts=prompts,
        device=device,
    )
    print(metrics)

    ### SAVING RESULTS  
    test_run_name = f"{model_name}_{solver}_{scheduler}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    my_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MmRiMzUyNy0xMmIwLTQ1NDUtODQyYS1iNTMxMDk0ZmNkMzEifQ=='
    metric_run = neptune.init_run(
        project="thecrazymage/reinforce-metrics",
        api_token=my_token,
        name=test_run_name,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
    )

    params = {
        "model_name" : model_name,
        "solver" : solver,
        "scheduler" : scheduler, 
        "teacher_nfe" : nfe,
        "student_nfe" : nfe,
        "gs" : gs,
        "test_size" : test_size,
        "batch_size" : batch_size,
        "device" : device,
    }

    params.update({
        "cached_timesteps": ts_to_skip,
        "not_cached_timesteps": not_cached_steps,
    })
    metric_run["parameters"] = params
    metric_run["metrics"] = metrics

    metric_run.stop()


if __name__ == "__main__":
    main()
