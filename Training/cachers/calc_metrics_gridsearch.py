import os
import sys
import gc
import click
import torch
import random
import numpy as np
import neptune
import pickle
from itertools import combinations
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


def generate_data(pipe, helper, ts_to_skip, test_dataloader, test_noise, nfe, gs, batch_size):

    seed_everything()

    outputs = []
    with torch.inference_mode():
        for batch_idx, anns in enumerate(tqdm(test_dataloader, leave=False)):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + len(anns)
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

    return generated

@click.command()
@click.option('--model_name',     type=str,   required=True,  default="REINFORCE_CACHER",                            help='which pipe to use')
@click.option('--solver',         type=str,   required=True,  default="DDIM",                                        help='supported methods are in lib/solvers')
@click.option('--scheduler',      type=str,   required=True,  default="LINEAR",                                      help='supported methods are in lib/schedulers')
@click.option('--nfe',            type=int,   required=True,  default=13,                                            help='num inference steps')
@click.option('--gs',             type=float, required=True,  default=5,                                             help='guidance scale')
@click.option('--test_size',      type=int,   required=True,  default=1000,                                          help='size of test data')
@click.option('--batch_size',     type=int,   required=True,  default=8,                                             help='size of test data batch')
@click.option('--metric_names',   type=str,   required=True,  default="LPIPS,PLPIPS,L1,AQ,IQ,HPS,CLIP,ImageReward",  help='list of metrics separated by comma')
@click.option('--num_not_cached', type=int,   required=False, default=None,                                          help='number of steps to not cache (including step 0)')
@click.option('--start_idx',      type=int,   required=False, default=0,                                             help='start from this combination index (0-based)')
@click.option("--device",         type=str,                   default="cuda")
def main(**kwargs):

    model_name          = kwargs['model_name']
    solver              = kwargs['solver']
    scheduler           = kwargs['scheduler']
    nfe                 = kwargs['nfe']
    gs                  = kwargs['gs']
    test_size           = kwargs['test_size']
    batch_size          = kwargs['batch_size']
    metric_names        = [m.strip() for m in kwargs['metric_names'].split(',') if m.strip()]
    num_not_cached      = kwargs['num_not_cached']
    device              = kwargs['device']
    start_at            = kwargs['start_idx']
    my_token            ='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MmRiMzUyNy0xMmIwLTQ1NDUtODQyYS1iNTMxMDk0ZmNkMzEifQ=='

    
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
    imgs_teacher = generate_data(pipe, helper, [], test_dataloader, test_noise, nfe, gs, batch_size)


    all_results = []
    k_select = num_not_cached - 1
    step_space = range(1, nfe)
    all_combinations = list(combinations(step_space, k_select))
    print(f"\nTOTAL COMBINATIONS: {len(all_combinations)}")  
    for combo_idx in tqdm(range(start_at, len(all_combinations)), desc="Evaluating combinations"):
        combo = all_combinations[combo_idx]

        ### STUDENT MODEL
        not_cached_steps = [0] + list(combo) 
        print("\n\tNOT CACHED:", not_cached_steps)
        all_steps_zero = list(range(nfe))
        ts_to_skip = sorted(set(all_steps_zero) - set(not_cached_steps))
        imgs_student = generate_data(pipe, helper, ts_to_skip, test_dataloader, test_noise, nfe, gs, batch_size)

        metrics = calc_metrics(
            metric_names=metric_names,
            imgs_gen=imgs_student, 
            imgs_real=imgs_teacher, 
            prompts=prompts,
            device=device,
        )

        ### SAVING RESULTS

        # test_run_name = f"{str(not_cached_steps)}_{combo_idx}"
        # metric_run = neptune.init_run(
        #     project="thecrazymage/reinforce-search",
        #     api_token=my_token,
        #     name=test_run_name,
        #     capture_stdout=False,
        #     capture_stderr=False,
        #     capture_hardware_metrics=False,
        # )

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
            "combination_idx": combo_idx,
            "total_combinations": len(all_combinations), 
            "cached_timesteps": str(ts_to_skip),
            "not_cached_timesteps": str(not_cached_steps),
        }

        # metric_run["parameters"] = params
        # metric_run["metrics"] = metrics

        result = {
            "parameters": params,
            "metrics": metrics
        }
        
        all_results.append(result)

        # metric_run.stop()

        # Save after each iteration (to prevent data loss)
        with open("/home/jovyan/maliev/DIFFUSION_SPEEDUP/DATA/cachers/results.pkl", "wb") as f:
            pickle.dump(all_results, f)

        del imgs_student
        gc.collect()
        torch.cuda.empty_cache()

    del pipe, helper, test_dataloader, imgs_teacher
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
