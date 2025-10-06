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


def save_pkl(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_combinations_from_file(filepath):
    """
    Читает комбинации из txt файла
    Ожидаемый формат: каждая строка содержит числа через пробел или запятую
    Например: 1 3 5 7 или 1,3,5,7
    """
    combinations = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                combo = [int(x.strip()) for x in line.replace(',', ' ').split()]
                combinations.append(combo)
    return combinations

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


def generate_data(pipe, helper, ts_to_skip, test_dataloader, test_noise, nfe, gs):
    seed_everything()
    
    images = {}
    with torch.inference_mode():
        for anns in tqdm(test_dataloader, leave=False):
            current_noise = torch.stack([test_noise[p] for p in anns]).to(pipe.device)
            
            with helper.inference(timesteps=ts_to_skip):
                gen = pipe(
                    anns,
                    num_inference_steps=nfe,
                    guidance_scale=gs,
                    latents=current_noise,
                    output_type="pt"
                )
            
            for p, img in zip(anns, gen):
                img = (img + 1) * 0.5  # [-1,1] -> [0,1]
                images[p] = img.cpu()
                
            del gen
            gc.collect()
            torch.cuda.empty_cache()
    
    return images


@click.command()
@click.option('--model_name',     type=str,   required=True,  default="REINFORCE_CACHER",                            help='which pipe to use')
@click.option('--solver',         type=str,   required=True,  default="DDIM",                                        help='supported methods are in lib/solvers')
@click.option('--scheduler',      type=str,   required=True,  default="LINEAR",                                      help='supported methods are in lib/schedulers')
@click.option('--nfe',            type=int,   required=True,  default=13,                                            help='num inference steps')
@click.option('--gs',             type=float, required=True,  default=5,                                             help='guidance scale')
@click.option('--test_size',      type=int,   required=True,  default=1000,                                          help='size of test data')
@click.option('--batch_size',     type=int,   required=True,  default=8,                                             help='size of test data batch')
@click.option('--metric_names',   type=str,   required=True,  default="LPIPS,PLPIPS,L1,AQ,IQ,HPS,CLIP,ImageReward",  help='list of metrics separated by comma')
@click.option('--combs_file',     type=str,   required=True,                                                         help='path to txt file with combinations')
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
    combinations_file   = kwargs['combs_file'] 
    device              = kwargs['device']

    combo_filename = os.path.splitext(os.path.basename(combinations_file))[0]
    res_path = os.path.join(ROOT, "DATA", "cachers", "results", f"results_{combo_filename}.pkl")
    
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
    test_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"test_noise_{test_size}.pkl")
    if os.path.exists(test_noise_path):
        test_noise = load_pkl(test_noise_path)
        print(f"\n\tLOADED NOISE FROM: {test_noise_path}")
    else:
        test_noise = {
            p: torch.randn((4, latent_size, latent_size), dtype=pipe.dtype, device='cpu') 
            for p in prompts
        }
        save_pkl(test_noise, test_noise_path)
        print(f"\n\tCREATED NOISE IN: {test_noise_path}")

    ### TEACHER MODEL
    imgs_teacher = generate_data(pipe, helper, [], test_dataloader, test_noise, nfe, gs)
    imgs_teacher_tensor = torch.stack([imgs_teacher[p] for p in prompts])

    all_combinations = load_combinations_from_file(combinations_file)
    print(f"\nLOADED {len(all_combinations)} COMBINATIONS FROM: {combinations_file}")

    start_at = 0
    if os.path.exists(res_path):
        all_results = load_pkl(res_path)
        print(f"\n\tLOADED EXISTING RESULTS FROM: {res_path}")
        print(f"\tFOUND {len(all_results)} ALREADY COMPUTED RESULTS")
        if len(all_results) > 0:
            start_at = max(r['parameters']['combination_idx'] for r in all_results) + 1
            print(f"\tCONTINUING FROM COMBINATION INDEX: {start_at}")
    else:
        all_results = []
        print(f"\n\tNO EXISTING RESULTS FOUND, STARTING FROM SCRATCH")

    for combo_idx in tqdm(range(start_at, len(all_combinations)), desc="Evaluating combinations"):
        combo = all_combinations[combo_idx]

        ### STUDENT MODEL
        not_cached_steps = list(combo) 
        print("\n\tNOT CACHED:", not_cached_steps)
        all_steps_zero = list(range(nfe))
        ts_to_skip = sorted(set(all_steps_zero) - set(not_cached_steps))
        imgs_student = generate_data(pipe, helper, ts_to_skip, test_dataloader, test_noise, nfe, gs)
        imgs_student_tensor = torch.stack([imgs_student[p] for p in prompts])
        
        metrics = calc_metrics(
            metric_names=metric_names,
            imgs_gen=imgs_student_tensor, 
            imgs_real=imgs_teacher_tensor, 
            prompts=prompts,
            device=device,
        )

        ### SAVING RESULTS
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

        result = {
            "parameters": params,
            "metrics": metrics
        }
        
        all_results.append(result)

        save_pkl(all_results, res_path)

        del imgs_student, imgs_student_tensor
        gc.collect()
        torch.cuda.empty_cache()

    del pipe, helper, test_dataloader, imgs_teacher, imgs_teacher_tensor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
