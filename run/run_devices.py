import os, sys
import subprocess
import yaml
import argparse
import queue
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_task(task, ROOT, str_generate_metric, wandb_key, metrics, device_queue, max_samples):
  if len(task) == 5:
    (dataset, nfe, solver, scheduler, model_name) = task
  else:
    (dataset, solver_scheduler_nfe,  model_name) = task
    solver, scheduler, nfe = solver_scheduler_nfe.split('_')

  device = device_queue.get()
  try:
    outdir = os.path.join(os.path.dirname(ROOT), f'_runs/{str_generate_metric}_{solver}')
    os.makedirs(outdir, exist_ok=True)
    method = f'{solver}_{scheduler}_{model_name}_{nfe}'
    main_script = os.path.join(ROOT, 'lib', f'main_{str_generate_metric}.py')
    cmd = (
      f"nohup python3 {main_script} "
      f"--dataset {dataset} "
      f"--model_name {model_name} "
      f"--solver {solver} "
      f"--scheduler {scheduler} "
      f"--nfe {nfe} "
      f"--key {wandb_key} "
      f"--metric_names {metrics} "
      f"--max_samples {max_samples} "
      f"--device {device} "
      f"> {os.path.join(outdir, method + '.log')} 2>&1"
    )
    print(f"Starting job: {method} on device {device}")
    sys.stdout.flush()
    subprocess.run(cmd, shell=True)
    print(f"Finished job: {method} on device {device}")
    sys.stdout.flush()
  finally:
    device_queue.put(device)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--wandb_key", type=str, required=True, help="WandB API key")
  parser.add_argument("--generate", type=int, required=True, help="Flag to indicate generate (1) or metric (0) mode")
  parser.add_argument("--max_samples", type=int, default=30_000)
  parser.add_argument("--devices", type=int, nargs='+', default=list(range(8)),
    help="List of accessible device indices (e.g., 0 1 2 ...). Defaults to 0-7.")
  args = parser.parse_args()

  wandb_key = args.wandb_key
  is_generate = args.generate
  max_samples = args.max_samples
  device_list = args.devices
  str_generate_metric = "generate" if is_generate else "metric"

  # Fetch configuration from the appropriate YAML file.
  ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  config_path = os.path.join(ROOT, 'run', f'config_{str_generate_metric}.yaml')
  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    nfes = config['nfes']
    solvers = config['solvers']
    schedulers = config['schedulers']
    solvers_schedulers_nfes = config['solvers_schedulers_nfes']
    models = config['model_names']
    datasets = config['datasets']
    metrics = ','.join(config.get('metrics', ['_']))

  if solvers_schedulers_nfes:
    all_tasks = list(product(datasets, solvers_schedulers_nfes, models))
  else:
    all_tasks = list(product(datasets, nfes, solvers, schedulers, models))

  device_queue = queue.Queue()
  for d in device_list:
    device_queue.put(d)

  max_workers = len(device_list)
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for task in all_tasks:
      futures.append(executor.submit(run_task, task, ROOT, str_generate_metric, wandb_key, metrics, device_queue, max_samples))
    for future in as_completed(futures):
      try:
        future.result()
      except Exception as e:
        print(f"Error occurred during job execution: {e}")