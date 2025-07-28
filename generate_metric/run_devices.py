import os, sys
import subprocess
import yaml
import argparse
import queue
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_task(task, ROOT, wandb_key, metrics, device_queue, max_samples, batch_size):
  if len(task) == 5:
    (dataset, nfe, solver, scheduler, model_name) = task
  else:
    (dataset, solver_scheduler_nfe,  model_name) = task
    solver, scheduler, nfe = solver_scheduler_nfe.split('_')

  device = device_queue.get()
  try:
    outdir = os.path.join(ROOT, f'logs/metrics_{solver}_{dataset}')
    os.makedirs(outdir, exist_ok=True)
    method = f'{solver}_{scheduler}_{model_name}_{nfe}'
    main_script = os.path.join(ROOT, 'generate_metric', f'generate_metric.py')
    cmd = (
      f"nohup python3 {main_script} "
      f"--dataset {dataset} "
      f"--model_name {model_name} "
      f"--solver {solver} "
      f"--scheduler {scheduler} "
      f"--nfe {nfe} "
      f"--key {wandb_key} "
      f"--metric_names {metrics} "
      f"--batch_size {batch_size} "
      f"--device {device} "
      f"--max_samples {max_samples} "
      f"> {os.path.join(outdir, method + '.log')} 2>&1 "
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
  parser.add_argument("--wandb_key", type=str, default='', help="WandB API key")
  parser.add_argument("--max_samples", type=int, default=30_000)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--devices", type=int, nargs='+', default=list(range(8)), help="List of accessible device indices (e.g., 0 1 2 ...). Defaults to 0-7.")
  args = parser.parse_args()

  wandb_key   = args.wandb_key
  max_samples = args.max_samples
  device_list = args.devices
  batch_size  = args.batch_size

  # Fetch configuration from the appropriate YAML file.
  ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  config_path = os.path.join(ROOT, 'configs', f'generate_metric.yaml')
  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    nfes = config.get('nfes', [])
    solvers = config.get('solvers', [])
    schedulers = config.get('schedulers', [])
    solvers_schedulers_nfes = config.get('solvers_schedulers_nfes', [])
    models = config['model_names']
    datasets = config['datasets']
    metrics = ','.join(config.get('metrics', ['_']))

    if not all([nfes, solvers, schedulers]):
      nfes = solvers = schedulers = []
  
  grid_tasks = list(product(datasets, solvers_schedulers_nfes, models))
  seq_tasks  = list(product(datasets, nfes, solvers, schedulers, models))

  all_tasks = grid_tasks + seq_tasks

  device_queue = queue.Queue()
  for d in device_list:
    device_queue.put(d)

  max_workers = len(device_list)
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for task in all_tasks:
      futures.append(executor.submit(run_task, task, ROOT, wandb_key, metrics, device_queue, max_samples, batch_size))
    for future in as_completed(futures):
      try:
        future.result()
      except Exception as e:
        print(f"Error occurred during job execution: {e}")