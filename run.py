import os, subprocess, yaml, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_key", type=str)
parser.add_argument("--generate", type=int)
wandb_key   = parser.parse_args().wandb_key
is_generate = parser.parse_args().generate
str_generate_metric = "generate" if is_generate else "metric"

# fetch config
ROOT = os.path.dirname(os.path.abspath(__file__)) # project root
with open(os.path.join(ROOT, f'config_{str_generate_metric}.yaml'), 'r') as file:
  config = yaml.safe_load(file)
  solvers    = config['solvers']
  schedulers = config['schedulers']
  models     = config['model_names']
  nfes       = config['nfes']
  datasets   = config['datasets']
  metrics    = ','.join(config.get('metrics', ['_']))

# run in parallel over all combinations
from itertools import product
for dataset, nfe, solver, scheduler, model_name in product(datasets, nfes, solvers, schedulers, models):

  outdir = os.path.join(os.path.dirname(ROOT), f'_runs/{str_generate_metric}_{solver}')
  os.makedirs(outdir, exist_ok=True)

  method =  f'{solver}_{scheduler}_{model_name}_{nfe}'
  approx_time = nfe if is_generate else 4

  # Build the script content
  script_content = f"""\
#!/bin/bash --login
#SBATCH --job-name={method}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time={approx_time}:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH --output={outdir}/{method}-%j.log

module load Python/Anaconda_v03.2023

conda deactivate
conda activate /home/ekneudachina/.conda/envs/philurame_venv

python3 {os.path.join(ROOT, f'main_{str_generate_metric}.py')} \\
--dataset {dataset} \\
--model_name {model_name} \\
--solver {solver} \\
--scheduler {scheduler} \\
--nfe {nfe} \\
--key {wandb_key} \\
--metric_names {metrics}
"""

  # Submit the script content via sbatch
  command = ['sbatch']
  subprocess.run(command, input=script_content, text=True)