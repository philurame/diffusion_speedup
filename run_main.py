import os, subprocess, yaml

# project root
ROOT = os.path.dirname(os.path.abspath(__file__))

# wandb key
with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
  wandb_key = f.read().strip()

# fetch config
with open(os.path.join(ROOT, 'config.yaml'), 'r') as file:
  config = yaml.safe_load(file)
  generate   = config['generate']
  solvers    = config['solvers']
  schedulers = config['schedulers']
  cachers    = config['cachers_quantizers']
  nfes       = config['nfes']
  datasets   = config['datasets']

# generates <=10k images
max_samples = 10000

from itertools import product
for dataset, nfe, solver, scheduler, cacher in product(datasets, nfes, solvers, schedulers, cachers):

  outdir = os.path.join(os.path.dirname(ROOT), f'_runs/{"generate" if generate else "metrics"}_{solver}')
  os.makedirs(outdir, exist_ok=True)

  method =  f'{solver}_{scheduler}_{cacher}_{nfe}'

  # Build the script content
  script_content = f"""\
#!/bin/bash --login
#SBATCH --job-name={method}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --constraint="[type_a|type_b|type_c|type_e]"
#SBATCH --output={outdir}/{method}-%j.log

module load Python/Anaconda_v03.2023

conda deactivate
conda activate philurame_venv

python3 {os.path.join(ROOT, 'main.py')} \\
--key {wandb_key} \\
--max_samples {max_samples} \\
--generate {generate} \\
--dataset {dataset} \\
--solver {solver} \\
--scheduler {scheduler} \\
--cacher_quantizer {cacher} \\
--nfe {nfe}
"""

  # Submit the script content via sbatch
  command = ['sbatch']
  subprocess.run(command, input=script_content, text=True)