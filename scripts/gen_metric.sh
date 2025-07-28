#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

export WANDB_CONFIG_DIR=/workspace-SR008.fs2/philurame/wandb/config
export WANDB_CACHE_DIR=/workspace-SR008.fs2/philurame/wandb/cache

cd /workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP
conda activate philurame_venv

wandb_key=ac57f7f673f3ffda25cbe4634e094c2b90edbd8f

outp_f="logs/generate_metrics.txt"
> $outp_f

nohup python3 generate_metric/run_devices.py \
  --wandb_key $wandb_key \
  --devices 3  \
  --batch_size 16 \
  --max_samples 100 >> $outp_f 2>&1 &
echo $! >> $outp_f

# ===============================================================
#  generate_mode param:
#  0: "generate"
#  1: "metric"
#  2: "generate+metric" (images not stored)
# ===============================================================