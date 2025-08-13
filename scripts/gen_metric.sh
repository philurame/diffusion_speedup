#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

export WANDB_CONFIG_DIR=/workspace-SR008.fs2/philurame/wandb/config
export WANDB_CACHE_DIR=/workspace-SR008.fs2/philurame/wandb/cache

cd /workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP
conda activate philurame_venv

wandb_key=_

outp_f="logs/generate_metrics.txt"
> $outp_f

nohup python3 generate_metric/rundev_generate_metric.py.py \
  --wandb_key $wandb_key \
  --save_gen 0 \
  --devices 6 \
  --batch_size 16 \
  --num_samples 10000 >> $outp_f 2>&1 &
echo $! >> $outp_f

# ===============================================================
#  generate_mode param:
#  0: "generate"
#  1: "metric"
#  2: "generate+metric" (images not stored)
# ===============================================================