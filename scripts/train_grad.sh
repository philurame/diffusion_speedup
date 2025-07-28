#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

export WANDB_CONFIG_DIR=/workspace-SR008.fs2/philurame/wandb/config
export WANDB_CACHE_DIR=/workspace-SR008.fs2/philurame/wandb/cache

cd /workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP

# ===============================================================

# venv="opensora"
venv="philurame_venv"
conda activate $venv
echo "running with $venv venv"

device_n='choose device (integer)'
if [ -n "$1" ]; then
  device_n="$1"
fi

outp_f="logs/train_gradient_$device_n.txt"
> $outp_f

nohup python3 Training/Grad_trainer/main.py \
  --device $device_n \
  --config configs/train_grad.yaml \
  >> $outp_f 2>&1 &

echo $! >> $outp_f