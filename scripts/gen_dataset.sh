#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

cd /workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP
conda activate ./DATA/HUNYUAN_FILES/env

declare -A params=(
  [device]=7
  [dataset]="HUNYUAN"
  [model_name]="HUNYUAN_BASE"
  [solver]="FLOW2-LOG"
  [scheduler]="FLOW"
  [nfe]=100
  [num_samples]=10
  [batch_size]=1
  [seed_shift]=1100
  [is_gen_img]=1
)
params[path_save]="DATA/TRAIN_DATA/val_${params[model_name]}_${params[solver]}_${params[scheduler]}_${params[nfe]}.pkl"

outp_f="logs/generate_${params[device]}.txt"
> $outp_f

args=""
for key in "${!params[@]}"; do
  args="$args --$key ${params[$key]}"
done

nohup python3 generate_metric/generate.py $args >> "$outp_f" 2>&1 &
echo $! >> "$outp_f"