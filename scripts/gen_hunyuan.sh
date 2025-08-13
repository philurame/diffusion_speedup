#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

cd /workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP
conda activate ./DATA/HUNYUAN_FILES/env

DEVICE=6
echo $DEVICE

outp_f="logs/gen_huyuan_${DEVICE}.txt"
> $outp_f

nohup python3 generate_metric/gen_hunyuan.py $DEVICE >> "$outp_f" 2>&1 &
echo $! >> $outp_f