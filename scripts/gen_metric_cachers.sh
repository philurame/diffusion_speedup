#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

cd /home/jovyan/maliev/DIFFUSION_SPEEDUP
conda activate maliev

# ns=2, epoch=100
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 5, 8, 14, 18, 22, 24" \
    --device cuda:0 \
    --run_name "fixed ns=2, epoch=100"

# ns=2, epoch=[200, 300]
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 3, 5, 8, 14, 22, 24" \
    --device cuda:0 \
    --run_name "fixed ns=2, epoch=[200, 300]"

# ns=4, epoch=100
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 4, 8, 15, 18, 20, 24" \
    --device cuda:0 \
    --run_name "fixed ns=4, epoch=100"

# ns=4, epoch=[200, 300]
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 4, 5, 8, 15, 20, 24" \
    --device cuda:0 \
    --run_name "fixed ns=4, epoch=[200, 300]"