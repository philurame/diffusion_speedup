#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh

cd /home/jovyan/maliev/DIFFUSION_SPEEDUP
conda activate maliev

# # Teacher
# python ./Training/cachers/calc_metrics.py \
#     --not_cached "" \
#     --device cuda:0 \
#     --run_name "TEACHER 25 NFE"

# Deepcache baseline
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 3, 6, 9, 12, 15, 18, 21, 24" \
    --device cuda:0 \
    --run_name "DEEPCACHE"

# ns=2, epoch=[100, 200, 300]
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 4, 5, 8, 14, 18, 23" \
    --device cuda:0 \
    --run_name "ns=2, epoch=[100, 200, 300]"

# ns=4, epoch=100
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 4, 6, 9, 15, 22, 24" \
    --device cuda:0 \
    --run_name "ns=4, epoch=100"

# ns=4, epoch=[200, 300]
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 4, 6, 9, 15, 20, 24" \
    --device cuda:0 \
    --run_name "ns=4, epoch=200"

# ns=6, epoch=[100, 200, 300]
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 4, 5, 9, 14, 22, 24" \
    --device cuda:0 \
    --run_name "ns=6, epoch=[100, 200, 300]"

# ns=8, epoch=[100, 200, 300]
python ./Training/cachers/calc_metrics.py \
    --not_cached "0, 1, 2, 3, 5, 7, 13, 20, 24" \
    --device cuda:0 \
    --run_name "ns=8, epoch=[100, 200, 300]"

