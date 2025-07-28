#!/bin/bash

device=0
nfe=3
model=SD15_BASE # possible values: SD15_BASE/SDXL_BASE/SORA_BASE/DEEPCACHE3/...
solver=DDIM      # possible values: DDIM/DPMS/DEIS/...
scheduler=LINEAR # possible values: LINEAR/LEADING/AYS/KARRAS/SNR/...
metric_names=CLIP,ImageReward # possible values: CLIP,ImageReward,AQ,IQ,HPS,FID,FID-CLIP,FID-DINO,FID

batch_size=16 
max_samples=10000 
wandb_key=

# load coco imgs if not already and FID-like metric is chosen
if [[ $metric_names == *"FID"* ]] && [ ! -f "DATA/coco_imgs299_10k.pt" ]; then
  FILE_ID=1N8v3LdTeKI5WtJVjxDqo1_N7C2feC4ZH
  OUTPUT_FILE=DATA/coco_imgs299_10k.pt
  gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_FILE"
fi

python generate_metric/generate_metric.py \
  --metric_names $metric_names \
  --dataset COCO-SHORT \
  --model_name $model \
  --solver $solver \
  --scheduler $scheduler \
  --nfe $nfe \
  --device $device \
  --max_samples $max_samples \
  --batch_size $batch_size \
  --key $wandb_key