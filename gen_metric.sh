#!/bin/bash

root_dir=DIFFUSION_SPEEDUP

device=0
nfe=10
model=SDXL_BASE # possible values: SDXL_BASE/DEEPCACHE2/DEEPCACHE3/DEEPCACHE4/HQQ
solver=DPMS      # possible values: DDIM/DDIM/DPMS/DEIS
scheduler=AYS # possible values: LINEAR/AYS/KARRAS/SNR/STUDENTDDIM200/STUDENTDEIS40
metric_names=SEC_PER_IMG,LPIPS,FID,IS,CLIP # possible values: TFLOPS,SEC_PER_IMG,LPIPS,FID,IS,CLIP

generate_batch_size=64 
metric_batch_size=16
max_samples=10000 


if [ ! -f "$root_dir/DATA/datasets_coco_parti.pkl" ]; then
  FILE_ID=1gOvJpJXUb-27aH24aNjzgCSrX37s83jc
  OUTPUT_FILE=$root_dir/DATA/datasets_coco_parti.pkl
  gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_FILE"
fi

if [ ! -f "$root_dir/DATA/imgs_ddim200_224.pt" ]; then
  FILE_ID=1qdMfxX2h7MndBlgiuU5m7Woq8BfeSJ67
  OUTPUT_FILE=$root_dir/DATA/imgs_ddim200_224.pt
  gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_FILE"
fi


python $root_dir/lib/main_generate.py \
  --dataset COCO \
  --model_name $model \
  --solver $solver \
  --scheduler $scheduler \
  --nfe $nfe \
  --device $device \
  --max_samples $max_samples \
  --batch_size $generate_batch_size

python $root_dir/lib/main_metric.py \
  --dataset COCO \
  --model_name $model \
  --solver $solver \
  --scheduler $scheduler \
  --nfe $nfe \
  --metric_names $metric_names \
  --device $device \
  --max_samples $max_samples \
  --batch_size $metric_batch_size