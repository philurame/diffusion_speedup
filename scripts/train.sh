#!/bin/bash
source /home/user/conda/etc/profile.d/conda.sh
cd /workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP

MLFLOW_DIR="/workspace-SR008.fs2/philurame/MLFLOW"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

ensure_mlflow_ui() {
  local uri="file:${MLFLOW_DIR}"
  local port="$1"
  # if nothing is listening on $port, start mlflow ui
  if ! (ss -ltn 2>/dev/null | grep -q ":${port} "); then
    outp_f="logs/.mlflow_ui_${port}.log"
    > $outp_f
    nohup mlflow ui \
      --backend-store-uri "$uri" \
      --host 127.0.0.1 \
      --port "$port" \
      >> $outp_f 2>&1 &
    echo $! >> $outp_f
    echo "Started MLflow UI on 127.0.0.1:${port}"
  else
    echo "MLflow UI already listening on 127.0.0.1:${port}"
  fi
}
# ===============================================================

# venv="opensora"
# venv="philurame_venv"
venv="/workspace-SR008.fs2/philurame/HunyuanVideo/env"

conda activate $venv
echo "running with venv $venv"

ensure_mlflow_ui "$MLFLOW_PORT"

if [ -z "${1:-}" ]; then
  echo "choose device (integer)"
  exit 1
fi
device_n="$1"

# export CUDA_VISIBLE_DEVICES="$device_n"

outp_f="logs/train_$device_n.txt"
> $outp_f

nohup python3 Training/main.py \
  --device $device_n \
  --config configs/train.yaml \
  >> $outp_f 2>&1 &

echo $! >> $outp_f