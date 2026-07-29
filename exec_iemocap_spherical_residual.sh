#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
GPU_ID="${GPU_ID:-0}"

for seed in $(seq 2024 2033)
do
  echo "mode=sdt_cse_learnable_angles residual=spherical selection=validation seed=${seed}"
  python -u "${SCRIPT_DIR}/train.py" \
    --experiment-mode sdt_cse_learnable_angles \
    --selection-protocol validation \
    --circular-geometry equal \
    --circular-weight 0.1 \
    --angle-weight 0.1 \
    --sdt-residual-update spherical \
    --spherical-attention-alpha-init 0.1 \
    --spherical-mlp-alpha-init 0.1 \
    --device cuda \
    --gpu-id "${GPU_ID}" \
    --seed "${seed}" \
    --epochs 150 \
    --batch-size 16 \
    --hidden-dim 1024 \
    --n-head 8 \
    --dropout 0.5 \
    --embedding-dim 256 \
    --projection-dropout 0.1 \
    --initial-cosine-scale 16 \
    --lr 0.0001 \
    --weight-decay 0.00001 \
    --temperature 1.0 \
    --fusion-ce-weight 1.0 \
    --unimodal-ce-weight 1.0 \
    --distillation-weight 1.0 \
    --output-dir "${OUTPUT_DIR}"
done

python "${SCRIPT_DIR}/aggregate_results.py" --output-dir "${OUTPUT_DIR}"
