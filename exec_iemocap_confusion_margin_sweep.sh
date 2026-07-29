#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
GPU_ID="${GPU_ID:-0}"
RESIDUAL_UPDATE="${RESIDUAL_UPDATE:-standard}"

for pair_weight in 2 5 10
do
  for seed in $(seq 2024 2033)
  do
    echo "mode=sdt_cse_confusion_margin geometry=confusion_separated minimum_gap=75 pair_weight=${pair_weight} classification_margin=0.1 classification_weight=0.1 residual=${RESIDUAL_UPDATE} seed=${seed}"
    python -u "${SCRIPT_DIR}/train.py" \
      --experiment-mode sdt_cse_confusion_margin \
      --selection-protocol validation \
      --circular-geometry confusion_separated \
      --circular-weight 0.1 \
      --minimum-confusion-gap-degrees 75 \
      --confused-cse-pair-weight "${pair_weight}" \
      --confusion-classification-margin 0.1 \
      --confusion-classification-weight 0.1 \
      --sdt-residual-update "${RESIDUAL_UPDATE}" \
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
done

python "${SCRIPT_DIR}/aggregate_results.py" --output-dir "${OUTPUT_DIR}"
