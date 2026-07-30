#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
GPU_ID="${GPU_ID:-0}"

for seed in $(seq 2024 2033)
do
  echo "mode=sdt_cse_bilevel_confusion_gap_train_holdout seed=${seed}"
  python -u "${SCRIPT_DIR}/train.py" \
    --experiment-mode sdt_cse_bilevel_confusion_gap_train_holdout \
    --selection-protocol validation \
    --validation-ratio 0.1 \
    --angle-holdout-ratio 0.1 \
    --device cuda \
    --gpu-id "${GPU_ID}" \
    --seed "${seed}" \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0001 \
    --weight-decay 0.00001 \
    --temperature 1.0 \
    --fusion-ce-weight 1.0 \
    --unimodal-ce-weight 1.0 \
    --distillation-weight 1.0 \
    --circular-weight 0.1 \
    --confused-cse-pair-weight 5 \
    --confusion-classification-margin 0.1 \
    --confusion-classification-weight 0.1 \
    --bilevel-gap-minimum-degrees 70 \
    --bilevel-gap-maximum-degrees 110 \
    --bilevel-gap-initial-degrees 90 \
    --bilevel-angle-learning-rate 0.001 \
    --bilevel-inner-step-size 0.0001 \
    --bilevel-hvp-radius 0.01 \
    --bilevel-outer-confusion-weight 0.1 \
    --output-dir "${OUTPUT_DIR}"
done

python "${SCRIPT_DIR}/aggregate_results.py" --output-dir "${OUTPUT_DIR}"
