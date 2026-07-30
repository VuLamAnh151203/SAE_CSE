#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
GPU_ID="${GPU_ID:-0}"

for seed in $(seq 2024 2033)
do
  echo "mode=sdt_cse_bilevel_all_gaps init=equal minimum_gap=20 seed=${seed}"
  python -u "${SCRIPT_DIR}/train.py" \
    --experiment-mode sdt_cse_bilevel_all_gaps \
    --selection-protocol validation \
    --circular-weight 0.1 \
    --confused-cse-pair-weight 5 \
    --confusion-classification-margin 0.1 \
    --confusion-classification-weight 0.1 \
    --bilevel-all-gaps-initialization equal \
    --bilevel-minimum-class-gap-degrees 20 \
    --bilevel-gap-prior-weight 0.01 \
    --bilevel-angle-learning-rate 0.001 \
    --bilevel-inner-step-size 0.0001 \
    --bilevel-hvp-radius 0.01 \
    --bilevel-outer-confusion-weight 0.1 \
    --bilevel-angle-gradient-clip 5 \
    --sdt-residual-update standard \
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
