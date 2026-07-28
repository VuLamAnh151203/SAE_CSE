#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
GPU_ID="${GPU_ID:-0}"

for mode in sdt sdt_cosine sdt_cse sdt_cse_all_cosine
do
  for seed in $(seq 2024 2033)
  do
    echo "mode=${mode} seed=${seed}"
    python -u "${SCRIPT_DIR}/train.py" \
      --experiment-mode "${mode}" \
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
      --output-dir "${OUTPUT_DIR}"
  done
done

python "${SCRIPT_DIR}/aggregate_results.py" --output-dir "${OUTPUT_DIR}"
