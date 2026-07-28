#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"

for circular_weight in 0.01 0.05 0.1 0.5 1.0
do
  for seed in $(seq 2024 2033)
  do
    echo "lambda=${circular_weight} seed=${seed}"
    python -u "${SCRIPT_DIR}/train.py" \
      --experiment-mode sdt_cse \
      --circular-weight "${circular_weight}" \
      --seed "${seed}" \
      --epochs 150 \
      --batch-size 16 \
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
