#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
source "$ROOT_DIR/wixarika/bin/activate"

# --model-name-or-path outputs/tiny-aya-global-wixarika/checkpoint-450 \
# --model-name-or-path outputs/tiny-aya-base-wixarika/checkpoint-510 \
# --model-name-or-path outputs/tiny-aya-global-wixarika-grpo/checkpoint-750 \

"$ROOT_DIR/wixarika/bin/python" -m test.chrf_eval \
  --model-name-or-path outputs/tiny-aya-wixarika \
  --dataset-path data/wixarika_spanish_hf \
  --split validation \
  --batch-size 512 \
  --generation-budget 10 \
  --show-examples 
