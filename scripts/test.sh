#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
source wixarika/bin/activate

# --model-name-or-path outputs/tiny-aya-global-wixarika/checkpoint-450 \
# --model-name-or-path outputs/tiny-aya-base-wixarika/checkpoint-510 \

./wixarika/bin/python -m test.chrf_eval \
  --model-name-or-path outputs/tiny-aya-base-wixarika/checkpoint-510 \
  --dataset-path data/wixarika_spanish_hf \
  --split validation \
  --batch-size 512 \
  --show-examples 
