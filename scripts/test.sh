#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=logs/test_output.log
#SBATCH --error=logs/test_error.log
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=main

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
source "$ROOT_DIR/wixarika/bin/activate"

# --model-name-or-path outputs/tiny-aya-global-wixarika/checkpoint-450 \
# --model-name-or-path outputs/tiny-aya-base-wixarika/checkpoint-510 \
# --model-name-or-path outputs/tiny-aya-global-wixarika-grpo/checkpoint-750 \

"$ROOT_DIR/wixarika/bin/python" -m test.eval \
  --model-name-or-path outputs/tiny-aya-wixarika\
  --dataset-path data/wixarika_spanish_hf \
  --split test \
  --batch-size 512 \
  --generation-budget 10 \
  --show-examples 

"$ROOT_DIR/wixarika/bin/python" -m test.eval \
  --model-name-or-path outputs/tiny-aya-wixarika \
  --dataset-path data/wixarika_spanish_hf \
  --split test \
  --batch-size 512 \
  --generation-budget 100 \
  --show-examples 
