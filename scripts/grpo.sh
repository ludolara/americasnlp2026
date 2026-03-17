#!/usr/bin/env bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo_output.log
#SBATCH --error=logs/grpo_error.log
#SBATCH --gres=gpu:a100l
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
EVAL_ARGS=("$@")

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
source "$ROOT_DIR/wixarika/bin/activate"

./wixarika/bin/python -m train.grpo --config configs/tiny_aya_grpo.yaml

./wixarika/bin/python -m test.eval \
  --model-name-or-path outputs/tiny-aya-americas-grpo/ \
  --dataset-path data/americasnlp2026 \
  --split validation \
  --target-column target \
  --batch-size 512 \
  --generation-budget 10 \
  --show-examples \
  "${EVAL_ARGS[@]}"

./wixarika/bin/python -m test.eval \
  --model-name-or-path outputs/tiny-aya-americas-grpo/ \
  --dataset-path data/americasnlp2026 \
  --split validation \
  --target-column target \
  --batch-size 512 \
  --generation-budget 100 \
  --show-examples \
  "${EVAL_ARGS[@]}"
