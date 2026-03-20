#!/usr/bin/env bash
#SBATCH --job-name=test_sft
#SBATCH --output=logs/test_sft_output.log
#SBATCH --error=logs/test_sft_error.log
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
source "$ROOT_DIR/wixarika/bin/activate"

"$ROOT_DIR/wixarika/bin/python" -m test.eval \
  --model-name-or-path outputs/tiny-aya-americas \
  --dataset-path data/americasnlp2026 \
  --split validation \
  --batch-size 4096 \
  --generation-budget 10 \
  --show-examples 

"$ROOT_DIR/wixarika/bin/python" -m test.eval \
  --model-name-or-path outputs/tiny-aya-americas \
  --dataset-path data/americasnlp2026 \
  --split validation \
  --batch-size 4096 \
  --generation-budget 100 \
  --show-examples 

# sbatch ./scripts/grpo.sh
