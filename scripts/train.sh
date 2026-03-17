#!/usr/bin/env bash
#SBATCH --job-name=sft_base
#SBATCH --output=logs/sft_base_output.log
#SBATCH --error=logs/sft_base_error.log
#SBATCH --gres=gpu:a100l
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=main

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
source "$ROOT_DIR/wixarika/bin/activate"

./wixarika/bin/python -m train.sft --config configs/tiny_aya_full_sft.yaml

./wixarika/bin/python -m test.eval \
  --model-name-or-path outputs/tiny-aya-americas-base/ \
  --dataset-path data/americasnlp2026 \
  --split validation \
  --batch-size 512 \
  --generation-budget 10 \
  --show-examples 

./wixarika/bin/python -m test.eval \
  --model-name-or-path outputs/tiny-aya-americas-base/ \
  --dataset-path data/americasnlp2026 \
  --split validation \
  --batch-size 512 \
  --generation-budget 100 \
  --show-examples 
