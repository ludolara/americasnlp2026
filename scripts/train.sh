#!/usr/bin/env bash
#SBATCH --job-name=global
#SBATCH --output=logs/global_output.log
#SBATCH --error=logs/global_error.log
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=unkillable

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
source "$ROOT_DIR/wixarika/bin/activate"

./wixarika/bin/python -m train.train --config configs/tiny_aya_full_sft.yaml
