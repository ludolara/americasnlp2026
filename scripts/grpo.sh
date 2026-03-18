#!/usr/bin/env bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo_output.log
#SBATCH --error=logs/grpo_error.log
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

GRPO_CONFIG="configs/tiny_aya_grpo.yaml"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NUM_GPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#CUDA_DEVICES[@]}"
fi
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  ./wixarika/bin/python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m train.grpo \
    --config "$GRPO_CONFIG"
else
  ./wixarika/bin/python -m train.grpo --config "$GRPO_CONFIG"
fi

sbatch ./scripts/test_grpo.sh

# ./wixarika/bin/python -m test.eval \
#   --model-name-or-path outputs/tiny-aya-americas-grpo/ \
#   --dataset-path data/americasnlp2026 \
#   --split validation \
#   --target-column target \
#   --batch-size 4096 \
#   --generation-budget 10 \
#   --show-examples

# ./wixarika/bin/python -m test.eval \
#   --model-name-or-path outputs/tiny-aya-americas-grpo/ \
#   --dataset-path data/americasnlp2026 \
#   --split validation \
#   --target-column target \
#   --batch-size 4096 \
#   --generation-budget 100 \
#   --show-examples
