#!/usr/bin/env bash
#SBATCH --job-name=merge_lora_vllm
#SBATCH --output=logs/merge_lora_vllm_output.log
#SBATCH --error=logs/merge_lora_vllm_error.log
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=main

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$ROOT_DIR/models/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$ROOT_DIR/wixarika/bin/activate"

ADAPTER_PATH="${ADAPTER_PATH:-outputs/aya-vision-32b-americas}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/aya-vision-32b-americas-merged}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
LOAD_DTYPE="${LOAD_DTYPE:-base}"
SAVE_DTYPE="${SAVE_DTYPE:-base}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-10GB}"
SAFE_MERGE="${SAFE_MERGE:-0}"
OVERWRITE="${OVERWRITE:-0}"

cmd=(
  "$ROOT_DIR/wixarika/bin/python" -m test.merge_lora_adapter_vllm
  --adapter-path "$ADAPTER_PATH"
  --output-dir "$OUTPUT_DIR"
  --load-dtype "$LOAD_DTYPE"
  --save-dtype "$SAVE_DTYPE"
  --device-map "$DEVICE_MAP"
  --max-shard-size "$MAX_SHARD_SIZE"
)

if [[ -n "$BASE_MODEL_PATH" ]]; then
  cmd+=(--base-model-name-or-path "$BASE_MODEL_PATH")
fi

if [[ "$SAFE_MERGE" == "1" ]]; then
  cmd+=(--safe-merge)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
