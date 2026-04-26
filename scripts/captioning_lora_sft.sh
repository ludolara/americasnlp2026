#!/usr/bin/env bash
#SBATCH --job-name=captioning_lora_sft
#SBATCH --output=logs/captioning_lora_sft_output.log
#SBATCH --error=logs/captioning_lora_sft_error.log
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$ROOT_DIR/models/.hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
source "$ROOT_DIR/wixarika/bin/activate"

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_HOME
  export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:${PATH}"
  if [[ -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

CAPTIONING_LORA_SFT_CONFIG="${CAPTIONING_LORA_SFT_CONFIG:-configs/captioning_lora_sft.yaml}"
for ((arg_index = 1; arg_index <= $#; arg_index++)); do
  arg="${!arg_index}"
  case "$arg" in
    --config)
      next_index=$((arg_index + 1))
      if [[ "$next_index" -le "$#" ]]; then
        CAPTIONING_LORA_SFT_CONFIG="${!next_index}"
      fi
      ;;
    --config=*) CAPTIONING_LORA_SFT_CONFIG="${arg#--config=}" ;;
  esac
done

CONFIG_DRY_RUN="$(
  CAPTIONING_LORA_SFT_CONFIG="$CAPTIONING_LORA_SFT_CONFIG" "$ROOT_DIR/wixarika/bin/python" - <<'PY'
import os
import yaml

with open(os.environ["CAPTIONING_LORA_SFT_CONFIG"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print("1" if cfg.get("dry_run") else "0")
PY
)"

normalize_bool() {
  case "${1,,}" in
    1|true|yes|on) echo "1" ;;
    0|false|no|off|"") echo "0" ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

LAUNCH_DRY_RUN="$(normalize_bool "${DRY_RUN:-$CONFIG_DRY_RUN}")"

cmd=(
  "$ROOT_DIR/wixarika/bin/python" -m train.captioning_lora_sft
  --config "$CAPTIONING_LORA_SFT_CONFIG"
)

if [[ -n "${DRY_RUN:-}" && "$LAUNCH_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
elif [[ -n "${DRY_RUN:-}" ]]; then
  cmd+=(--no-dry-run)
fi

for arg in "$@"; do
  case "$arg" in
    --dry-run) LAUNCH_DRY_RUN="1" ;;
    --no-dry-run) LAUNCH_DRY_RUN="0" ;;
  esac
done

cmd+=("$@")

NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NUM_GPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#CUDA_DEVICES[@]}"
fi
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "$NUM_GPUS" -gt 1 && "$LAUNCH_DRY_RUN" != "1" ]]; then
  "$ROOT_DIR/wixarika/bin/accelerate" launch \
    --multi_gpu \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    "${cmd[@]:1}"
else
  "${cmd[@]}"
fi
