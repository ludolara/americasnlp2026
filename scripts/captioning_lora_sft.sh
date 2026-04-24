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

MODEL_PATH="${MODEL_PATH:-outputs/aya-vision-32b-americas}"
DATASET_PATH="${DATASET_PATH:-data/captioning}"
TRAIN_SPLIT="${TRAIN_SPLIT:-validation}"
LANGUAGES="${LANGUAGES:-hch,bzd,grn}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/aya-vision-32b-americas-captioning}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LANGUAGE_SAMPLING_ALPHA="${LANGUAGE_SAMPLING_ALPHA:-1.0}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_lora_zero3.json}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "$ROOT_DIR/wixarika/bin/python" -m train.captioning_lora_sft
  --model-name-or-path "$MODEL_PATH"
  --dataset-path "$DATASET_PATH"
  --train-split "$TRAIN_SPLIT"
  --languages "$LANGUAGES"
  --output-dir "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --warmup-ratio "$WARMUP_RATIO"
  --weight-decay "$WEIGHT_DECAY"
  --logging-steps "$LOGGING_STEPS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --language-sampling-alpha "$LANGUAGE_SAMPLING_ALPHA"
)

if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  cmd+=(--deepspeed "$DEEPSPEED_CONFIG")
else
  cmd+=(--deepspeed "")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

cmd+=("$@")

NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NUM_GPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#CUDA_DEVICES[@]}"
fi
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "$NUM_GPUS" -gt 1 && "$DRY_RUN" != "1" ]]; then
  "$ROOT_DIR/wixarika/bin/accelerate" launch \
    --multi_gpu \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    "${cmd[@]:1}"
else
  "${cmd[@]}"
fi
