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
GRPO_CONFIG_PATH="$ROOT_DIR/$GRPO_CONFIG"
TEST_GRPO_OUTPUT_LOG="$ROOT_DIR/logs/test_grpo_output.log"
TEST_GRPO_ERROR_LOG="$ROOT_DIR/logs/test_grpo_error.log"
TEST_GRPO_SUBMITTED_FLAG="$ROOT_DIR/logs/test_grpo_submitted.flag"
GRPO_OUTPUT_DIR="$(GRPO_CONFIG_PATH="$GRPO_CONFIG_PATH" "$ROOT_DIR/wixarika/bin/python" - <<'PY'
import os
from train.config import load_grpo_config
cfg = load_grpo_config(os.environ["GRPO_CONFIG_PATH"])
print(cfg.output_dir)
PY
)"
GRPO_MODEL_SAFETENSORS="$ROOT_DIR/$GRPO_OUTPUT_DIR/model.safetensors"
GRPO_MODEL_BIN="$ROOT_DIR/$GRPO_OUTPUT_DIR/pytorch_model.bin"
GRPO_ADAPTER_MODEL="$ROOT_DIR/$GRPO_OUTPUT_DIR/adapter_model.safetensors"
NEXT_JOB_ID=""

test_logs_exist() {
  [[ -f "$TEST_GRPO_OUTPUT_LOG" && -f "$TEST_GRPO_ERROR_LOG" ]]
}

grpo_model_exists() {
  [[ -f "$GRPO_MODEL_SAFETENSORS" || -f "$GRPO_MODEL_BIN" || -f "$GRPO_ADAPTER_MODEL" ]]
}

submit_test_job() {
  if test_logs_exist; then
    echo "Found $TEST_GRPO_OUTPUT_LOG and $TEST_GRPO_ERROR_LOG; test already completed."
    return 0
  fi

  if [[ -f "$TEST_GRPO_SUBMITTED_FLAG" ]]; then
    local existing_test_job_id
    existing_test_job_id="$(tr -d '[:space:]' < "$TEST_GRPO_SUBMITTED_FLAG")"
    if [[ -n "$existing_test_job_id" ]] && command -v squeue >/dev/null 2>&1; then
      if [[ -n "$(squeue -h -j "$existing_test_job_id" 2>/dev/null)" ]]; then
        echo "test_grpo.sh job $existing_test_job_id is already queued/running."
        return 0
      fi
    fi
    rm -f "$TEST_GRPO_SUBMITTED_FLAG"
  fi

  local -a sbatch_args=()
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sbatch_args+=(--dependency=afterany:${SLURM_JOB_ID})
  fi

  local test_job_id
  test_job_id="$(sbatch --parsable "${sbatch_args[@]}" "$ROOT_DIR/scripts/test_grpo.sh")"
  printf '%s\n' "$test_job_id" > "$TEST_GRPO_SUBMITTED_FLAG"
  echo "Submitted test_grpo job: $test_job_id"
}

schedule_follow_up_grpo() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi

  NEXT_JOB_ID="$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$ROOT_DIR/scripts/grpo.sh")"
  echo "Scheduled follow-up GRPO job: $NEXT_JOB_ID"
}

cancel_follow_up_grpo() {
  if [[ -n "$NEXT_JOB_ID" ]]; then
    scancel "$NEXT_JOB_ID" >/dev/null 2>&1 || true
  fi
}

if test_logs_exist; then
  echo "Found $TEST_GRPO_OUTPUT_LOG and $TEST_GRPO_ERROR_LOG; GRPO/test pipeline is complete."
  exit 0
fi

if grpo_model_exists; then
  echo "Found completed GRPO model in $ROOT_DIR/$GRPO_OUTPUT_DIR; submitting test job."
  submit_test_job
  exit 0
fi

schedule_follow_up_grpo

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

if grpo_model_exists; then
  echo "GRPO training completed; submitting test job."
  cancel_follow_up_grpo
  submit_test_job
fi
