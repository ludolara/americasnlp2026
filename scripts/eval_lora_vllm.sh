#!/usr/bin/env bash
#SBATCH --job-name=eval_lora_vllm
#SBATCH --output=logs/eval_lora_vllm_output.log
#SBATCH --error=logs/eval_lora_vllm_error.log
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$ROOT_DIR/models/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
source "$ROOT_DIR/wixarika/bin/activate"

MODEL_PATH="${MODEL_PATH:-outputs/aya-vision-32b-americas-merged}"
DATASET_PATH="${DATASET_PATH:-data/americasnlp2026}"
SPLIT="${SPLIT:-test}"
LANGUAGES="${LANGUAGES:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.3}"
TOP_K="${TOP_K:-1}"
TOP_P="${TOP_P:-}"
DTYPE="${DTYPE:-float16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-1}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"
GENERATION_BUDGET="${GENERATION_BUDGET:-100}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

if [[ "$#" -gt 0 ]]; then
  SCORE_BUDGETS=("$@")
  GENERATION_BUDGET="${SCORE_BUDGETS[0]}"
  for score_budget in "${SCORE_BUDGETS[@]}"; do
    if (( score_budget > GENERATION_BUDGET )); then
      GENERATION_BUDGET="$score_budget"
    fi
  done
else
  if (( GENERATION_BUDGET > 10 )); then
    SCORE_BUDGETS=(10 "$GENERATION_BUDGET")
  else
    SCORE_BUDGETS=("$GENERATION_BUDGET")
  fi
fi

LANGUAGE_ARGS=()
if [[ -n "$LANGUAGES" ]]; then
  read -r -a LANGUAGE_ARGS <<< "$LANGUAGES"
fi

cmd=(
  "$ROOT_DIR/wixarika/bin/python" -m test.eval_lora_vllm
  --model-name-or-path "$MODEL_PATH"
  --dataset-path "$DATASET_PATH"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --generation-budget "$GENERATION_BUDGET"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --seed "$SEED"
  --results-dir "$RESULTS_DIR"
  --num-examples "$NUM_EXAMPLES"
)

if [[ "${#LANGUAGE_ARGS[@]}" -gt 0 ]]; then
  cmd+=(--languages "${LANGUAGE_ARGS[@]}")
fi

if [[ "${#SCORE_BUDGETS[@]}" -gt 0 ]]; then
  cmd+=(--score-budgets "${SCORE_BUDGETS[@]}")
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ -n "$TOP_P" ]]; then
  cmd+=(--top-p "$TOP_P")
fi

if [[ "$SHOW_EXAMPLES" == "1" ]]; then
  cmd+=(--show-examples)
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  cmd+=(--disable-custom-all-reduce)
fi

if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
  cmd+=(--language-model-only)
else
  cmd+=(--no-language-model-only)
fi

"${cmd[@]}"
