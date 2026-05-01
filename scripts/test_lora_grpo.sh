#!/usr/bin/env bash
#SBATCH --job-name=test_lora_grpo
#SBATCH --output=logs/test_lora_grpo_output.log
#SBATCH --error=logs/test_lora_grpo_error.log
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=main

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$ROOT_DIR/models/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$ROOT_DIR/wixarika/bin/activate"

MODEL_PATH="${MODEL_PATH:-outputs/aya-vision-32b-americas-grpo/checkpoint-125}"
DATASET_PATH="${DATASET_PATH:-data/americasnlp2026}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-1}"
LIMIT="${LIMIT:-}"
EVAL_LANGUAGE="${EVAL_LANGUAGE:-}"
# eval_lora filters by language_code: hch, bzd, gn, nah.
EVAL_LANGUAGES_RAW="${EVAL_LANGUAGES:-hch,bzd,gn,nah}"
EVAL_LANGUAGES_RAW="${EVAL_LANGUAGES_RAW//,/ }"
read -r -a EVAL_LANGUAGE_LIST <<< "$EVAL_LANGUAGES_RAW"
if [[ "$#" -gt 0 ]]; then
  SCORE_BUDGETS=("$@")
else
  SCORE_BUDGETS=(10 100)
fi

GENERATION_BUDGET="${GENERATION_BUDGET:-100}"
if [[ "${#SCORE_BUDGETS[@]}" -gt 0 ]]; then
  GENERATION_BUDGET="${SCORE_BUDGETS[0]}"
  for score_budget in "${SCORE_BUDGETS[@]}"; do
    if (( score_budget > GENERATION_BUDGET )); then
      GENERATION_BUDGET="$score_budget"
    fi
  done
fi

cmd=(
  "$ROOT_DIR/wixarika/bin/python" -m test.eval_lora
  --model-name-or-path "$MODEL_PATH"
  --dataset-path "$DATASET_PATH"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --generation-budget "$GENERATION_BUDGET"
  --dtype "$DTYPE"
  --results-dir "$RESULTS_DIR"
  --num-examples "$NUM_EXAMPLES"
)

if [[ -n "$EVAL_LANGUAGE" ]]; then
  cmd+=(--languages "$EVAL_LANGUAGE")
else
  if [[ "${#EVAL_LANGUAGE_LIST[@]}" -eq 0 ]]; then
    echo "No evaluation languages configured." >&2
    exit 1
  fi

  EVAL_LANGUAGES_CSV="$(IFS=,; echo "${EVAL_LANGUAGE_LIST[*]}")"
  cmd+=(--languages "$EVAL_LANGUAGES_CSV")
fi

if [[ "${#SCORE_BUDGETS[@]}" -gt 0 ]]; then
  cmd+=(--score-budgets "${SCORE_BUDGETS[@]}")
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ "$SHOW_EXAMPLES" == "1" ]]; then
  cmd+=(--show-examples)
fi

"${cmd[@]}"
