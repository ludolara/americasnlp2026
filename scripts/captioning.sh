#!/usr/bin/env bash
#SBATCH --job-name=captioning
#SBATCH --output=logs/captioning_output.log
#SBATCH --error=logs/captioning_error.log
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

MODEL_PATH="${MODEL_PATH:-outputs/aya-vision-32b-americas-captioning}"
if [[ "$#" -gt 0 ]]; then
  MODEL_PATH="$1"
  shift
fi

DATASET_PATH="${DATASET_PATH:-data/captioning}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_DIR="${OUTPUT_DIR:-results/captioning}"
TEAM_NAME="${TEAM_NAME:-mila}"
VERSION="${VERSION:-0}"
LIMIT="${LIMIT:-}"
LANGUAGES="${LANGUAGES:-wixarika,bribri,guarani,nahuatl}"
ZIP_FILE="${ZIP_FILE:-}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.8}"

cmd=(
  "$ROOT_DIR/wixarika/bin/python" -m test.captioning
  --model-name-or-path "$MODEL_PATH"
  --dataset-path "$DATASET_PATH"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --output-dir "$OUTPUT_DIR"
  --team-name "$TEAM_NAME"
  --version "$VERSION"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ -n "$LANGUAGES" ]]; then
  cmd+=(--languages "$LANGUAGES")
fi

if [[ -n "$ZIP_FILE" ]]; then
  cmd+=(--zip-file "$ZIP_FILE")
fi

if [[ "$DO_SAMPLE" == "1" ]]; then
  cmd+=(--do-sample)
fi

cmd+=("$@")

"${cmd[@]}"
