#!/usr/bin/env bash
#SBATCH --job-name=eval-captioning-dev
#SBATCH --output=logs/eval_captioning_dev_%j_output.log
#SBATCH --error=logs/eval_captioning_dev_%j_error.log
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
DATASET_PATH="${DATASET_PATH:-data/captioning}"
SPLIT="${SPLIT:-validation}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bfloat16}"
VERSION="${VERSION:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dev_captioning/version-${VERSION}}"
TEAM_NAME="${TEAM_NAME:-Mila-dev}"
LANGUAGES="${LANGUAGES:-wixarika,bribri,guarani,nahuatl}"
HELDOUT_PERCENTAGE="${HELDOUT_PERCENTAGE:-0.1}"
HELDOUT_SEED="${HELDOUT_SEED:-42}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.8}"

mkdir -p "$OUTPUT_DIR"

caption_cmd=(
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
  --languages "$LANGUAGES"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
)

if [[ "$DO_SAMPLE" == "1" ]]; then
  caption_cmd+=(--do-sample)
fi

"${caption_cmd[@]}"

"$ROOT_DIR/wixarika/bin/python" -m test.captioning_score \
  --model-name-or-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --split "$SPLIT" \
  --predictions-file "$OUTPUT_DIR/predictions.checkpoint.jsonl" \
  --output-file "$OUTPUT_DIR/dev_chrf_summary.json" \
  --records-output-file "$OUTPUT_DIR/dev_scored_records.jsonl" \
  --languages "$LANGUAGES" \
  --heldout-percentage "$HELDOUT_PERCENTAGE" \
  --heldout-seed "$HELDOUT_SEED"
