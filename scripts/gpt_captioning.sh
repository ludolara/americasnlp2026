#!/usr/bin/env bash
#SBATCH --job-name=gpt-captioning
#SBATCH --output=logs/gpt_captioning_output.log
#SBATCH --error=logs/gpt_captioning_error.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --partition=long

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/wixarika/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/wixarika/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export RAW_DIR="${RAW_DIR:-data/raw/captioning}"
export SPLIT="${SPLIT:-test}"
if [[ -z "${LANGUAGES:-}" && -z "${LANGUAGE:-}" ]]; then
  export LANGUAGES="bribri,guarani,nahuatl,wixarika"
fi
export VERSION="${VERSION:-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_captioning}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"
export REASONING_EFFORT="${REASONING_EFFORT:-medium}"
export IMAGE_DETAIL="${IMAGE_DETAIL:-high}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/gpt_captioning.py" "$@"
