#!/usr/bin/env bash
#SBATCH --job-name=lora_sft
#SBATCH --output=logs/lora_sft_output.log
#SBATCH --error=logs/lora_sft_error.log
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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

SFT_CONFIG="configs/lora_sft.yaml"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NUM_GPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#CUDA_DEVICES[@]}"
fi
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  ./wixarika/bin/accelerate launch \
    --multi_gpu \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m train.lora_sft \
    --config "$SFT_CONFIG"
else
  ./wixarika/bin/python -m train.lora_sft --config "$SFT_CONFIG"
fi

TEST_LANGUAGES=(hch bzd gn nah)

for language in "${TEST_LANGUAGES[@]}"; do
  job_id="$(
    sbatch --parsable \
      --job-name="test_lora_sft_${language}" \
      --output="logs/test_lora_sft_${language}_output.log" \
      --error="logs/test_lora_sft_${language}_error.log" \
      --export="ALL,EVAL_LANGUAGE=${language}" \
      scripts/test_lora_sft.sh
  )"
  echo "Submitted test job ${job_id} for language ${language}."
done
