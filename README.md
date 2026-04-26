# Aya Vision Americas

This repository trains and runs Aya Vision for AmericasNLP translation and image captioning.

The backbone model is `aya-vision-32b`.

The current workflow is:

1. LoRA SFT for machine translation.
2. LoRA SFT for image captioning, initialized from the translation model.
3. Caption generation.

## Setup

```bash
uv venv --python /usr/bin/python3 wixarika
source wixarika/bin/activate
uv pip install -r requirements.txt
```

The Slurm scripts set these environment variables by default:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="$PWD/models/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Data

Machine translation uses the local Hugging Face dataset:

```text
data/americasnlp2026
```

Expected splits are `train`, `validation`, and `test`. The MT columns used by the config are:

- `es`: Spanish source text
- `target`: Indigenous-language target text
- `language`: language name, such as `wixarika`, `bribri`, `guarani`, `nahuatl`
- `language_code`: language code, such as `hch`, `bzd`, `gn`, `nah`

Image captioning uses:

```text
data/captioning
```

Build it from raw captioning data when needed:

```bash
PYTHONPATH=src ./wixarika/bin/python scripts/build_captioning_hf.py --overwrite
```

The captioning dataset stores `validation` examples with labels and `test` examples for submission generation.

## MT SFT

The translation SFT config is [`configs/lora_sft.yaml`](configs/lora_sft.yaml).

It starts from the `aya-vision-32b` backbone:

```text
models/aya-vision-32b
```

and writes the trained Americas translation adapter to:

```text
outputs/aya-vision-32b-americas
```

Run only the MT SFT training entrypoint with:

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.lora_sft --config configs/lora_sft.yaml
```

For a multi-GPU run:

```bash
PYTHONPATH=src ./wixarika/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m train.lora_sft \
  --config configs/lora_sft.yaml
```

Important config fields:

- `local_dataset_path: data/americasnlp2026`
- `bidirectional_translation: true`
- `language_sampling_alpha: 0.3`
- `max_seq_length: 256`
- `save_only_model: true`
- `deepspeed: configs/deepspeed_lora_zero3.json`

## Image Captioning SFT

Captioning SFT starts from the MT SFT output:

```text
outputs/aya-vision-32b-americas
```

and writes the captioning adapter to:

```text
outputs/aya-vision-32b-americas-captioning
```

Run it with the Slurm/local wrapper:

```bash
./scripts/captioning_lora_sft.sh
```

or directly:

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.captioning_lora_sft \
  --model-name-or-path outputs/aya-vision-32b-americas \
  --dataset-path data/captioning \
  --train-split validation \
  --languages hch,bzd,grn \
  --output-dir outputs/aya-vision-32b-americas-captioning
```

Useful overrides:

```bash
MODEL_PATH=outputs/aya-vision-32b-americas \
DATASET_PATH=data/captioning \
TRAIN_SPLIT=validation \
LANGUAGES=hch,bzd,grn \
OUTPUT_DIR=outputs/aya-vision-32b-americas-captioning \
./scripts/captioning_lora_sft.sh
```

The default captioning prompt asks Aya Vision to produce one culturally appropriate caption in the requested language and to output only the caption.

## Captioning Process

Generate captioning predictions from the captioning SFT output:

```bash
./scripts/captioning.sh
```

By default this uses:

```text
MODEL_PATH=outputs/aya-vision-32b-americas-captioning
DATASET_PATH=data/captioning
SPLIT=test
LANGUAGES=hch,bzd,grn
OUTPUT_DIR=results/captioning
TEAM_NAME=mila
VERSION=0
```

The process:

1. Loads the selected captioning split.
2. Filters by `LANGUAGES`.
3. Resumes from `results/captioning/predictions.checkpoint.jsonl` if it exists.
4. Generates captions with Aya Vision.
5. Writes per-language JSONL files under `results/captioning/submission/`.
6. Creates the submission zip at `results/captioning/<TEAM_NAME>.zip`.

Example custom run:

```bash
MODEL_PATH=outputs/aya-vision-32b-americas-captioning \
LANGUAGES=hch,bzd,grn \
TEAM_NAME=mila \
VERSION=0 \
./scripts/captioning.sh
```

For a quick smoke test:

```bash
LIMIT=10 ./scripts/captioning.sh
```

The generated submission files are named like:

```text
bribri-0.jsonl
guarani-0.jsonl
wixarika-0.jsonl
```

Each row includes the original image metadata plus `predicted_caption`.
