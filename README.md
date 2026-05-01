# Aya Vision Americas

This repository contains the Mila entry code for the
[AmericasNLP 2026 Shared Task: Cultural Image Captioning for Indigenous Languages](https://turing.iimas.unam.mx/americasnlp/2026_st.html).
The shared task asks systems to generate culturally grounded captions for unseen
images in Indigenous languages of the Americas. Official first-stage ranking is
by ChrF++, followed by human evaluation for the top systems.

The system adapts `aya-vision-32b` with LoRA. The main idea is to warm up the
model on Spanish-to-Indigenous-language translation before adapting it to the
vision captioning task.

## Pipeline

1. Build local Hugging Face datasets for translation and captioning.
2. Train a multilingual LoRA SFT adapter on AmericasNLP machine translation.
3. Optionally refine the translation adapter with GRPO using ChrF++ rewards.
4. Train a captioning LoRA SFT adapter on labeled captioning development data.
5. Generate per-language JSONL files and a zipped shared-task submission.

The checked-in configs target Wixarika, Bribri, Guarani, and Nahuatl. The task
also includes Yucatec Maya; the captioning submission code can handle another
language if the raw data is present and the language filters/configs are updated.

## Repository Layout

```text
configs/
  lora_sft.yaml                 # MT LoRA SFT config
  lora_grpo.yaml                # optional MT GRPO refinement config
  captioning_lora_sft.yaml      # final captioning LoRA SFT config
  deepspeed_lora_zero3.json     # DeepSpeed ZeRO-3 config

scripts/
  build_americasnlp2026_hf.py   # raw MT files -> data/americasnlp2026
  build_captioning_hf.py        # raw captioning files -> data/captioning
  lora_sft.sh                   # Slurm/local wrapper for MT SFT
  lora_grpo.sh                  # Slurm/local wrapper for MT GRPO
  captioning_lora_sft.sh        # Slurm/local wrapper for captioning SFT
  test_captioning_lora_sft.sh   # submission generation wrapper

src/train/
  lora_sft.py                   # MT SFT entrypoint
  lora_grpo.py                  # MT GRPO entrypoint
  captioning_lora_sft.py        # captioning SFT entrypoint

src/test/
  eval_lora.py                  # MT generation/evaluation
  captioning.py                 # captioning submission generation
```

Large artifacts are expected to live outside git under `data/`, `models/`,
`outputs/`, and `results/`.

## Setup

Create the environment from the repository root:

```bash
uv venv --python /usr/bin/python3 wixarika
source wixarika/bin/activate
uv pip install -r requirements.txt
mkdir -p logs outputs results models/.hf
```

The Slurm wrappers set the runtime environment automatically. For direct Python
commands, use the same environment:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="$PWD/models/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The default configs expect the Aya Vision backbone at:

```text
models/aya-vision-32b
```

Set `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` only if you intentionally
want Hugging Face downloads during a run.

## Data

### Machine Translation

The MT builder reads raw Spanish-parallel data from:

```text
data/raw/wixarika-spanish
data/raw/bribri-spanish
data/raw/guarani-spanish
data/raw/nahuatl-spanish
```

It writes a local Hugging Face `DatasetDict` to:

```text
data/americasnlp2026
```

Build or rebuild it with:

```bash
PYTHONPATH=src ./wixarika/bin/python scripts/build_americasnlp2026_hf.py --overwrite
```

The MT dataset columns used by `configs/lora_sft.yaml` are:

| Column | Meaning |
| --- | --- |
| `es` | Spanish source text |
| `target` | Indigenous-language target text |
| `language` | Language name, for example `wixarika` or `bribri` |
| `language_code` | Language code, for example `hch`, `bzd`, `gn`, `nah` |

### Captioning

The captioning builder expects raw task data under:

```text
data/raw/captioning/dev/<language>/<language>.jsonl
data/raw/captioning/test/<language>/<language>.jsonl
```

Images may be either next to the JSONL file or inside an `images/` directory for
that language. The builder maps raw `dev` to the Hugging Face `validation` split
and raw `test` to the `test` split:

```bash
PYTHONPATH=src ./wixarika/bin/python scripts/build_captioning_hf.py --overwrite
```

The output dataset is:

```text
data/captioning
```

Captioning rows keep the official task metadata:

```text
id, filename, split, culture, language, iso_lang
```

The `validation` split also contains `target_caption`. The `test` split is used
for submission generation.

## Model Artifacts

The default artifact flow is:

| Stage | Config | Input | Output |
| --- | --- | --- | --- |
| MT SFT | `configs/lora_sft.yaml` | `models/aya-vision-32b` | `outputs/aya-vision-32b-americas` |
| MT GRPO | `configs/lora_grpo.yaml` | `outputs/aya-vision-32b-americas` | `outputs/aya-vision-32b-americas-grpo` |
| Captioning SFT | `configs/captioning_lora_sft.yaml` | `outputs/aya-vision-32b-americas-grpo/checkpoint-125` | `outputs/aya-vision-32b-americas-grpo-captioning` |

To train captioning directly from the MT SFT adapter instead of the GRPO
checkpoint, update `model_name_or_path` and `output_dir` in
`configs/captioning_lora_sft.yaml`.

## Training

### 1. MT LoRA SFT

On Slurm:

```bash
sbatch scripts/lora_sft.sh
```

The wrapper trains the MT SFT adapter, then submits MT evaluation jobs and the
GRPO follow-up job. To run only the MT SFT entrypoint:

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.lora_sft \
  --config configs/lora_sft.yaml
```

For a manual multi-GPU launch:

```bash
PYTHONPATH=src ./wixarika/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m train.lora_sft \
  --config configs/lora_sft.yaml
```

### 2. Optional MT GRPO

Run this stage after `outputs/aya-vision-32b-americas` exists:

```bash
sbatch scripts/lora_grpo.sh
```

Direct entrypoint:

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.lora_grpo \
  --config configs/lora_grpo.yaml
```

This stage uses `data/americasnlp2026`, samples languages with
`language_sampling_alpha`, and evaluates on ChrF++ during training.

### 3. Captioning LoRA SFT

The captioning config currently starts from the GRPO checkpoint and trains on
the labeled captioning `validation` split:

```bash
sbatch scripts/captioning_lora_sft.sh
```

For a local shell run:

```bash
bash scripts/captioning_lora_sft.sh
```

Direct entrypoint:

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.captioning_lora_sft \
  --config configs/captioning_lora_sft.yaml
```

Useful captioning config fields:

| Field | Default purpose |
| --- | --- |
| `dataset_path` | `data/captioning` |
| `train_split` | `validation` |
| `languages` | selected captioning languages |
| `eval_percentage` | per-language holdout used for runtime ChrF++ eval |
| `prompt_template` | instruction passed with each image |
| `output_dir` | final captioning adapter directory |

## Generate Submission Files

Use the captioning test wrapper after the final adapter exists:

```bash
LANGUAGES=wixarika,bribri,guarani,nahuatl \
TEAM_NAME=Mila \
VERSION=0 \
bash scripts/test_captioning_lora_sft.sh
```

For a quick smoke test:

```bash
LIMIT=10 bash scripts/test_captioning_lora_sft.sh
```

Equivalent direct command:

```bash
PYTHONPATH=src ./wixarika/bin/python -m test.captioning \
  --model-name-or-path outputs/aya-vision-32b-americas-grpo-captioning \
  --dataset-path data/captioning \
  --split test \
  --languages wixarika,bribri,guarani,nahuatl \
  --output-dir results/captioning \
  --team-name Mila \
  --version 0
```

The generator:

1. Loads the selected captioning split.
2. Filters by `--languages`, matching either `submission_language` or `iso_lang`.
3. Resumes from `results/captioning/predictions.checkpoint.jsonl` when present.
4. Writes per-language JSONL files under `results/captioning/submission/`.
5. Creates `results/captioning/<TEAM_NAME>.zip`.

Expected submission files look like:

```text
bribri-0.jsonl
guarani-0.jsonl
nahuatl-0.jsonl
wixarika-0.jsonl
```

Each row preserves the official task metadata and adds:

```text
predicted_caption
```

## Common Overrides

The submission wrapper reads these environment variables:

| Variable | Default |
| --- | --- |
| `MODEL_PATH` | `outputs/aya-vision-32b-americas-grpo-captioning` |
| `DATASET_PATH` | `data/captioning` |
| `SPLIT` | `test` |
| `LANGUAGES` | `wixarika,bribri,guarani,nahuatl` |
| `OUTPUT_DIR` | `results/captioning` |
| `TEAM_NAME` | `Mila` |
| `VERSION` | `0` |
| `LIMIT` | unset |
| `DO_SAMPLE` | `0` |
| `TEMPERATURE` | `0.4` |
| `TOP_P` | `0.8` |

Example:

```bash
MODEL_PATH=outputs/aya-vision-32b-americas-grpo-captioning \
LANGUAGES=wixarika,bribri \
TEAM_NAME=Mila \
VERSION=1 \
bash scripts/test_captioning_lora_sft.sh
```

## Evaluation Helpers

Translation adapters can be evaluated with:

```bash
bash scripts/test_lora_sft.sh
bash scripts/test_lora_grpo.sh
```

Both wrappers call `python -m test.eval_lora` and support language filters such
as:

```bash
EVAL_LANGUAGE=hch bash scripts/test_lora_sft.sh
EVAL_LANGUAGES=hch,bzd,gn,nah bash scripts/test_lora_grpo.sh
```

Captioning SFT can run a small held-out validation evaluation by setting
`eval_percentage` in `configs/captioning_lora_sft.yaml`.

## Task Reference

Task rules, language list, dates, and submission instructions are maintained by
the organizers at:

```text
https://turing.iimas.unam.mx/americasnlp/2026_st.html
```

A local copy of the task description is also kept in
[`captioning.md`](captioning.md).
