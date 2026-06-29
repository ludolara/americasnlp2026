# Mila's Submission — AmericasNLP 2026

<!-- Replace the arXiv placeholder below once the paper is public. -->
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![HF Model: SFT(MT)](https://img.shields.io/badge/HF%20Model-SFT%28MT%29-FFD21E.svg)](https://huggingface.co/ludolara/aya-vision-32b-americas)
[![HF Model: SFT(MT) + SFT(IC)](https://img.shields.io/badge/HF%20Model-SFT%28MT%29%20%2B%20SFT%28IC%29-FFD21E.svg)](https://huggingface.co/ludolara/aya-vision-32b-americas-captioning)
[![HF Model: SFT(MT) + RLVR(MT) + SFT(IC)](https://img.shields.io/badge/HF%20Model-SFT%28MT%29%20%2B%20RLVR%28MT%29%20%2B%20SFT%28IC%29-FFD21E.svg)](https://huggingface.co/ludolara/aya-vision-32b-americas-grpo-captioning)

This repository contains the Mila entry code for the
[AmericasNLP 2026 Shared Task: Cultural Image Captioning for Indigenous Languages](https://turing.iimas.unam.mx/americasnlp/2026_st.html).
The shared task asks systems to generate culturally grounded captions for unseen
images in Indigenous languages of the Americas. Official first-stage ranking is
by ChrF++, followed by human evaluation for the top systems.

The system adapts `aya-vision-32b` with LoRA. The main idea is to warm up the
model on Spanish-to-Indigenous-language translation before adapting it to the
vision captioning task.

## Pipeline

1. Train a multilingual LoRA SFT adapter on AmericasNLP machine translation.
2. Optionally refine the translation adapter with GRPO using ChrF++ rewards.
3. Train a captioning LoRA SFT adapter on labeled captioning development data.
4. Generate per-language JSONL files and a zipped shared-task submission.

The checked-in configs target Wixarika, Bribri, Guarani, and Nahuatl. The captioning submission code can handle another
language if the raw data is present and the language filters/configs are updated.

## Setup

Create the environment from the repository root:

```bash
uv venv --python /usr/bin/python3 wixarika
source wixarika/bin/activate
uv pip install -r requirements.txt
mkdir -p logs outputs results models/.hf
```

The default configs expect the Aya Vision backbone at:

```text
models/aya-vision-32b
```

Large artifacts are expected to live outside git under `data/`, `models/`,
`outputs/`, and `results/`.

## Training

### 1. Machine Translation LoRA SFT

On Slurm:

```bash
sbatch scripts/lora_sft.sh
```

The wrapper trains the MT SFT adapter, then submits MT evaluation jobs and the
GRPO follow-up job.

### 2. Optional Machine Translation RLVR

Run this stage after `outputs/aya-vision-32b-americas` exists:

```bash
sbatch scripts/lora_grpo.sh
```

This stage uses `data/americasnlp2026`, samples languages with
`language_sampling_alpha`, and evaluates on ChrF++ during training.

### 3. Image Captioning LoRA SFT

```bash
sbatch scripts/captioning_lora_sft.sh
```

## Model Artifacts

The default artifact flow is:

| Stage | Config | Input | Output |
| --- | --- | --- | --- |
| MT SFT | `configs/lora_sft.yaml` | `models/aya-vision-32b` | `outputs/aya-vision-32b-americas` |
| MT GRPO | `configs/lora_grpo.yaml` | `outputs/aya-vision-32b-americas` | `outputs/aya-vision-32b-americas-grpo` |
| Captioning SFT | `configs/captioning_lora_sft.yaml` | `outputs/aya-vision-32b-americas-grpo` | `outputs/aya-vision-32b-americas-grpo-captioning` |

To train captioning directly from the MT SFT adapter instead of the GRPO
checkpoint, update `model_name_or_path` and `output_dir` in
`configs/captioning_lora_sft.yaml`.

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

Put the official captioning release under `data/raw/captioning`:

```text
data/raw/captioning/
  dev/
    bribri/bribri.jsonl
    guarani/guarani.jsonl
    nahuatl/nahuatl.jsonl
    wixarika/wixarika.jsonl
  test/
    bribri/bribri.jsonl
    guarani/guarani.jsonl
    nahuatl/nahuatl.jsonl
    wixarika/wixarika.jsonl
```

Images can be placed beside each JSONL file or in that language folder's
`images/` directory. The JSONL rows should include the official fields
`id`, `filename`, `split`, `culture`, `language`, and `iso_lang`; the dev rows
also include `target_caption`.

Build the local Hugging Face dataset with:

```bash
PYTHONPATH=src ./wixarika/bin/python scripts/build_captioning_hf.py --overwrite
```

This writes:

```text
data/captioning
```

Raw `dev` becomes the `validation` split for captioning SFT. Raw `test` becomes
the `test` split for submission generation.

## Citation

```bibtex
@inproceedings{lara-raval-2026-machine,
    title = "From Machine Translation to Image Captioning: Training Vision-Language Models for Indigenous Languages of the {A}mericas",
    author = "Lara, Luis  and
      Raval, Param",
    editor = "Mager, Manuel  and
      Ebrahimi, Abteen  and
      Bui, Minh Duc  and
      Pugh, Robert  and
      Oncevay, Arturo  and
      Chiruzzo, Luis  and
      Solano, Rolando Coto  and
      Rijhwani, Shruti  and
      Von Der Wense, Katharina",
    booktitle = "Proceedings of the Sixth Workshop on {NLP} for Indigenous Languages of the {A}mericas ({A}mericas{NLP})",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://preview.aclanthology.org/ingest-acl-workshops/2026.americasnlp-6.20/",
    pages = "224--235",
    ISBN = "979-8-89176-415-6"
}
```

## Acknowledgements 

We thank the AmericasNLP 2026 organizers for preparing and releasing the Cultural Image Captioning shared-task data and evaluation setup. We also acknowledge Cohere For AI and Cohere Labs for releasing Aya Vision, which served as the base vision-language model in our experiments. We are grateful to the creators of the machine-translation resources used in this work, including Axolotl for Spanish--Nahuatl (Gutiérrez-Vásques et al., 2016), the Wixárika resource derived from Mager et al. (2018), the Guaraní--Spanish corpus by Chiruzzo et al. (2020), the Bribri resource by Feldman and Coto-Solano (2020), and the AmericasNLP 2025 ST1 data following De Gibert et al. (2025) and De Gibert et al. (2023).
