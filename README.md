# Tiny Aya Translation Training with SFT and GRPO

Minimal codebase to run:

- supervised fine-tuning (SFT) with `trl.SFTTrainer`
- GRPO fine-tuning with `trl.GRPOTrainer`
- BLEU/chrF evaluation for translation checkpoints with chrF++ sentence selection

## What this does

- Full model fine-tuning (no LoRA/PEFT)
- Works with:
  - local `json/jsonl/csv/parquet` files
  - local Hugging Face datasets saved with `save_to_disk`
  - Hugging Face Hub datasets (`dataset_name`)
- Supports common schemas:
  - `text`
  - `prompt` + `completion`
  - `instruction` + `input` + `output`
  - `es` + `wix` (formatted as translation SFT prompts)
  - `messages` (chat template via tokenizer)

## Setup

```bash
uv venv --python /usr/bin/python3 wixarika
source wixarika/bin/activate
uv pip install -r requirements.txt
```

## Configure

Edit one of:

- `configs/full_sft.yaml`: full fine-tuning for the text Tiny Aya model
- `configs/lora_sft.yaml`: LoRA fine-tuning for Aya Vision using text-only examples
- Choose one data source:
  - `local_dataset_path` (HF `save_to_disk` folder), or
  - `dataset_name` (HF dataset), or
  - `train_file` / `eval_file` (local files)
- Optional: set `early_stopping_patience` to stop after that many evals without improvement in `eval_loss`
- Optional: set `early_stopping_threshold` to require a minimum `eval_loss` improvement before patience resets

## Run Full SFT

```bash
./scripts/full_sft.sh
```

or

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.full_sft --config configs/full_sft.yaml
```

## Run LoRA SFT

```bash
./scripts/lora_sft.sh
```

or

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.lora_sft --config configs/lora_sft.yaml
```

## Run GRPO

The default GRPO config starts from the SFT checkpoint and uses sentence-level `chrF++` as the reward.

```bash
./scripts/grpo.sh
```

or

```bash
PYTHONPATH=src ./wixarika/bin/python -m train.grpo --config configs/grpo.yaml
```

## Notes for full fine-tuning

- Full fine-tuning is memory-heavy. Start with:
  - low batch size (`1`)
  - gradient accumulation
  - gradient checkpointing
  - bf16 if supported by your GPU
- If OOM:
  - lower `max_seq_length`
  - lower `per_device_train_batch_size`
  - increase `gradient_accumulation_steps`
- If you enable `early_stopping_patience`, keep `save_steps` as a multiple of `eval_steps`

## Expected dataset examples

Instruction format (`jsonl`):

```json
{"instruction":"Explain overfitting","input":"for a beginner","output":"Overfitting is when..."}
```

Prompt/completion format:

```json
{"prompt":"User: Explain overfitting\nAssistant:","completion":" Overfitting is when..."}
```

Pre-rendered text format:

```json
{"text":"<|user|> Explain overfitting\n<|assistant|> Overfitting is when..."}
```

## Test Evaluation

Run BLEU/chrF evaluation on the dataset `test` split. The evaluator still uses sentence-level `chrF++` for best-of-n candidate selection:

```bash
./scripts/test.sh
```

Print five qualitative generation examples as well:

```bash
./scripts/test.sh --show-examples
```

Evaluate only specific target languages:

```bash
./scripts/test.sh --languages wixarika
```
