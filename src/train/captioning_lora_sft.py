from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, set_seed
from transformers.trainer_utils import EvalLoopOutput

from train.config import LoraSFTTrainConfig, pretty_config
from train.data import ensure_chat_template
from train.lora_sft import (
    _export_best_checkpoint,
    _resolve_resume_checkpoint,
    _validate_saved_adapter,
)
from train.sft_common import (
    MultilingualSFTTrainer,
    build_callbacks,
    build_sft_args,
    build_train_sampler,
)


DEFAULT_PROMPT_TEMPLATE = (
    "Escribe un solo pie de foto en {language} para esta imagen. "
    "Debe ser una descripcion culturalmente adecuada de la imagen. "
    "Responde solo con el pie de foto en {language}, sin explicaciones."
)
DEFAULT_CONFIG_PATH = "configs/captioning_lora_sft.yaml"
DEFAULT_LANGUAGES = ["wixarika", "bribri", "guarani", "nahuatl"]
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _parse_string_list(
    raw_values: str | list[str] | None,
    *,
    arg_name: str,
) -> list[str] | None:
    if raw_values is None:
        return None
    if isinstance(raw_values, str):
        raw_items = [raw_values]
    elif isinstance(raw_values, list):
        raw_items = raw_values
    else:
        raise TypeError(f"{arg_name} must be a string, list of strings, or null.")

    values: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str):
            raise TypeError(f"{arg_name} entries must be strings.")
        for item in raw_item.split(","):
            cleaned = item.strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    if not values:
        raise ValueError(f"{arg_name} must include at least one non-empty value.")
    return values


def _parse_languages(raw_languages: str | list[str] | None) -> list[str] | None:
    return _parse_string_list(raw_languages, arg_name="--languages")


def _load_config_defaults(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected YAML mapping in {config_path!r}.")
    return dict(raw)


def _config_default(defaults: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return defaults[key] if key in defaults else fallback


def _build_arg_parser(
    *,
    defaults: Mapping[str, Any],
    config_path: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA SFT for Aya Vision on labeled captioning validation data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to YAML training config.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=_config_default(
            defaults,
            "model_name_or_path",
            "outputs/v3/aya-vision-32b-americas",
        ),
        help="Merged model path, base model path, or existing LoRA adapter to continue.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=_config_default(defaults, "dataset_path", "data/captioning"),
        help="Path to the captioning Hugging Face dataset saved with save_to_disk.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=_config_default(defaults, "train_split", "validation"),
        help="Captioning split with target_caption labels.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=_config_default(defaults, "languages", DEFAULT_LANGUAGES),
        help="Language filter by iso_lang, submission language, culture, or language name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_config_default(
            defaults,
            "output_dir",
            "outputs/aya-vision-32b-americas-captioning",
        ),
        help="Directory where the captioning LoRA adapter is saved.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=_config_default(defaults, "prompt_template", DEFAULT_PROMPT_TEMPLATE),
        help="Python format string with {language}, {iso_lang}, and {culture}.",
    )
    parser.add_argument(
        "--eval-percentage",
        type=float,
        default=_config_default(defaults, "eval_percentage", None),
        help=(
            "Optional per-language fraction of captioning examples to hold out for "
            "runtime evaluation. The held-out rows are removed from training."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=_config_default(defaults, "max_seq_length", 4096),
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=_config_default(defaults, "per_device_train_batch_size", 1),
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=_config_default(defaults, "per_device_eval_batch_size", 1),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=_config_default(defaults, "gradient_accumulation_steps", 8),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_config_default(defaults, "learning_rate", 2.0e-5),
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=_config_default(defaults, "num_train_epochs", 20.0),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=_config_default(defaults, "warmup_ratio", 0.03),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=_config_default(defaults, "weight_decay", 0.01),
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default=_config_default(defaults, "lr_scheduler_type", "linear"),
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=_config_default(defaults, "logging_steps", 5),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=_config_default(defaults, "save_steps", 25),
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=_config_default(defaults, "save_total_limit", 3),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=_config_default(defaults, "early_stopping_patience", None),
        help=(
            "Stop after this many eval rounds without improving held-out "
            "caption ChrF++."
        ),
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=_config_default(defaults, "early_stopping_threshold", None),
        help="Minimum ChrF++ improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--language-sampling-alpha",
        type=float,
        default=_config_default(defaults, "language_sampling_alpha", 1.0),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=_config_default(defaults, "lora_r", 32),
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=_config_default(defaults, "lora_alpha", 64),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=_config_default(defaults, "lora_dropout", 0.1),
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default=_config_default(defaults, "lora_bias", "none"),
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=_config_default(
            defaults,
            "lora_target_modules",
            DEFAULT_LORA_TARGET_MODULES,
        ),
        help="LoRA target modules. Accepts comma-separated values, space-separated values, or both.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=_config_default(defaults, "bf16", True),
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=_config_default(defaults, "fp16", False),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=_config_default(defaults, "gradient_checkpointing", True),
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=_config_default(
            defaults,
            "deepspeed",
            "configs/deepspeed_lora_zero3.json",
        ),
        help="DeepSpeed config path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=_config_default(defaults, "ddp_find_unused_parameters", False),
    )
    parser.add_argument("--seed", type=int, default=_config_default(defaults, "seed", 42))
    parser.add_argument(
        "--report-to",
        type=str,
        default=_config_default(defaults, "report_to", "none"),
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=_config_default(defaults, "dry_run", False),
        help="Build the captioning dataset and config, then exit before loading the model.",
    )
    return parser


def _validate_config_keys(
    defaults: Mapping[str, Any],
    parser: argparse.ArgumentParser,
) -> None:
    allowed = {
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(defaults) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in captioning LoRA SFT config: {', '.join(unknown)}"
        )


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()

    defaults = _load_config_defaults(config_args.config)
    parser = _build_arg_parser(defaults=defaults, config_path=config_args.config)
    _validate_config_keys(defaults, parser)
    args = parser.parse_args()
    args.languages = _parse_languages(args.languages)
    args.lora_target_modules = _parse_string_list(
        args.lora_target_modules,
        arg_name="--lora-target-modules",
    )
    if args.max_seq_length < 1:
        raise ValueError("--max-seq-length must be at least 1.")
    if args.per_device_train_batch_size < 1:
        raise ValueError("--per-device-train-batch-size must be at least 1.")
    if args.per_device_eval_batch_size < 1:
        raise ValueError("--per-device-eval-batch-size must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1.")
    if args.save_steps < 1:
        raise ValueError("--save-steps must be at least 1.")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1 when provided.")
    if args.early_stopping_threshold is not None and args.early_stopping_threshold < 0:
        raise ValueError("--early-stopping-threshold must be non-negative when provided.")
    if args.eval_percentage is not None and not 0 < args.eval_percentage < 1:
        raise ValueError("--eval-percentage must be between 0 and 1 when provided.")
    if args.language_sampling_alpha < 0:
        raise ValueError("--language-sampling-alpha must be non-negative.")
    return args


def _load_adapter_base_model_name(model_name_or_path: str) -> str | None:
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    with adapter_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    base_model_name = str(config.get("base_model_name_or_path") or "").strip()
    return base_model_name or None


def _assert_adapter_has_weights(model_name_or_path: str) -> None:
    adapter_path = Path(model_name_or_path) / "adapter_model.safetensors"
    if not adapter_path.exists():
        return

    from safetensors import safe_open

    with safe_open(adapter_path, framework="pt") as f:
        if any(True for _ in f.keys()):
            return

    raise ValueError(
        f"Adapter at {adapter_path} contains zero tensors; refusing to continue training."
    )


def _format_caption_prompt(example: Mapping[str, Any], prompt_template: str) -> str:
    return prompt_template.format(
        language=str(example.get("language") or "").strip(),
        iso_lang=str(example.get("iso_lang") or "").strip(),
        culture=str(example.get("culture") or "").strip(),
    )


def _matches_language(example: Mapping[str, Any], requested: set[str] | None) -> bool:
    if requested is None:
        return True
    values = {
        str(example.get("iso_lang") or "").strip(),
        str(example.get("submission_language") or "").strip(),
        str(example.get("culture") or "").strip(),
        str(example.get("language") or "").strip(),
    }
    return bool(values & requested)


def _prepare_captioning_dataset(
    *,
    dataset_path: str,
    split: str,
    languages: list[str] | None,
    prompt_template: str,
) -> Dataset:
    raw = load_from_disk(dataset_path)
    if not isinstance(raw, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {dataset_path!r}.")
    if split not in raw:
        available = ", ".join(raw.keys())
        raise ValueError(f"Missing split '{split}'. Available: {available}")

    dataset = raw[split]
    required_columns = {"image", "target_caption", "language", "iso_lang", "culture"}
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        available = ", ".join(dataset.column_names)
        raise ValueError(f"Missing columns for captioning SFT: {missing}. Available: {available}")

    requested = {language.strip() for language in languages} if languages else None

    def keep_example(example: Mapping[str, Any]) -> bool:
        return bool(str(example.get("target_caption") or "").strip()) and _matches_language(
            example, requested
        )

    dataset = dataset.filter(keep_example, desc=f"Selecting captioning {split} examples")
    if len(dataset) == 0:
        raise ValueError("No labeled captioning examples matched the requested languages.")

    def mapper(example: Mapping[str, Any]) -> dict[str, Any]:
        language = str(example.get("language") or "").strip()
        prompt = _format_caption_prompt(example, prompt_template)
        caption = str(example.get("target_caption") or "").strip()
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                }
            ],
            "language": language,
            "iso_lang": str(example.get("iso_lang") or "").strip(),
            "reference_caption": caption,
        }

    remove_columns = [column for column in dataset.column_names if column != "image"]
    return dataset.map(
        mapper,
        remove_columns=remove_columns,
        desc="Formatting captioning examples for VLM SFT",
    )


def _split_train_eval_dataset(
    dataset: Dataset,
    *,
    eval_percentage: float | None,
    seed: int,
    language_column: str = "language",
) -> tuple[Dataset, Dataset | None, dict[str, Any]]:
    if eval_percentage is None:
        return dataset, None, {}
    if language_column not in dataset.column_names:
        available = ", ".join(dataset.column_names)
        raise ValueError(
            f"Missing language column '{language_column}' for eval split. "
            f"Available: {available}"
        )

    language_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(dataset[language_column]):
        language_to_indices[str(language).strip()].append(index)

    rng = random.Random(seed)
    eval_indices: list[int] = []
    eval_counts: dict[str, int] = {}
    for language, indices in sorted(language_to_indices.items()):
        if len(indices) < 2:
            raise ValueError(
                f"Cannot hold out eval examples for language {language!r}: "
                "at least two rows are required."
            )
        eval_count = int(round(len(indices) * eval_percentage))
        eval_count = max(1, eval_count)
        if eval_count >= len(indices):
            raise ValueError(
                f"Eval percentage {eval_percentage:.4f} would leave no training rows "
                f"for language {language!r}."
            )
        eval_counts[language] = eval_count
        eval_indices.extend(rng.sample(indices, k=eval_count))

    eval_index_set = set(eval_indices)
    train_indices = [
        index
        for index in range(len(dataset))
        if index not in eval_index_set
    ]
    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)
    split_info = {
        "eval_percentage": eval_percentage,
        "eval_seed": seed,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "train_language_counts": dict(
            sorted(Counter(train_dataset[language_column]).items())
        ),
        "eval_language_counts": dict(
            sorted(Counter(eval_dataset[language_column]).items())
        ),
        "requested_eval_counts": eval_counts,
    }
    return train_dataset, eval_dataset, split_info


def _build_config(args: argparse.Namespace) -> LoraSFTTrainConfig:
    return LoraSFTTrainConfig(
        model_name_or_path=args.model_name_or_path,
        local_dataset_path=args.dataset_path,
        train_split=args.train_split,
        eval_split=None,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        packing=False,
        language_sampling_alpha=args.language_sampling_alpha,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=0,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed or None,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_target_modules=args.lora_target_modules,
        seed=args.seed,
        report_to=args.report_to,
    )


def _world_size() -> int:
    for env_var in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_GPUS_ON_NODE"):
        raw_value = os.environ.get(env_var)
        if raw_value:
            try:
                return max(int(raw_value), 1)
            except ValueError:
                continue
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        visible_devices = [
            device.strip()
            for device in cuda_visible_devices.split(",")
            if device.strip()
        ]
        if visible_devices:
            return len(visible_devices)
    raw_slurm_tasks = os.environ.get("SLURM_NTASKS")
    if raw_slurm_tasks:
        try:
            return max(int(raw_slurm_tasks), 1)
        except ValueError:
            pass
    return 1


def _steps_per_epoch(
    *,
    num_train_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    per_rank_examples = math.ceil(num_train_examples / _world_size())
    batches_per_rank = math.ceil(per_rank_examples / per_device_train_batch_size)
    return max(math.ceil(batches_per_rank / gradient_accumulation_steps), 1)


def _extract_assistant_response(text: str) -> str:
    cleaned = text.strip()
    if "<|assistant|>" in cleaned:
        cleaned = cleaned.rsplit("<|assistant|>", maxsplit=1)[-1].strip()
    if "<|CHATBOT_TOKEN|>" in cleaned:
        cleaned = cleaned.rsplit("<|CHATBOT_TOKEN|>", maxsplit=1)[-1].strip()
    if "<|START_RESPONSE|>" in cleaned:
        cleaned = cleaned.rsplit("<|START_RESPONSE|>", maxsplit=1)[-1].strip()
    for marker in ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, maxsplit=1)[0].strip()
    if "<|user|>" in cleaned:
        cleaned = cleaned.split("<|user|>", maxsplit=1)[0].strip()
    return cleaned


def _clean_caption(text: str) -> str:
    cleaned = _extract_assistant_response(text).strip()
    cleaned = re.split(
        r"\s*<\|(?:END|END_RESPONSE|END_OF_TURN_TOKEN|START_OF_TURN_TOKEN)",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = re.sub(r"\s*<\s*$", "", cleaned)
    cleaned = re.sub(
        r"^(predicted_caption|caption|pie de foto)\s*:\s*",
        "",
        cleaned,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _get_input_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def _move_inputs_to_device(inputs: Any, device: Any, dtype: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[name] = value.to(device=device, dtype=dtype)
            else:
                moved[name] = value.to(device=device)
        else:
            moved[name] = value
    return moved


def _reference_caption(example: Mapping[str, Any]) -> str:
    reference = str(example.get("reference_caption") or "").strip()
    if reference:
        return reference

    completion = example.get("completion")
    if isinstance(completion, list):
        for message in completion:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") == "text":
                        return str(item.get("text") or "").strip()
    return ""


class CaptioningChrFTrainer(MultilingualSFTTrainer):
    def __init__(
        self,
        *args: Any,
        caption_eval_dataset: Dataset | None = None,
        caption_eval_max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> None:
        self.caption_eval_dataset = caption_eval_dataset
        self.caption_eval_max_new_tokens = caption_eval_max_new_tokens
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self.caption_eval_dataset is None or metric_key_prefix != "eval":
            return output

        metrics = dict(output.metrics or {})
        metrics.update(self._compute_caption_chrf_metrics(metric_key_prefix))
        return output._replace(metrics=metrics)

    def _generation_model(self) -> Any:
        wrapped_model = getattr(self, "model_wrapped", None)
        if wrapped_model is not None and hasattr(wrapped_model, "generate"):
            return wrapped_model
        if wrapped_model is not None and hasattr(wrapped_model, "module"):
            module = wrapped_model.module
            if hasattr(module, "generate"):
                return module

        model = self.model
        if hasattr(model, "generate"):
            return model

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            unwrapped = accelerator.unwrap_model(model)
            if hasattr(unwrapped, "generate"):
                return unwrapped
        return model

    def _compute_caption_chrf_metrics(self, metric_key_prefix: str) -> dict[str, float]:
        from sacrebleu.metrics import CHRF

        dataset = self.caption_eval_dataset
        if dataset is None or len(dataset) == 0:
            return {}

        model = self._generation_model()
        processor = self.processing_class
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Captioning ChrF++ eval requires a processor tokenizer.")

        batch_size = max(int(getattr(self.args, "per_device_eval_batch_size", 1)), 1)
        dtype = _resolve_dtype_from_args(self.args, model)
        device = _get_input_device(model)

        predictions: list[str] = []
        references: list[str] = []
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            for start in range(0, len(dataset), batch_size):
                examples = [
                    dataset[index]
                    for index in range(start, min(start + batch_size, len(dataset)))
                ]
                prompts = [
                    processor.apply_chat_template(
                        example["prompt"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for example in examples
                ]
                images = [example["image"] for example in examples]
                inputs = processor(
                    images=images,
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                )
                input_width = inputs["input_ids"].shape[1]
                inputs = _move_inputs_to_device(inputs, device, dtype)

                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": self.caption_eval_max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if _world_size() > 1:
                    generation_kwargs["synced_gpus"] = True

                with torch.no_grad():
                    generated = model.generate(**inputs, **generation_kwargs)

                for example, sequence in zip(examples, generated, strict=True):
                    new_tokens = sequence[input_width:].detach().cpu()
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    predictions.append(_clean_caption(text))
                    references.append(_reference_caption(example))
        finally:
            if was_training:
                model.train()

        chrf_pp = CHRF(word_order=2).corpus_score(predictions, [references]).score
        return {
            f"{metric_key_prefix}_chrf_pp": chrf_pp,
        }


def _resolve_dtype_from_args(args: Any, model: Any):
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _processor_load_path(model_name_or_path: str, model_load_path: str) -> str:
    model_path = Path(model_name_or_path)
    if (model_path / "processor_config.json").exists():
        return model_name_or_path
    return model_load_path


def _load_processor(model_name_or_path: str, model_load_path: str, trust_remote_code: bool):
    processor = AutoProcessor.from_pretrained(
        _processor_load_path(model_name_or_path, model_load_path),
        trust_remote_code=trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Aya Vision processor did not expose a tokenizer.")
    tokenizer = ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer
    return processor


def _configure_deepspeed_buckets(cfg: LoraSFTTrainConfig, training_args: Any, model_load_path: str) -> Any:
    if not getattr(training_args, "deepspeed", None):
        return None

    try:
        import deepspeed  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DeepSpeed is not installed in the current environment."
        ) from exc

    deepspeed_init = getattr(training_args, "hf_deepspeed_config", None)
    if deepspeed_init is None:
        raise ValueError(
            "TrainingArguments did not create `hf_deepspeed_config` for DeepSpeed."
        )

    model_config = AutoConfig.from_pretrained(
        model_load_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    hidden_size = None
    if hasattr(model_config, "hidden_size"):
        hidden_size = model_config.hidden_size
    elif hasattr(model_config, "hidden_sizes"):
        hidden_size = max(model_config.hidden_sizes)
    elif hasattr(model_config, "text_config") and hasattr(
        model_config.text_config, "hidden_size"
    ):
        hidden_size = model_config.text_config.hidden_size
    elif hasattr(model_config, "text_config") and hasattr(
        model_config.text_config, "hidden_sizes"
    ):
        hidden_size = max(model_config.text_config.hidden_sizes)

    if hidden_size is not None:
        deepspeed_init.fill_only(
            "zero_optimization.reduce_bucket_size",
            hidden_size * hidden_size,
        )
        if deepspeed_init.is_zero3():
            deepspeed_init.fill_only(
                "zero_optimization.stage3_prefetch_bucket_size",
                int(0.9 * hidden_size * hidden_size),
            )
            deepspeed_init.fill_only(
                "zero_optimization.stage3_param_persistence_threshold",
                10 * hidden_size,
            )
    return deepspeed_init


def _resolve_dtype(cfg: LoraSFTTrainConfig) -> torch.dtype | None:
    if cfg.bf16:
        return torch.bfloat16
    if cfg.fp16:
        return torch.float16
    return None


def _load_model(cfg: LoraSFTTrainConfig, model_load_path: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_load_path,
        trust_remote_code=cfg.trust_remote_code,
        dtype=_resolve_dtype(cfg),
    )

    adapter_base_model_name = _load_adapter_base_model_name(cfg.model_name_or_path)
    if adapter_base_model_name is not None:
        _assert_adapter_has_weights(cfg.model_name_or_path)
        model = PeftModel.from_pretrained(
            model,
            cfg.model_name_or_path,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.lora_target_modules,
            modules_to_save=cfg.modules_to_save,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def _resolve_model_load_path(model_name_or_path: str) -> str:
    return _load_adapter_base_model_name(model_name_or_path) or model_name_or_path


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)
    set_seed(cfg.seed)

    dataset = _prepare_captioning_dataset(
        dataset_path=args.dataset_path,
        split=args.train_split,
        languages=args.languages,
        prompt_template=args.prompt_template,
    )
    train_dataset, eval_dataset, split_info = _split_train_eval_dataset(
        dataset,
        eval_percentage=args.eval_percentage,
        seed=cfg.seed,
    )
    if eval_dataset is not None:
        cfg.eval_steps = _steps_per_epoch(
            num_train_examples=len(train_dataset),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )
        cfg.save_steps = cfg.eval_steps
    else:
        cfg.early_stopping_patience = None
        cfg.early_stopping_threshold = None

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **pretty_config(cfg),
        "captioning_languages": args.languages,
        "prompt_template": args.prompt_template,
        "num_captioning_examples": len(dataset),
        "num_train_examples": len(train_dataset),
        "num_eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "captioning_eval_metric": "chrf_pp" if eval_dataset is not None else None,
        "runtime_eval_split": split_info,
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    print(
        "Captioning LoRA SFT plan: "
        + json.dumps(
            {
                "model_name_or_path": cfg.model_name_or_path,
                "dataset_path": args.dataset_path,
                "train_split": args.train_split,
                "languages": args.languages,
                "num_captioning_examples": len(dataset),
                "num_train_examples": len(train_dataset),
                "num_eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
                "eval_steps": cfg.eval_steps if eval_dataset is not None else None,
                "eval_metric": "chrf_pp" if eval_dataset is not None else None,
                "runtime_eval_split": split_info,
                "output_dir": cfg.output_dir,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    has_eval = eval_dataset is not None
    training_args = build_sft_args(cfg, has_eval=has_eval)
    if has_eval:
        training_args.metric_for_best_model = "eval_chrf_pp"
        training_args.greater_is_better = True

    model_load_path = _resolve_model_load_path(cfg.model_name_or_path)
    processor = _load_processor(
        cfg.model_name_or_path,
        model_load_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    deepspeed_init = _configure_deepspeed_buckets(cfg, training_args, model_load_path)

    model = _load_model(cfg, model_load_path)
    if cfg.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    train_sampler = build_train_sampler(
        train_dataset,
        seed=cfg.seed,
        alpha=cfg.language_sampling_alpha,
        packing=cfg.packing,
        world_size=getattr(training_args, "world_size", 1),
        rank=getattr(training_args, "process_index", 0),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": processor,
        "train_sampler": train_sampler,
    }
    callbacks = build_callbacks(cfg, has_eval=has_eval)
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = CaptioningChrFTrainer(
        **trainer_kwargs,
        caption_eval_dataset=eval_dataset,
    )
    _ = deepspeed_init

    resume_checkpoint = _resolve_resume_checkpoint(output_dir)
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_checkpoint:
        if trainer.args.should_save:
            print(f"Exporting best checkpoint from {best_checkpoint} to {cfg.output_dir}.")
            _export_best_checkpoint(Path(best_checkpoint), output_dir)
    else:
        trainer.save_model(cfg.output_dir)

    if trainer.args.should_save:
        processor.save_pretrained(cfg.output_dir)
        _validate_saved_adapter(output_dir)


if __name__ == "__main__":
    main()
