from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import yaml


@dataclass
class DatasetConfigMixin:
    model_name_or_path: str = "models/tiny-aya-base"
    trust_remote_code: bool = True

    dataset_name: str | None = None
    dataset_config_name: str | None = None
    local_dataset_path: str | None = None
    train_split: str = "train"
    eval_split: str | None = "validation"
    train_file: str | None = None
    eval_file: str | None = None
    text_column: str | None = None

    source_column: str = "es"
    target_column: str = "wix"

    seed: int = 42
    report_to: str = "none"


@dataclass
class SFTTrainConfig(DatasetConfigMixin):
    output_dir: str = "outputs/tiny-aya-full-sft"
    max_seq_length: int = 2048
    packing: bool = False
    language_sampling_alpha: float = 1.0

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    warmup_steps: float | int | None = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    early_stopping_patience: int | None = None

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True


@dataclass
class GRPOTrainConfig(DatasetConfigMixin):
    output_dir: str = "outputs/tiny-aya-full-grpo"
    max_completion_length: int = 2048
    num_generations: int = 4
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 0
    beta: float = 0.0
    language_sampling_alpha: float = 1.0

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    num_train_epochs: float = 3.0
    warmup_steps: float | int | None = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"

    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2
    early_stopping_patience: int | None = None

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True


ConfigT = TypeVar("ConfigT", SFTTrainConfig, GRPOTrainConfig)


def _normalize_warmup_fields(raw: dict) -> dict:
    normalized = dict(raw)
    if normalized.get("warmup_steps") is not None and normalized.get("warmup_ratio") is not None:
        raise ValueError("Set only one of `warmup_steps` or `warmup_ratio`.")

    if normalized.get("warmup_ratio") is not None:
        normalized["warmup_steps"] = normalized.pop("warmup_ratio")

    warmup_steps = normalized.get("warmup_steps")
    if isinstance(warmup_steps, float) and warmup_steps.is_integer():
        normalized["warmup_steps"] = int(warmup_steps)

    return normalized


def _load_config(path: str | Path, config_cls: type[ConfigT]) -> ConfigT:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _normalize_warmup_fields(raw)

    valid_fields = set(config_cls.__dataclass_fields__.keys())
    unknown = sorted(set(raw.keys()) - valid_fields)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown config keys in {config_path}: {unknown_str}")

    patience = raw.get("early_stopping_patience")
    if patience is not None and patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 when set")

    language_sampling_alpha = raw.get("language_sampling_alpha")
    if language_sampling_alpha is not None and language_sampling_alpha < 0:
        raise ValueError("language_sampling_alpha must be >= 0.")

    return config_cls(**raw)


def load_sft_config(path: str | Path) -> SFTTrainConfig:
    return _load_config(path, SFTTrainConfig)


def load_grpo_config(path: str | Path) -> GRPOTrainConfig:
    return _load_config(path, GRPOTrainConfig)


def pretty_config(config: SFTTrainConfig | GRPOTrainConfig) -> dict:
    return asdict(config)
