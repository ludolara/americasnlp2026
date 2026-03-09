from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
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

    output_dir: str = "outputs/tiny-aya-full-sft"
    max_seq_length: int = 2048
    packing: bool = False

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    warmup_steps: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    early_stopping_patience: int | None = None

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    seed: int = 42
    report_to: str = "none"


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "warmup_ratio" in raw:
        raw["warmup_steps"] = raw.pop("warmup_ratio")

    valid_fields = set(TrainConfig.__dataclass_fields__.keys())
    unknown = sorted(set(raw.keys()) - valid_fields)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown config keys in {config_path}: {unknown_str}")

    patience = raw.get("early_stopping_patience")
    if patience is not None and patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 when set")

    return TrainConfig(**raw)


def pretty_config(config: TrainConfig) -> dict[str, Any]:
    return {k: getattr(config, k) for k in TrainConfig.__dataclass_fields__.keys()}
