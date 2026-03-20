from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
from torch.utils.data import Sampler

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None  # type: ignore[assignment]

from train.config import load_sft_config, pretty_config
from train.data import ensure_chat_template, load_datasets, prepare_sft_splits
from train.language_sampling import (
    DistributedTemperatureSmoothedLanguageSampler,
    TemperatureSmoothedLanguageSampler,
    compute_smoothed_mix,
    count_languages,
    extract_languages,
    format_weighted_mix,
)


class MultilingualSFTTrainer(SFTTrainer):
    def __init__(self, *args, train_sampler: Sampler[int] | None = None, **kwargs):
        self._train_sampler_override = train_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset: Dataset | None = None):
        if self._train_sampler_override is not None:
            return self._train_sampler_override
        return super()._get_train_sampler(train_dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning (SFT) for the translation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


def _build_sft_args(cfg, has_eval: bool):
    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    params = inspect.signature(args_cls.__init__).parameters

    strategy = "steps" if has_eval else "no"
    load_best_model = has_eval and cfg.early_stopping_patience is not None
    kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps if has_eval else None,
        "evaluation_strategy": strategy,
        "eval_strategy": strategy,
        "save_strategy": "steps",
        "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": "eval_loss" if load_best_model else None,
        "greater_is_better": False if load_best_model else None,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "report_to": cfg.report_to,
        "seed": cfg.seed,
        "dataset_text_field": "text",
        "max_seq_length": cfg.max_seq_length,
        "max_length": cfg.max_seq_length,
        "packing": cfg.packing,
    }

    filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return args_cls(**filtered)


def _build_callbacks(cfg, has_eval: bool):
    if cfg.early_stopping_patience is None:
        return []
    if not has_eval:
        raise ValueError("early_stopping_patience requires an eval split or eval file")
    if cfg.save_steps % cfg.eval_steps != 0:
        raise ValueError(
            "early_stopping_patience requires save_steps to be a multiple of eval_steps"
        )
    callback_kwargs = {
        "early_stopping_patience": cfg.early_stopping_patience,
    }
    if cfg.early_stopping_threshold is not None:
        callback_kwargs["early_stopping_threshold"] = cfg.early_stopping_threshold
    return [EarlyStoppingCallback(**callback_kwargs)]


def _build_train_sampler(
    dataset: Dataset,
    *,
    seed: int,
    alpha: float,
    packing: bool,
    world_size: int,
    rank: int,
) -> Sampler[int] | None:
    if "language" not in dataset.column_names:
        return None

    counts = count_languages(dataset)
    print(f"Train language mix (raw): {format_weighted_mix(counts)}")

    if packing:
        print("Skipping alpha-smoothed sampler because packing=True changes dataset length.")
        return None

    if alpha == 1.0:
        print("Using default shuffled sampling because language_sampling_alpha=1.0.")
        return None

    languages = extract_languages(dataset)
    if len(counts) < 2:
        return None

    smoothed_mix = compute_smoothed_mix(counts, alpha)
    print(
        f"Train language mix (alpha={alpha:.2f}): "
        f"{format_weighted_mix(smoothed_mix)}"
    )

    if world_size != 1:
        print(
            f"Using distributed alpha-smoothed sampler "
            f"(alpha={alpha:.2f}, world_size={world_size}, rank={rank})."
        )
        return DistributedTemperatureSmoothedLanguageSampler(
            languages=languages,
            seed=seed,
            alpha=alpha,
            epoch_size=len(languages),
            num_replicas=world_size,
            rank=rank,
        )

    return TemperatureSmoothedLanguageSampler(
        languages=languages,
        seed=seed,
        alpha=alpha,
        epoch_size=len(languages),
    )


def main() -> None:
    args = parse_args()
    cfg = load_sft_config(args.config)

    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    tokenizer = ensure_chat_template(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = None
    if cfg.bf16:
        model_dtype = torch.bfloat16
    elif cfg.fp16:
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        dtype=model_dtype,
    )

    if cfg.gradient_checkpointing:
        model.config.use_cache = False

    raw = load_datasets(cfg)
    train_dataset, eval_dataset = prepare_sft_splits(
        raw=raw,
        config=cfg,
        eos_token=tokenizer.eos_token or "",
        tokenizer=tokenizer,
    )

    has_eval = eval_dataset is not None
    training_args = _build_sft_args(cfg, has_eval=has_eval)
    train_sampler = _build_train_sampler(
        raw[cfg.train_split],
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
        "train_sampler": train_sampler,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    callbacks = _build_callbacks(cfg, has_eval=has_eval)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "callbacks" in trainer_params and callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = MultilingualSFTTrainer(**trainer_kwargs)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(pretty_config(cfg), f, indent=2)

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
