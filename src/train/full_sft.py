from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer

from train.config import load_full_sft_config, pretty_config
from train.data import ensure_chat_template, load_datasets, prepare_sft_splits
from train.sft_common import (
    MultilingualSFTTrainer,
    build_callbacks,
    build_sft_args,
    build_train_sampler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning (SFT) for the text Tiny Aya model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_sft.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_full_sft_config(args.config)

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
    training_args = build_sft_args(cfg, has_eval=has_eval)
    train_sampler = build_train_sampler(
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
    callbacks = build_callbacks(cfg, has_eval=has_eval)
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
