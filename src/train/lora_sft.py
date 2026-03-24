from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, set_seed
from trl import SFTTrainer

from train.config import load_lora_sft_config, pretty_config
from train.data import ensure_chat_template, load_datasets, prepare_sft_splits
from train.sft_common import (
    MultilingualSFTTrainer,
    build_callbacks,
    build_sft_args,
    build_train_sampler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT for the Aya Vision model using text-only examples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_sft.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_lora_sft_config(args.config)

    set_seed(cfg.seed)

    processor = AutoProcessor.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Vision processor does not expose a tokenizer for text-only LoRA SFT."
        )
    tokenizer = ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer

    raw = load_datasets(cfg)
    train_dataset, eval_dataset = prepare_sft_splits(
        raw=raw,
        config=cfg,
        eos_token=tokenizer.eos_token or "",
        tokenizer=tokenizer,
    )

    has_eval = eval_dataset is not None
    training_args = build_sft_args(cfg, has_eval=has_eval)
    deepspeed_init = None
    if getattr(training_args, "deepspeed", None):
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
            cfg.model_name_or_path,
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

    model_dtype = None
    if cfg.bf16:
        model_dtype = torch.bfloat16
    elif cfg.fp16:
        model_dtype = torch.float16

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        dtype=model_dtype,
    )
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

    if cfg.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

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
    _ = deepspeed_init

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(pretty_config(cfg), f, indent=2)

    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
