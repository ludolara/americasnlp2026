from __future__ import annotations

import argparse
import copy
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from sacrebleu.metrics import CHRF
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import (
    FSDP,
    is_conversational,
    profiling_context,
    unwrap_model_for_generation,
)

from train.config import load_grpo_config, pretty_config
from train.data import load_datasets, prepare_grpo_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO on the translation model with chrF++ reward"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny_aya_grpo.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


def _build_grpo_args(cfg, has_eval: bool) -> GRPOConfig:
    params = inspect.signature(GRPOConfig.__init__).parameters
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
        "remove_unused_columns": False,
        "max_completion_length": cfg.max_completion_length,
        "num_generations": cfg.num_generations,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "beta": cfg.beta,
    }

    filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return GRPOConfig(**filtered)


def _build_callbacks(cfg, has_eval: bool):
    if cfg.early_stopping_patience is None:
        return []
    if not has_eval:
        raise ValueError("early_stopping_patience requires an eval split or eval file")
    if cfg.save_steps % cfg.eval_steps != 0:
        raise ValueError(
            "early_stopping_patience requires save_steps to be a multiple of eval_steps"
        )
    return [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]


class PatchedGRPOTrainer(GRPOTrainer):
    def _generate_single_turn(self, prompts: list):
        if self.use_vllm or self.use_transformers_paged or is_conversational({"prompt": prompts[0]}):
            return super()._generate_single_turn(prompts)

        device = self.accelerator.device
        generate_inputs = self.processing_class(
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.disable_compile = True

        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if self.is_fsdp_enabled
            else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=generation_config,
            )

        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        return prompt_ids, completion_ids, None, {}


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    parts.extend(str(part.get("text", "")).strip() for part in content if isinstance(part, dict))
                else:
                    parts.append(str(content).strip())
            else:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(completion).strip()


def extract_assistant_response(text: str) -> str:
    cleaned = text.strip()
    if "<|assistant|>" in cleaned:
        cleaned = cleaned.rsplit("<|assistant|>", maxsplit=1)[-1].strip()
    if "<|user|>" in cleaned:
        cleaned = cleaned.split("<|user|>", maxsplit=1)[0].strip()
    return cleaned


def build_chrf_reward():
    metric = CHRF(word_order=2)

    def reward_func(*, completions: list[Any], reference: list[str], **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion, target in zip(completions, reference, strict=True):
            prediction = extract_assistant_response(_completion_to_text(completion))
            score = metric.sentence_score(prediction, [str(target).strip()]).score
            # Keep the reward on a 0..1 scale while preserving chrF++ ordering.
            rewards.append(score / 100.0)
        return rewards

    return reward_func


def main() -> None:
    args = parse_args()
    cfg = load_grpo_config(args.config)

    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    train_dataset, eval_dataset = prepare_grpo_splits(raw=raw, config=cfg)
    has_eval = eval_dataset is not None
    training_args = _build_grpo_args(cfg, has_eval=has_eval)
    callbacks = _build_callbacks(cfg, has_eval=has_eval)

    trainer = PatchedGRPOTrainer(
        model=model,
        reward_funcs=build_chrf_reward(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "trainer": "grpo",
                "reward_metric": "chrF++",
                **pretty_config(cfg),
            },
            f,
            indent=2,
        )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
