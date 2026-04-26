from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from collections import Counter, OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Sampler
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig

from train.config import (
    GRPOTrainConfig,
    _normalize_warmup_fields,
    _validate_config_values,
)
from train.data import ensure_chat_template, load_datasets, prepare_grpo_splits
from train.grpo import (
    PatchedGRPOTrainer,
    _build_callbacks,
    _build_train_sampler,
    build_chrf_reward,
)
from train.language_sampling import format_weighted_mix, sample_dataset_by_language
from train.sft_common import (
    _configure_deepspeed_scheduler,
    ensure_cuda_home,
    prime_deepspeed_for_model_loading,
)


@dataclass
class LoraGRPOTrainConfig(GRPOTrainConfig):
    model_name_or_path: str = "outputs/aya-vision-32b-americas"
    output_dir: str = "outputs/aya-vision-32b-americas-lora-grpo"
    eval_sample_size: int | None = None
    eval_language_sample_weights: dict[str, float] | None = None

    save_only_model: bool = True
    deepspeed: str | None = None
    ddp_find_unused_parameters: bool | None = None
    gradient_checkpointing_kwargs: dict[str, Any] | None = None

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA GRPO from an Aya Vision LoRA SFT adapter."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_grpo.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


def load_lora_grpo_config(path: str | Path) -> LoraGRPOTrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _normalize_warmup_fields(raw)

    valid_fields = set(LoraGRPOTrainConfig.__dataclass_fields__.keys())
    unknown = sorted(set(raw.keys()) - valid_fields)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown config keys in {config_path}: {unknown_str}")

    _validate_config_values(raw)

    return LoraGRPOTrainConfig(**raw)


def _resolve_resume_checkpoint(cfg: LoraGRPOTrainConfig) -> str | None:
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(cfg.output_dir)

    if cfg.resume_from_checkpoint:
        checkpoint_path = Path(cfg.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Configured resume checkpoint does not exist: {checkpoint_path}"
            )
        if rank == 0:
            print(f"Resuming LoRA GRPO from configured checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    if not cfg.auto_resume_from_checkpoint:
        if rank == 0:
            print("LoRA GRPO auto-resume disabled in config; starting fresh.")
        return None

    if not output_dir.exists():
        if rank == 0:
            print(
                f"Output dir {output_dir} does not exist yet; starting LoRA GRPO from scratch."
            )
        return None

    checkpoint = get_last_checkpoint(str(output_dir))
    if rank == 0:
        if checkpoint is None:
            print(f"No LoRA GRPO checkpoint found in {output_dir}; starting fresh.")
        else:
            print(f"Resuming LoRA GRPO from latest checkpoint: {checkpoint}")
    return checkpoint


def _build_lora_grpo_args(cfg: LoraGRPOTrainConfig, has_eval: bool) -> GRPOConfig:
    params = inspect.signature(GRPOConfig.__init__).parameters
    strategy = "steps" if has_eval else "no"
    track_best_metric = has_eval and cfg.early_stopping_patience is not None
    load_best_model = track_best_metric and not cfg.save_only_model
    gradient_checkpointing_kwargs = cfg.gradient_checkpointing_kwargs
    if (
        cfg.gradient_checkpointing
        and cfg.deepspeed
        and gradient_checkpointing_kwargs is None
    ):
        # TRL 0.29 defaults GRPO to non-reentrant checkpointing. With ZeRO-3,
        # non-reentrant recompute can see partitioned DeepSpeed weights as
        # zero-length tensors and fail PyTorch's metadata check during backward.
        gradient_checkpointing_kwargs = {"use_reentrant": True}
    kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "warmup_steps": cfg.warmup_steps,
        "warmup_ratio": getattr(cfg, "warmup_ratio", None),
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps if has_eval else None,
        "evaluation_strategy": strategy,
        "eval_strategy": strategy,
        "save_strategy": "steps",
        "save_total_limit": cfg.save_total_limit,
        "save_only_model": cfg.save_only_model,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": "eval_reward" if track_best_metric else None,
        "greater_is_better": True if track_best_metric else None,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "deepspeed": cfg.deepspeed,
        "ddp_find_unused_parameters": cfg.ddp_find_unused_parameters,
        "report_to": cfg.report_to,
        "seed": cfg.seed,
        "remove_unused_columns": False,
        "max_completion_length": cfg.max_completion_length,
        "num_generations": cfg.num_generations,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "beta": cfg.beta,
        "ds3_gather_for_generation": cfg.ds3_gather_for_generation,
    }

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in params and value is not None
    }
    if cfg.deepspeed:
        ensure_cuda_home()

    args = GRPOConfig(**filtered)
    prime_deepspeed_for_model_loading(args)
    _configure_deepspeed_scheduler(args)
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
        f"Adapter at {adapter_path} contains zero tensors. Refusing to continue "
        "training from an empty LoRA export."
    )


def _validate_saved_adapter(output_dir: Path) -> None:
    adapter_path = output_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise RuntimeError(f"Missing saved LoRA adapter: {adapter_path}")

    from safetensors import safe_open

    with safe_open(adapter_path, framework="pt") as f:
        if any(True for _ in f.keys()):
            return

    raise RuntimeError(
        "Saved LoRA adapter is empty. Training appears to have completed, but the "
        "exported adapter contains zero tensors."
    )


def _export_best_checkpoint(best_checkpoint: Path, output_dir: Path) -> None:
    artifacts = (
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
    )

    for artifact in artifacts:
        source = best_checkpoint / artifact
        if source.exists():
            shutil.copy2(source, output_dir / artifact)


def _resolve_model_load_path(model_name_or_path: str) -> str:
    return _load_adapter_base_model_name(model_name_or_path) or model_name_or_path


def _tokenizer_load_path(model_name_or_path: str, model_load_path: str) -> str:
    model_path = Path(model_name_or_path)
    if (model_path / "tokenizer_config.json").exists():
        return model_name_or_path
    return model_load_path


def _resolve_dtype(cfg: LoraGRPOTrainConfig) -> torch.dtype | None:
    if cfg.bf16:
        return torch.bfloat16
    if cfg.fp16:
        return torch.float16
    return None


def _load_tokenizer(cfg: LoraGRPOTrainConfig, model_load_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_load_path(cfg.model_name_or_path, model_load_path),
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
        padding_side="left",
    )
    tokenizer = ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _configure_deepspeed_buckets(
    cfg: LoraGRPOTrainConfig,
    training_args: GRPOConfig,
    model_load_path: str,
) -> Any:
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


def _load_model(cfg: LoraGRPOTrainConfig, model_load_path: str):
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

    if cfg.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    return model


class LoraGRPOTrainer(PatchedGRPOTrainer):
    def __init__(
        self,
        *args,
        train_sampler: Sampler[int] | None = None,
        eval_sample_size: int | None = None,
        eval_language_sample_weights: dict[str, float] | None = None,
        eval_language_column: str = "dataset_language",
        **kwargs,
    ):
        self.eval_sample_size = eval_sample_size
        self.eval_language_sample_weights = eval_language_sample_weights
        self.eval_language_column = eval_language_column
        super().__init__(*args, train_sampler=train_sampler, **kwargs)

    def _get_train_sampler(self, dataset: Dataset | None = None):
        return super()._get_train_sampler(dataset)

    def _sample_eval_dataset(self, dataset: Dataset, metric_key_prefix: str) -> Dataset:
        if self.eval_sample_size is None:
            return dataset

        global_step = int(getattr(self.state, "global_step", 0) or 0)
        seed = int(getattr(self.args, "seed", 0) or 0)
        sampled_dataset = sample_dataset_by_language(
            dataset,
            sample_size=self.eval_sample_size,
            seed=seed,
            language_weights=self.eval_language_sample_weights,
            language_column=self.eval_language_column,
        )

        if self.is_world_process_zero():
            counts = Counter(
                str(language).strip()
                for language in sampled_dataset[self.eval_language_column]
            )
            mix = format_weighted_mix(counts)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            suffix = f" ({mix})" if mix else ""
            print(
                f"[lora-grpo {timestamp}] Eval sample for {metric_key_prefix} "
                f"at step {global_step}: {len(sampled_dataset)} rows{suffix}.",
                flush=True,
            )

        return sampled_dataset

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        selected_eval_dataset = (
            self.eval_dataset if eval_dataset is None else eval_dataset
        )

        if isinstance(selected_eval_dataset, Dataset):
            selected_eval_dataset = self._sample_eval_dataset(
                selected_eval_dataset,
                metric_key_prefix,
            )
        elif isinstance(selected_eval_dataset, dict):
            selected_eval_dataset = {
                name: self._sample_eval_dataset(
                    dataset,
                    f"{metric_key_prefix}_{name}",
                )
                for name, dataset in selected_eval_dataset.items()
            }

        return super().evaluate(
            eval_dataset=selected_eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def _can_save_zero3_adapter(self) -> bool:
        if not getattr(self, "is_deepspeed_enabled", False):
            return False

        deepspeed_plugin = getattr(getattr(self, "accelerator", None), "state", None)
        deepspeed_plugin = getattr(deepspeed_plugin, "deepspeed_plugin", None)
        if getattr(deepspeed_plugin, "zero_stage", None) != 3:
            return False

        return hasattr(self.model, "peft_config") and hasattr(
            self.model,
            "save_pretrained",
        )

    def _gather_zero3_trainable_state_dict(
        self,
    ) -> OrderedDict[str, torch.Tensor] | None:
        try:
            import deepspeed
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DeepSpeed is required to gather ZeRO-3 adapter weights."
            ) from exc

        state_dict: OrderedDict[str, torch.Tensor] | None = None
        if self.args.should_save:
            state_dict = OrderedDict()

        zero3_engine = getattr(self, "deepspeed", None)
        if (
            zero3_engine is None
            and getattr(self, "model_wrapped", None) is not self.model
        ):
            zero3_engine = getattr(self, "model_wrapped", None)

        def add_module_state(module: torch.nn.Module, prefix: str = "") -> None:
            module_trainable_params = [
                param for param in module.parameters(recurse=False) if param.requires_grad
            ]
            gather_ctx = (
                deepspeed.zero.GatheredParameters(
                    module_trainable_params,
                    modifier_rank=0,
                )
                if module_trainable_params
                else nullcontext()
            )

            with gather_ctx:
                if state_dict is not None:
                    for name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            state_dict[prefix + name] = param.detach().cpu().clone()

            for name, child in module.named_children():
                if child is not None:
                    add_module_state(child, prefix + name + ".")

        if (
            zero3_engine is not None
            and hasattr(zero3_engine, "_optimizer_has_ckpt_event_prologue")
            and zero3_engine._optimizer_has_ckpt_event_prologue()
        ):
            zero3_engine.optimizer.checkpoint_event_prologue()

        try:
            add_module_state(self.model)
        finally:
            if (
                zero3_engine is not None
                and hasattr(zero3_engine, "_optimizer_has_ckpt_event_epilogue")
                and zero3_engine._optimizer_has_ckpt_event_epilogue()
            ):
                zero3_engine.optimizer.checkpoint_event_epilogue()

        return state_dict

    def _save_zero3_adapter(self, output_dir: str | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        state_dict = self._gather_zero3_trainable_state_dict()

        if self.args.should_save:
            if not state_dict or not any("lora_" in key for key in state_dict):
                raise RuntimeError(
                    "DeepSpeed ZeRO-3 adapter export gathered no LoRA tensors."
                )

            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            elif (
                self.data_collator is not None
                and hasattr(self.data_collator, "tokenizer")
                and self.data_collator.tokenizer is not None
            ):
                self.data_collator.tokenizer.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def save_lora_adapter(self, output_dir: str | None = None) -> None:
        if not self._can_save_zero3_adapter():
            self.save_model(output_dir, _internal_call=True)
            return

        self._save_zero3_adapter(output_dir)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not self._can_save_zero3_adapter():
            return super().save_model(output_dir, _internal_call=_internal_call)

        if not getattr(self.args, "save_only_model", False):
            result = super().save_model(output_dir, _internal_call=True)
            self._save_zero3_adapter(output_dir)
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(
                    commit_message="Model save",
                    revision=self.args.hub_revision,
                )
            return result

        self._save_zero3_adapter(output_dir)

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(
                commit_message="Model save",
                revision=self.args.hub_revision,
            )


class LoraGRPOProgressCallback(TrainerCallback):
    def __init__(self, *, has_eval: bool):
        self.has_eval = has_eval

    @staticmethod
    def _is_world_process_zero(args, state) -> bool:
        is_world_process_zero = getattr(state, "is_world_process_zero", None)
        if is_world_process_zero is not None:
            return bool(is_world_process_zero)

        process_index = getattr(args, "process_index", None)
        if process_index is not None:
            return process_index == 0

        return int(os.environ.get("RANK", "0")) == 0

    def _print(self, args, state, message: str) -> None:
        if not self._is_world_process_zero(args, state):
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[lora-grpo {timestamp}] {message}", flush=True)

    @staticmethod
    def _format_value(value: Any) -> str | None:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.6g}"
        return None

    def _format_metrics(self, metrics: dict[str, Any] | None) -> str:
        if not metrics:
            return ""

        preferred_keys = (
            "loss",
            "eval_loss",
            "reward",
            "eval_reward",
            "rewards/mean",
            "eval_rewards/mean",
            "learning_rate",
            "grad_norm",
            "epoch",
            "train_runtime",
            "eval_runtime",
        )
        parts: list[str] = []
        seen: set[str] = set()

        for key in preferred_keys:
            if key not in metrics:
                continue
            value = self._format_value(metrics[key])
            if value is not None:
                parts.append(f"{key}={value}")
                seen.add(key)

        for key in sorted(metrics):
            if key in seen or key == "total_flos":
                continue
            value = self._format_value(metrics[key])
            if value is not None:
                parts.append(f"{key}={value}")

        return ", ".join(parts)

    def on_train_begin(self, args, state, control, **kwargs):
        eval_strategy = getattr(
            args,
            "eval_strategy",
            getattr(args, "evaluation_strategy", "no"),
        )
        eval_status = (
            f"eval_steps={args.eval_steps}"
            if self.has_eval and str(eval_strategy) != "no"
            else "eval disabled"
        )
        self._print(
            args,
            state,
            "Training started: "
            f"max_steps={state.max_steps}, "
            f"logging_steps={args.logging_steps}, "
            f"save_steps={args.save_steps}, "
            f"{eval_status}, "
            f"output_dir={args.output_dir}",
        )

    def on_step_end(self, args, state, control, **kwargs):
        scheduled = []
        if control.should_log:
            scheduled.append("log")
        if control.should_evaluate:
            scheduled.append("eval")
        if control.should_save:
            scheduled.append("save")

        if scheduled:
            self._print(
                args,
                state,
                f"Step {state.global_step}: scheduled {' + '.join(scheduled)}.",
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics = self._format_metrics(logs)
        suffix = f": {metrics}" if metrics else "."
        self._print(args, state, f"Logged step {state.global_step}{suffix}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        details = []
        formatted_metrics = self._format_metrics(metrics)
        if formatted_metrics:
            details.append(formatted_metrics)

        best_metric = getattr(state, "best_metric", None)
        best_metric_text = self._format_value(best_metric)
        if best_metric_text is not None:
            details.append(f"best_metric={best_metric_text}")

        best_checkpoint = getattr(state, "best_model_checkpoint", None)
        if best_checkpoint:
            details.append(f"best_checkpoint={best_checkpoint}")

        suffix = f": {'; '.join(details)}" if details else "."
        self._print(args, state, f"Finished eval at step {state.global_step}{suffix}")

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self._print(
            args,
            state,
            f"Saved checkpoint at step {state.global_step}: {checkpoint_dir}",
        )

    def on_train_end(self, args, state, control, **kwargs):
        self._print(
            args,
            state,
            f"Training ended at step {state.global_step}; best_model_checkpoint="
            f"{getattr(state, 'best_model_checkpoint', None)}",
        )


def main() -> None:
    args = parse_args()
    cfg = load_lora_grpo_config(args.config)
    resume_checkpoint = _resolve_resume_checkpoint(cfg)

    set_seed(cfg.seed)

    model_load_path = _resolve_model_load_path(cfg.model_name_or_path)
    tokenizer = _load_tokenizer(cfg, model_load_path)

    raw = load_datasets(cfg)
    train_dataset, eval_dataset = prepare_grpo_splits(
        raw=raw,
        config=cfg,
        tokenizer=tokenizer,
    )
    has_eval = eval_dataset is not None
    training_args = _build_lora_grpo_args(cfg, has_eval=has_eval)
    deepspeed_init = _configure_deepspeed_buckets(cfg, training_args, model_load_path)

    model = _load_model(cfg, model_load_path)
    callbacks = _build_callbacks(cfg, has_eval=has_eval)
    callbacks.append(LoraGRPOProgressCallback(has_eval=has_eval))
    train_sampler = _build_train_sampler(
        raw[cfg.train_split],
        seed=cfg.seed,
        alpha=cfg.language_sampling_alpha,
        generation_batch_size=training_args.generation_batch_size,
        num_generations=training_args.num_generations,
        repeat_count=training_args.num_iterations * training_args.steps_per_generation,
        world_size=getattr(training_args, "world_size", 1),
    )

    trainer = LoraGRPOTrainer(
        model=model,
        reward_funcs=build_chrf_reward(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        train_sampler=train_sampler,
        eval_sample_size=cfg.eval_sample_size,
        eval_language_sample_weights=cfg.eval_language_sample_weights,
    )
    _ = deepspeed_init

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "trainer": "lora_grpo",
                "reward_metric": "chrF++",
                **asdict(cfg),
            },
            f,
            indent=2,
        )

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_checkpoint and cfg.save_only_model:
        if trainer.args.should_save:
            print(
                f"Exporting best checkpoint from {best_checkpoint} to {cfg.output_dir}."
            )
            _export_best_checkpoint(Path(best_checkpoint), output_dir)
    else:
        trainer.save_lora_adapter(cfg.output_dir)

    if trainer.args.should_save:
        tokenizer.save_pretrained(cfg.output_dir)
        _validate_saved_adapter(output_dir)


if __name__ == "__main__":
    main()
