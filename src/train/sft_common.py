from __future__ import annotations

from contextlib import nullcontext
import inspect
import os
import shutil
from collections import OrderedDict
from pathlib import Path

from datasets import Dataset
import torch
from torch.utils.data import Sampler
from transformers import EarlyStoppingCallback, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None  # type: ignore[assignment]

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

    def _should_use_zero3_adapter_only_save(self) -> bool:
        if not getattr(self, "is_deepspeed_enabled", False):
            return False
        if not getattr(self.args, "save_only_model", False):
            return False

        deepspeed_plugin = getattr(getattr(self, "accelerator", None), "state", None)
        deepspeed_plugin = getattr(deepspeed_plugin, "deepspeed_plugin", None)
        if getattr(deepspeed_plugin, "zero_stage", None) != 3:
            return False

        return hasattr(self.model, "peft_config") and hasattr(
            self.model, "save_pretrained"
        )

    def _gather_zero3_trainable_state_dict(self) -> OrderedDict[str, torch.Tensor] | None:
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
        if zero3_engine is None and getattr(self, "model_wrapped", None) is not self.model:
            zero3_engine = getattr(self, "model_wrapped", None)

        def add_module_state(module: torch.nn.Module, prefix: str = "") -> None:
            module_trainable_params = [
                param for param in module.parameters(recurse=False) if param.requires_grad
            ]
            gather_ctx = (
                deepspeed.zero.GatheredParameters(
                    module_trainable_params,
                    # Mirror DeepSpeed's own ZeRO-3 consolidation path by
                    # gathering one module at a time and forcing release from rank 0.
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

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not self._should_use_zero3_adapter_only_save():
            return super().save_model(output_dir, _internal_call=_internal_call)

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        state_dict = self._gather_zero3_trainable_state_dict()

        if self.args.should_save:
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

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)


def _discover_cuda_home() -> Path | None:
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        candidate = os.environ.get(env_var)
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return candidate_path

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent

    for candidate in (
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
        "/usr/lib/cuda",
        "/opt/cuda",
    ):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return None


def ensure_cuda_home() -> str | None:
    cuda_home = _discover_cuda_home()
    if cuda_home is None:
        return None

    resolved = str(cuda_home)
    os.environ["CUDA_HOME"] = resolved
    os.environ["CUDA_PATH"] = resolved

    cuda_bin = str(cuda_home / "bin")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if (cuda_home / "bin").exists() and cuda_bin not in path_entries:
        os.environ["PATH"] = (
            f"{cuda_bin}{os.pathsep}{os.environ['PATH']}"
            if os.environ.get("PATH")
            else cuda_bin
        )

    cuda_lib64 = str(cuda_home / "lib64")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_entries = ld_library_path.split(os.pathsep) if ld_library_path else []
    if (cuda_home / "lib64").exists() and cuda_lib64 not in ld_entries:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{cuda_lib64}{os.pathsep}{ld_library_path}"
            if ld_library_path
            else cuda_lib64
        )

    return resolved


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def prime_deepspeed_for_model_loading(training_args):
    deepspeed_init = getattr(training_args, "hf_deepspeed_config", None)
    if deepspeed_init is None:
        return None

    world_size = _env_int("WORLD_SIZE")
    if world_size is None:
        world_size = max(int(getattr(training_args, "world_size", 1) or 1), 1)

    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    train_batch_size = (
        world_size * per_device_train_batch_size * gradient_accumulation_steps
    )

    # ZeRO-3 can consult the DeepSpeed config during model construction, before Trainer
    # finalization has a chance to replace "auto" placeholders.
    deepspeed_init.fill_only(
        "train_micro_batch_size_per_gpu",
        per_device_train_batch_size,
    )
    deepspeed_init.fill_only(
        "gradient_accumulation_steps",
        gradient_accumulation_steps,
    )
    deepspeed_init.fill_only("train_batch_size", train_batch_size)

    max_grad_norm = getattr(training_args, "max_grad_norm", None)
    if max_grad_norm is not None:
        deepspeed_init.fill_only("gradient_clipping", max_grad_norm)

    fp16_enabled = bool(
        getattr(training_args, "fp16", False)
        or getattr(training_args, "fp16_full_eval", False)
    )
    bf16_enabled = bool(
        getattr(training_args, "bf16", False)
        or getattr(training_args, "bf16_full_eval", False)
    )
    deepspeed_init.fill_only("fp16.enabled", fp16_enabled)
    deepspeed_init.fill_only("bf16.enabled", bf16_enabled)

    return deepspeed_init


def _configure_deepspeed_scheduler(training_args) -> None:
    deepspeed_init = getattr(training_args, "hf_deepspeed_config", None)
    if deepspeed_init is None:
        return

    config = deepspeed_init.config
    if "scheduler" in config:
        return

    scheduler_type = getattr(training_args, "lr_scheduler_type", "linear")
    scheduler_type = getattr(scheduler_type, "value", scheduler_type)
    scheduler_type = str(scheduler_type).lower()
    learning_rate = float(training_args.learning_rate)

    if scheduler_type == "linear":
        config["scheduler"] = {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
                "warmup_type": "linear",
            },
        }
        return

    if scheduler_type == "cosine":
        config["scheduler"] = {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0.0,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
                "cos_min_ratio": 0.0,
                "warmup_type": "linear",
            },
        }
        return

    if scheduler_type == "constant":
        config["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": "auto",
                "warmup_type": "linear",
            },
        }
        return

    raise ValueError(
        "DeepSpeed training currently supports only linear, cosine, and constant "
        f"lr_scheduler_type values via this trainer, got {scheduler_type!r}."
    )


def build_sft_args(cfg, has_eval: bool):
    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    params = inspect.signature(args_cls.__init__).parameters

    strategy = "steps" if has_eval else "no"
    track_best_metric = has_eval and cfg.early_stopping_patience is not None
    save_only_model = getattr(cfg, "save_only_model", False)
    load_best_model = track_best_metric and not save_only_model
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
        "save_only_model": save_only_model,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": "eval_loss" if track_best_metric else None,
        "greater_is_better": False if track_best_metric else None,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "deepspeed": getattr(cfg, "deepspeed", None),
        "ddp_find_unused_parameters": getattr(
            cfg, "ddp_find_unused_parameters", None
        ),
        "report_to": cfg.report_to,
        "seed": cfg.seed,
        "dataset_text_field": "text",
        "max_seq_length": cfg.max_seq_length,
        "max_length": cfg.max_seq_length,
        "packing": cfg.packing,
    }

    filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
    if getattr(cfg, "deepspeed", None):
        ensure_cuda_home()

    args = args_cls(**filtered)
    prime_deepspeed_for_model_loading(args)
    _configure_deepspeed_scheduler(args)
    return args


def build_callbacks(cfg, has_eval: bool):
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


def build_train_sampler(
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


def resolve_model_dtype(cfg):
    if cfg.bf16:
        return "bfloat16"
    if cfg.fp16:
        return "float16"
    return None
