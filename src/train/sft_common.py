from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path

from datasets import Dataset
from torch.utils.data import Sampler
from transformers import EarlyStoppingCallback, TrainingArguments
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


def build_sft_args(cfg, has_eval: bool):
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
