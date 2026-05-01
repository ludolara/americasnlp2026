from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
from datasets import Dataset
from sacrebleu.metrics import CHRF
from torch.utils.data import Sampler
from transformers import EarlyStoppingCallback, Trainer
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import (
    FSDP,
    is_conversational,
    profiling_context,
    unwrap_model_for_generation,
)

from train.language_sampling import (
    TemperatureSmoothedRepeatSampler,
    compute_smoothed_mix,
    count_languages,
    extract_languages,
    format_weighted_mix,
)


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


class PatchedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, train_sampler: Sampler[int] | None = None, **kwargs):
        self._train_sampler_override = train_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset: Dataset | None = None):
        if self._train_sampler_override is not None:
            return self._train_sampler_override
        return super()._get_train_sampler(dataset)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        # GRPOTrainer computes reward metrics during eval, but upstream only merges them
        # into a copy of `logs`. Mutating in place keeps them visible to early stopping and
        # best-model selection, which inspect the original evaluation metrics dict.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time)

    def _generate_single_turn(self, prompts: list):
        if (
            self.use_vllm
            or self.use_transformers_paged
            or is_conversational({"prompt": prompts[0]})
        ):
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

        prompt_ids = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_ids = [
            p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)
        ]
        completion_ids = [
            c[m].tolist()
            for c, m in zip(completion_ids, completion_mask.bool(), strict=True)
        ]

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
                    parts.extend(
                        str(part.get("text", "")).strip()
                        for part in content
                        if isinstance(part, dict)
                    )
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
    if "<|CHATBOT_TOKEN|>" in cleaned:
        cleaned = cleaned.rsplit("<|CHATBOT_TOKEN|>", maxsplit=1)[-1].strip()
    if "<|START_RESPONSE|>" in cleaned:
        cleaned = cleaned.rsplit("<|START_RESPONSE|>", maxsplit=1)[-1].strip()
    for marker in (
        "<|END_RESPONSE|>",
        "<|END_OF_TURN_TOKEN|>",
        "<|START_OF_TURN_TOKEN|>",
    ):
        if marker in cleaned:
            cleaned = cleaned.split(marker, maxsplit=1)[0].strip()
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


def _build_train_sampler(
    dataset: Dataset,
    *,
    seed: int,
    alpha: float,
    generation_batch_size: int,
    num_generations: int,
    repeat_count: int,
    world_size: int,
) -> Sampler[int] | None:
    if "language" not in dataset.column_names:
        return None

    counts = count_languages(dataset)
    print(f"Train language mix (raw): {format_weighted_mix(counts)}")

    if alpha == 1.0:
        print("Using default GRPO sampling because language_sampling_alpha=1.0.")
        return None

    if len(counts) < 2:
        return None

    smoothed_mix = compute_smoothed_mix(counts, alpha)
    print(
        f"Train language mix (alpha={alpha:.2f}): "
        f"{format_weighted_mix(smoothed_mix)}"
    )
    if world_size > 1:
        print(
            f"Using distributed GRPO alpha-smoothed sampler "
            f"(alpha={alpha:.2f}, world_size={world_size}) with shared-seed batching."
        )

    prompt_batch_size = generation_batch_size // num_generations
    return TemperatureSmoothedRepeatSampler(
        languages=extract_languages(dataset),
        seed=seed,
        alpha=alpha,
        batch_size=prompt_batch_size,
        mini_repeat_count=num_generations,
        repeat_count=repeat_count,
    )
