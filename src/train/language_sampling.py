from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
import random

from datasets import Dataset
import torch
from torch.utils.data import Sampler


def extract_languages(dataset: Dataset) -> list[str]:
    return [str(language).strip() for language in dataset["language"]]


def count_languages(dataset: Dataset) -> Counter[str]:
    return Counter(extract_languages(dataset))


def format_weighted_mix(weights: Counter[str] | dict[str, float]) -> str:
    total = sum(weights.values())
    if total <= 0:
        return ""
    return ", ".join(
        f"{language}={value / total:.1%}"
        for language, value in sorted(weights.items())
    )


def compute_language_sample_counts(
    language_weights: dict[str, float],
    sample_size: int,
) -> dict[str, int]:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1.")

    normalized_weights = {
        str(language).strip(): float(weight)
        for language, weight in language_weights.items()
        if float(weight) > 0
    }
    total_weight = sum(normalized_weights.values())
    if total_weight <= 0:
        raise ValueError("language_weights must contain at least one positive weight.")

    exact_counts = {
        language: sample_size * weight / total_weight
        for language, weight in normalized_weights.items()
    }
    counts = {
        language: int(exact_count)
        for language, exact_count in exact_counts.items()
    }

    remainder = sample_size - sum(counts.values())
    if remainder > 0:
        ranked_languages = sorted(
            exact_counts,
            key=lambda language: (-(exact_counts[language] - counts[language]), language),
        )
        for language in ranked_languages[:remainder]:
            counts[language] += 1

    return counts


def sample_dataset_by_language(
    dataset: Dataset,
    *,
    sample_size: int,
    seed: int,
    language_weights: dict[str, float] | None = None,
    language_column: str = "language",
) -> Dataset:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1.")
    if sample_size > len(dataset):
        raise ValueError(
            f"Cannot sample {sample_size} rows from a dataset with {len(dataset)} rows."
        )
    if language_column not in dataset.column_names:
        available = ", ".join(dataset.column_names)
        raise ValueError(
            f"Missing language column '{language_column}'. Available: {available}"
        )

    rng = random.Random(seed)
    if language_weights is None:
        return dataset.select(rng.sample(range(len(dataset)), k=sample_size))

    language_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(dataset[language_column]):
        language_to_indices[str(language).strip()].append(index)

    sample_counts = compute_language_sample_counts(language_weights, sample_size)
    missing_languages = [
        language
        for language, count in sample_counts.items()
        if count > 0 and language not in language_to_indices
    ]
    if missing_languages:
        available = ", ".join(sorted(language_to_indices))
        missing = ", ".join(sorted(missing_languages))
        raise ValueError(
            f"Requested eval sample languages not found: {missing}. Available: {available}"
        )

    insufficient_languages = [
        f"{language} requested={count} available={len(language_to_indices[language])}"
        for language, count in sorted(sample_counts.items())
        if count > len(language_to_indices[language])
    ]
    if insufficient_languages:
        details = "; ".join(insufficient_languages)
        raise ValueError(f"Not enough eval rows for requested language mix: {details}")

    selected_indices: list[int] = []
    for language, count in sorted(sample_counts.items()):
        if count > 0:
            selected_indices.extend(rng.sample(language_to_indices[language], k=count))

    rng.shuffle(selected_indices)
    return dataset.select(selected_indices)


def compute_smoothed_mix(counts: Counter[str], alpha: float) -> dict[str, float]:
    return {
        language: count ** alpha
        for language, count in counts.items()
    }


class TemperatureSmoothedLanguageSampler(Sampler[int]):
    """Resamples examples by language using p(lang) proportional to count(lang)^alpha."""

    def __init__(
        self,
        languages: list[str],
        seed: int,
        alpha: float,
        epoch_size: int,
    ) -> None:
        self.seed = seed
        self.alpha = alpha
        self.epoch_size = epoch_size
        self.epoch = 0
        self.language_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, language in enumerate(languages):
            self.language_to_indices[language].append(index)
        self.language_weights = {
            language: len(indices) ** self.alpha
            for language, indices in self.language_to_indices.items()
        }

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        languages = tuple(self.language_weights.keys())
        weights = tuple(self.language_weights[language] for language in languages)

        for _ in range(self.epoch_size):
            language = rng.choices(languages, weights=weights, k=1)[0]
            yield rng.choice(self.language_to_indices[language])

        self.epoch += 1


class DistributedTemperatureSmoothedLanguageSampler(Sampler[int]):
    """DDP-safe language sampler that shards a shared alpha-smoothed stream by rank."""

    def __init__(
        self,
        languages: list[str],
        seed: int,
        alpha: float,
        epoch_size: int,
        *,
        num_replicas: int,
        rank: int,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")

        self.seed = seed
        self.alpha = alpha
        self.epoch_size = epoch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.language_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, language in enumerate(languages):
            self.language_to_indices[language].append(index)
        self.language_weights = {
            language: len(indices) ** self.alpha
            for language, indices in self.language_to_indices.items()
        }

        self.num_samples = (self.epoch_size + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        languages = tuple(self.language_weights.keys())
        weights = tuple(self.language_weights[language] for language in languages)

        for draw_index in range(self.total_size):
            language = rng.choices(languages, weights=weights, k=1)[0]
            index = rng.choice(self.language_to_indices[language])
            if draw_index % self.num_replicas == self.rank:
                yield index

        self.epoch += 1


class TemperatureSmoothedRepeatSampler(Sampler[int]):
    """Samples GRPO prompt groups by language before applying TRL's repeat pattern.

    This sampler intentionally mirrors TRL's `RepeatSampler` contract: every rank
    builds the same global prompt stream, and Accelerate shards batches by process.
    Exposing a `torch.Generator` lets Accelerate keep the sampler RNG aligned
    across ranks in distributed training.
    """

    def __init__(
        self,
        languages: list[str],
        seed: int,
        alpha: float,
        batch_size: int,
        mini_repeat_count: int,
        repeat_count: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if mini_repeat_count < 1:
            raise ValueError("mini_repeat_count must be >= 1.")
        if repeat_count < 1:
            raise ValueError("repeat_count must be >= 1.")

        self.seed = seed
        self.alpha = alpha
        self.batch_size = batch_size
        self.mini_repeat_count = mini_repeat_count
        self.repeat_count = repeat_count
        self.num_samples = len(languages)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.language_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, language in enumerate(languages):
            self.language_to_indices[language].append(index)
        self.language_weights = {
            language: len(indices) ** self.alpha
            for language, indices in self.language_to_indices.items()
        }

    def __len__(self) -> int:
        usable_samples = (self.num_samples // self.batch_size) * self.batch_size
        return usable_samples * self.mini_repeat_count * self.repeat_count

    @staticmethod
    def _shuffled(indices: list[int], generator: torch.Generator) -> list[int]:
        if len(indices) <= 1:
            return list(indices)
        order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[index] for index in order]

    def __iter__(self) -> Iterator[int]:
        usable_samples = (self.num_samples // self.batch_size) * self.batch_size
        num_batches = usable_samples // self.batch_size
        languages = tuple(self.language_weights.keys())
        weights = torch.tensor(
            [self.language_weights[language] for language in languages],
            dtype=torch.float,
        )
        pools = {
            language: self._shuffled(indices, self.generator)
            for language, indices in self.language_to_indices.items()
        }

        def draw_index(language: str) -> int:
            pool = pools[language]
            if not pool:
                pool = self._shuffled(self.language_to_indices[language], self.generator)
                pools[language] = pool
            return pool.pop()

        for _ in range(num_batches):
            batch_indices = [
                draw_index(
                    languages[
                        torch.multinomial(
                            weights,
                            num_samples=1,
                            replacement=True,
                            generator=self.generator,
                        ).item()
                    ]
                )
                for _ in range(self.batch_size)
            ]
            for _ in range(self.repeat_count):
                for index in batch_indices:
                    for _ in range(self.mini_repeat_count):
                        yield index
