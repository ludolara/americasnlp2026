from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
import random

from datasets import Dataset
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


class TemperatureSmoothedRepeatSampler(Sampler[int]):
    """Samples GRPO prompt groups by language before applying TRL's repeat pattern."""

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
        self.iteration = 0
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
    def _shuffled(indices: list[int], rng: random.Random) -> list[int]:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        return shuffled

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.iteration)
        usable_samples = (self.num_samples // self.batch_size) * self.batch_size
        num_batches = usable_samples // self.batch_size
        languages = tuple(self.language_weights.keys())
        weights = tuple(self.language_weights[language] for language in languages)
        pools = {
            language: self._shuffled(indices, rng)
            for language, indices in self.language_to_indices.items()
        }

        def draw_index(language: str) -> int:
            pool = pools[language]
            if not pool:
                pool = self._shuffled(self.language_to_indices[language], rng)
                pools[language] = pool
            return pool.pop()

        for _ in range(num_batches):
            batch_indices = [
                draw_index(rng.choices(languages, weights=weights, k=1)[0])
                for _ in range(self.batch_size)
            ]
            for _ in range(self.repeat_count):
                for index in batch_indices:
                    for _ in range(self.mini_repeat_count):
                        yield index

        self.iteration += 1
