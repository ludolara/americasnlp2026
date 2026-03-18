from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from train.config import DatasetConfigMixin, GRPOTrainConfig, SFTTrainConfig
from train.constants import (
    AYA_BASE_CHAT_TEMPLATE,
    AYA_END_OF_TURN_TOKEN,
    AYA_REQUIRED_CHAT_TOKENS,
    SPANISH_LANGUAGE_NAME,
)


def _distributed_map_kwargs() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return {}

    # DDP ranks can race on the same Hugging Face Arrow cache file when mapping a
    # dataset loaded from disk. Keeping the mapped dataset in memory avoids the
    # shared on-disk cache path entirely.
    return {
        "keep_in_memory": True,
        "load_from_cache_file": False,
    }


def load_datasets(config: DatasetConfigMixin) -> DatasetDict:
    if config.local_dataset_path:
        loaded = load_from_disk(config.local_dataset_path)
        if isinstance(loaded, DatasetDict):
            return loaded
        if isinstance(loaded, Dataset):
            return DatasetDict({"train": loaded})
        raise TypeError(
            "Unsupported object returned by load_from_disk for "
            f"`local_dataset_path={config.local_dataset_path}`."
        )

    if config.dataset_name:
        return load_dataset(config.dataset_name, config.dataset_config_name)

    if not config.train_file:
        raise ValueError(
            "Provide one of `local_dataset_path`, `dataset_name`, or `train_file` in config."
        )

    data_files: dict[str, str] = {"train": config.train_file}
    if config.eval_file:
        data_files["validation"] = config.eval_file

    extension = config.train_file.rsplit(".", maxsplit=1)[-1].lower()
    if extension == "jsonl":
        extension = "json"

    return load_dataset(extension, data_files=data_files)


def _has_token(tokenizer: PreTrainedTokenizerBase, token: str) -> bool:
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id is not None and token_id != tokenizer.unk_token_id


def ensure_chat_template(
    tokenizer: PreTrainedTokenizerBase | None,
) -> PreTrainedTokenizerBase:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tiny Aya formatting requires a tokenizer with `apply_chat_template`."
        )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    missing_tokens = [
        token for token in AYA_REQUIRED_CHAT_TOKENS if not _has_token(tokenizer, token)
    ]
    if missing_tokens:
        missing = ", ".join(missing_tokens)
        raise ValueError(
            "Tokenizer does not define a chat template and is missing Aya turn tokens: "
            f"{missing}"
        )

    tokenizer.chat_template = AYA_BASE_CHAT_TEMPLATE
    tokenizer.eos_token = AYA_END_OF_TURN_TOKEN
    return tokenizer


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase | None,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    tokenizer = ensure_chat_template(tokenizer)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _format_translation_user_text(
    source: str,
    target_name: str,
    source_name: str = SPANISH_LANGUAGE_NAME,
) -> str:
    return f"Traduce del {source_name} al {target_name}.\n\n{source.strip()}"


def format_translation_prompt(
    source: str,
    target_name: str,
    tokenizer: PreTrainedTokenizerBase | None,
    *,
    source_name: str = SPANISH_LANGUAGE_NAME,
) -> str:
    return _apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": _format_translation_user_text(
                    source,
                    target_name,
                    source_name=source_name,
                ),
            }
        ],
        add_generation_prompt=True,
    )


def _resolve_target_name(example: Mapping[str, Any]) -> str:
    target_name = str(example.get("language") or "").strip()
    if not target_name:
        raise ValueError("Translation examples must include a non-empty `language` field.")
    return target_name


def _format_instruction_example(
    example: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase | None,
) -> str:
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return _apply_chat_template(
        tokenizer,
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output},
        ],
        add_generation_prompt=False,
    )


def _format_prompt_completion(example: Mapping[str, Any]) -> str:
    prompt = str(example.get("prompt", ""))
    completion = str(example.get("completion", ""))
    return f"{prompt}{completion}"


def _format_translation_example(
    example: Mapping[str, Any],
    source_column: str,
    target_column: str,
    tokenizer: PreTrainedTokenizerBase | None,
    *,
    reverse_direction: bool = False,
) -> str:
    source, target, source_name, target_name = _resolve_translation_pair(
        example,
        source_column=source_column,
        target_column=target_column,
        reverse_direction=reverse_direction,
    )
    return _apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": _format_translation_user_text(
                    source,
                    target_name,
                    source_name=source_name,
                ),
            },
            {"role": "assistant", "content": target},
        ],
        add_generation_prompt=False,
    )


def _should_reverse_translation(
    *,
    bidirectional_translation: bool,
    seed: int,
    split_name: str,
    index: int,
) -> bool:
    if not bidirectional_translation:
        return False

    pair_index, pair_offset = divmod(index, 2)
    digest = hashlib.blake2b(
        f"{split_name}:{seed}:{pair_index}".encode("utf-8"),
        digest_size=1,
    ).digest()
    return pair_offset == (digest[0] & 1)


def _resolve_translation_pair(
    example: Mapping[str, Any],
    *,
    source_column: str,
    target_column: str,
    reverse_direction: bool,
) -> tuple[str, str, str, str]:
    source = str(example.get(source_column, "")).strip()
    target = str(example.get(target_column, "")).strip()
    target_name = _resolve_target_name(example)

    if reverse_direction:
        return target, source, target_name, SPANISH_LANGUAGE_NAME

    return source, target, SPANISH_LANGUAGE_NAME, target_name


def build_text_dataset(
    dataset: Dataset,
    config: SFTTrainConfig,
    eos_token: str,
    split_name: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Dataset:
    columns = set(dataset.column_names)

    if config.text_column and config.text_column in columns:
        chosen = config.text_column

        def format_fn(example: Mapping[str, Any], _: int) -> str:
            return str(example[chosen])

    elif "text" in columns:

        def format_fn(example: Mapping[str, Any], _: int) -> str:
            return str(example["text"])

    elif {"prompt", "completion"}.issubset(columns):

        def format_fn(example: Mapping[str, Any], _: int) -> str:
            return _format_prompt_completion(example)

    elif {"instruction", "output"}.issubset(columns):

        def format_fn(example: Mapping[str, Any], _: int) -> str:
            return _format_instruction_example(example, tokenizer)

    elif {config.source_column, config.target_column}.issubset(columns):
        if "language" not in columns:
            raise ValueError(
                "Translation SFT datasets must include a `language` column for multilingual prompts."
            )

        def format_fn(example: Mapping[str, Any], index: int) -> str:
            return _format_translation_example(
                example=example,
                source_column=config.source_column,
                target_column=config.target_column,
                tokenizer=tokenizer,
                reverse_direction=_should_reverse_translation(
                    bidirectional_translation=config.bidirectional_translation,
                    seed=config.seed,
                    split_name=split_name,
                    index=index,
                ),
            )

    elif "messages" in columns:

        def format_fn(example: Mapping[str, Any], _: int) -> str:
            return _apply_chat_template(
                tokenizer,
                example["messages"],
                add_generation_prompt=False,
            )

    else:
        col_list = ", ".join(sorted(columns))
        raise ValueError(
            "Unsupported dataset schema. Provide a `text` column, "
            "`prompt`/`completion`, `instruction`/`output`, or set `text_column`. "
            f"For this project, `{config.source_column}`/`{config.target_column}` is also supported. "
            f"Got: {col_list}"
        )

    def mapper(example: Mapping[str, Any], index: int) -> dict[str, str]:
        text = format_fn(example, index).strip()
        if eos_token and text and not text.endswith(eos_token):
            text = text + eos_token
        return {"text": text}

    return dataset.map(
        mapper,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting {split_name} split to text",
        **_distributed_map_kwargs(),
    )


def prepare_sft_splits(
    raw: DatasetDict,
    config: SFTTrainConfig,
    eos_token: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[Dataset, Dataset | None]:
    if config.train_split not in raw:
        available = ", ".join(raw.keys())
        raise ValueError(f"Missing train split '{config.train_split}'. Available: {available}")

    train_dataset = build_text_dataset(
        dataset=raw[config.train_split],
        config=config,
        eos_token=eos_token,
        split_name=config.train_split,
        tokenizer=tokenizer,
    )

    eval_dataset = None
    if config.eval_split and config.eval_split in raw:
        eval_dataset = build_text_dataset(
            dataset=raw[config.eval_split],
            config=config,
            eos_token=eos_token,
            split_name=config.eval_split,
            tokenizer=tokenizer,
        )

    return train_dataset, eval_dataset


def build_grpo_dataset(
    dataset: Dataset,
    config: GRPOTrainConfig,
    split_name: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Dataset:
    columns = set(dataset.column_names)
    missing_columns = [
        name for name in (config.source_column, config.target_column) if name not in columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(sorted(columns))
        raise ValueError(
            f"Missing columns for GRPO split '{split_name}': {missing}. Available: {available}"
        )
    if "language" not in columns:
        raise ValueError(
            f"Translation GRPO datasets must include a `language` column for prompts in split '{split_name}'."
        )

    def mapper(example: Mapping[str, Any], index: int) -> dict[str, str]:
        reverse_direction = _should_reverse_translation(
            bidirectional_translation=config.bidirectional_translation,
            seed=config.seed,
            split_name=split_name,
            index=index,
        )
        source, reference, source_name, target_name = _resolve_translation_pair(
            example,
            source_column=config.source_column,
            target_column=config.target_column,
            reverse_direction=reverse_direction,
        )
        row = {
            "prompt": format_translation_prompt(
                source,
                target_name,
                tokenizer,
                source_name=source_name,
            ),
            "source": source,
            "reference": reference,
        }
        if "language" in example:
            row["language"] = target_name
        if "language_code" in example:
            row["language_code"] = (
                config.source_column
                if reverse_direction
                else str(example.get("language_code") or "").strip()
            )
        if "source_type" in example:
            row["source_type"] = str(example.get("source_type") or "").strip()
        if "source_tag" in example:
            row["source_tag"] = str(example.get("source_tag") or "").strip()
        return row

    return dataset.map(
        mapper,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting {split_name} split for GRPO",
        **_distributed_map_kwargs(),
    )


def prepare_grpo_splits(
    raw: DatasetDict,
    config: GRPOTrainConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[Dataset, Dataset | None]:
    if config.train_split not in raw:
        available = ", ".join(raw.keys())
        raise ValueError(f"Missing train split '{config.train_split}'. Available: {available}")

    train_dataset = build_grpo_dataset(
        dataset=raw[config.train_split],
        config=config,
        split_name=config.train_split,
        tokenizer=tokenizer,
    )

    eval_dataset = None
    if config.eval_split and config.eval_split in raw:
        eval_dataset = build_grpo_dataset(
            dataset=raw[config.eval_split],
            config=config,
            split_name=config.eval_split,
            tokenizer=tokenizer,
        )

    return train_dataset, eval_dataset
