from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from train.config import DatasetConfigMixin, GRPOTrainConfig, SFTTrainConfig


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


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase | None,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tiny Aya formatting requires a tokenizer with `apply_chat_template`."
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def format_translation_prompt(
    source: str,
    source_name: str,
    target_name: str,
    tokenizer: PreTrainedTokenizerBase | None,
) -> str:
    return _apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    f"Translate from {source_name} to {target_name}.\n\n{source.strip()}"
                ),
            }
        ],
        add_generation_prompt=True,
    )


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
) -> str:
    source = str(example.get(source_column, "")).strip()
    target = str(example.get(target_column, "")).strip()
    target_name = str(example.get("language") or "").strip()
    if not target_name:
        raise ValueError("Translation SFT examples must include a non-empty `language` field.")
    return _apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": f"Traduce del español al {target_name}.\n\n{source}",
            },
            {"role": "assistant", "content": target},
        ],
        add_generation_prompt=False,
    )


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

        def format_fn(example: Mapping[str, Any]) -> str:
            return str(example[chosen])

    elif "text" in columns:

        def format_fn(example: Mapping[str, Any]) -> str:
            return str(example["text"])

    elif {"prompt", "completion"}.issubset(columns):
        format_fn = _format_prompt_completion
    elif {"instruction", "output"}.issubset(columns):

        def format_fn(example: Mapping[str, Any]) -> str:
            return _format_instruction_example(example, tokenizer)

    elif {config.source_column, config.target_column}.issubset(columns):
        if "language" not in columns:
            raise ValueError(
                "Translation SFT datasets must include a `language` column for multilingual prompts."
            )

        def format_fn(example: Mapping[str, Any]) -> str:
            return _format_translation_example(
                example=example,
                source_column=config.source_column,
                target_column=config.target_column,
                tokenizer=tokenizer,
            )

    elif "messages" in columns:
        def format_fn(example: Mapping[str, Any]) -> str:
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

    def mapper(example: Mapping[str, Any]) -> dict[str, str]:
        text = format_fn(example).strip()
        if eos_token and text and not text.endswith(eos_token):
            text = text + eos_token
        return {"text": text}

    return dataset.map(
        mapper,
        remove_columns=dataset.column_names,
        desc=f"Formatting {split_name} split to text",
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

    def mapper(example: Mapping[str, Any]) -> dict[str, str]:
        source = str(example[config.source_column]).strip()
        reference = str(example[config.target_column]).strip()
        return {
            "prompt": format_translation_prompt(
                source,
                config.source_name,
                config.target_name,
                tokenizer,
            ),
            "source": source,
            "reference": reference,
        }

    return dataset.map(
        mapper,
        remove_columns=dataset.column_names,
        desc=f"Formatting {split_name} split for GRPO",
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
