from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from train.config import TrainConfig


def load_datasets(config: TrainConfig) -> DatasetDict:
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


def _format_instruction_example(example: Mapping[str, Any]) -> str:
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return (
        "<|user|>\n"
        f"{user_text}\n"
        "<|assistant|>\n"
        f"{output}"
    )


def _format_prompt_completion(example: Mapping[str, Any]) -> str:
    prompt = str(example.get("prompt", ""))
    completion = str(example.get("completion", ""))
    return f"{prompt}{completion}"


def _format_translation_example(
    example: Mapping[str, Any],
    source_column: str,
    target_column: str,
    source_name: str,
    target_name: str,
) -> str:
    source = str(example.get(source_column, "")).strip()
    target = str(example.get(target_column, "")).strip()
    return (
        "<|user|>\n"
        f"Translate from {source_name} to {target_name}.\n\n"
        f"{source}\n"
        "<|assistant|>\n"
        f"{target}"
    )


def build_text_dataset(
    dataset: Dataset,
    text_column: str | None,
    eos_token: str,
    split_name: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Dataset:
    columns = set(dataset.column_names)

    if text_column and text_column in columns:
        chosen = text_column

        def format_fn(example: Mapping[str, Any]) -> str:
            return str(example[chosen])

    elif "text" in columns:

        def format_fn(example: Mapping[str, Any]) -> str:
            return str(example["text"])

    elif {"prompt", "completion"}.issubset(columns):
        format_fn = _format_prompt_completion
    elif {"instruction", "output"}.issubset(columns):
        format_fn = _format_instruction_example
    elif {"es", "wix"}.issubset(columns):

        def format_fn(example: Mapping[str, Any]) -> str:
            return _format_translation_example(
                example=example,
                source_column="es",
                target_column="wix",
                source_name="Spanish",
                target_name="Wixarika",
            )

    elif "messages" in columns:
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "Dataset has `messages` column but tokenizer chat template is unavailable. "
                "Either preprocess to `text` or use a tokenizer with `apply_chat_template`."
            )

        def format_fn(example: Mapping[str, Any]) -> str:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

    else:
        col_list = ", ".join(sorted(columns))
        raise ValueError(
            "Unsupported dataset schema. Provide a `text` column, "
            "`prompt`/`completion`, `instruction`/`output`, or set `text_column`. "
            "For this project, `es`/`wix` is also supported. "
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


def prepare_splits(
    raw: DatasetDict,
    config: TrainConfig,
    eos_token: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[Dataset, Dataset | None]:
    if config.train_split not in raw:
        available = ", ".join(raw.keys())
        raise ValueError(f"Missing train split '{config.train_split}'. Available: {available}")

    train_dataset = build_text_dataset(
        dataset=raw[config.train_split],
        text_column=config.text_column,
        eos_token=eos_token,
        split_name=config.train_split,
        tokenizer=tokenizer,
    )

    eval_dataset = None
    if config.eval_split and config.eval_split in raw:
        eval_dataset = build_text_dataset(
            dataset=raw[config.eval_split],
            text_column=config.text_column,
            eos_token=eos_token,
            split_name=config.eval_split,
            tokenizer=tokenizer,
        )

    return train_dataset, eval_dataset
