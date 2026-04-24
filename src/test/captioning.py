from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from test.eval import extract_assistant_response, write_jsonl


DEFAULT_PROMPT_TEMPLATE = (
    "Escribe un solo pie de foto en {language} para esta imagen. "
    "Debe ser una descripcion culturalmente adecuada de la imagen. "
    "Responde solo con el pie de foto en {language}, sin explicaciones."
)

SUBMISSION_COLUMNS = ("id", "filename", "split", "culture", "language", "iso_lang")


def parse_language_filter(raw_languages: list[str] | None) -> list[str] | None:
    if raw_languages is None:
        return None

    languages: list[str] = []
    for raw_language in raw_languages:
        for language in raw_language.split(","):
            cleaned = language.strip()
            if cleaned and cleaned not in languages:
                languages.append(cleaned)
    if not languages:
        raise ValueError("--languages must include at least one non-empty language.")
    return languages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AmericasNLP captioning submissions with Aya Vision."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="outputs/v3/aya-vision-32b-americas",
        help="Path to the merged model, LoRA adapter, or model id.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/captioning",
        help="Path to the captioning Hugging Face dataset saved with save_to_disk.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to caption.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional language filter. Accepts submission language names "
            "(bribri, maya, ...), iso_lang values, or comma-separated lists."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per caption.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples after language filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/captioning"),
        help="Directory for checkpoints, per-language JSONL files, and the zip.",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default="wixarika",
        help="Team name used for the zip filename.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="Single-digit submission version used in <language>-<version>.jsonl.",
    )
    parser.add_argument(
        "--zip-file",
        type=Path,
        default=None,
        help="Optional explicit zip path. Defaults to <output-dir>/<team-name>.zip.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format string with {language}, {iso_lang}, and {culture}.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when --do-sample is set.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p when --do-sample is set.",
    )
    args = parser.parse_args()
    args.languages = parse_language_filter(args.languages)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")
    if not 0 <= args.version <= 9:
        raise ValueError("--version must be a single digit from 0 to 9.")
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
        f"Adapter at {adapter_path} contains zero tensors; refusing to run inference."
    )


def _resolve_model_dtype(dtype_name: str):
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def _get_input_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def _load_processor(model_name_or_path: str, load_path: str):
    from transformers import AutoProcessor

    model_path = Path(model_name_or_path)
    processor_path = (
        model_name_or_path
        if (model_path / "processor_config.json").exists()
        else load_path
    )
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Aya Vision processor did not expose a tokenizer.")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer
    return processor


def load_model_and_processor(model_name_or_path: str, dtype_name: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model_dtype = _resolve_model_dtype(dtype_name)
    adapter_base_model_name = _load_adapter_base_model_name(model_name_or_path)
    load_path = adapter_base_model_name or model_name_or_path

    processor = _load_processor(model_name_or_path, load_path)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": model_dtype,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(load_path, **model_kwargs)
    if adapter_base_model_name is not None:
        _assert_adapter_has_weights(model_name_or_path)
        model = PeftModel.from_pretrained(model, model_name_or_path)
    elif not torch.cuda.is_available():
        model.to(torch.device("cpu"))

    model.eval()
    return model, processor


def _clean_caption(text: str) -> str:
    cleaned = extract_assistant_response(text).strip()
    cleaned = re.split(r"\s*<\|(?:END|END_RESPONSE|END_OF_TURN_TOKEN|START_OF_TURN_TOKEN)", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"\s*<\s*$", "", cleaned)
    cleaned = re.sub(r"^(predicted_caption|caption|pie de foto)\s*:\s*", "", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip()


def _make_prompt(example: dict[str, Any], prompt_template: str) -> str:
    return prompt_template.format(
        language=str(example.get("language") or "").strip(),
        iso_lang=str(example.get("iso_lang") or "").strip(),
        culture=str(example.get("culture") or "").strip(),
    )


def _make_chat_prompt(processor: Any, example: dict[str, Any], prompt_template: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _make_prompt(example, prompt_template)},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except OSError as exc:
        print(
            f"Warning: could not read image {path!r}; using a blank placeholder. {exc}",
            file=sys.stderr,
            flush=True,
        )
        return Image.new("RGB", (512, 512), color="white")


def _move_inputs_to_device(inputs: Any, device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for name, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[name] = value.to(device=device, dtype=dtype)
            else:
                moved[name] = value.to(device=device)
        else:
            moved[name] = value
    return moved


def _generate_batch(
    *,
    model: Any,
    processor: Any,
    examples: list[dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    dtype_name: str,
) -> list[str]:
    import torch

    prompts = [
        _make_chat_prompt(processor, example, prompt_template)
        for example in examples
    ]
    images = [_load_image(str(example["image_path"])) for example in examples]
    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    )
    input_width = inputs["input_ids"].shape[1]
    model_dtype = _resolve_model_dtype(dtype_name)
    inputs = _move_inputs_to_device(inputs, _get_input_device(model), model_dtype)

    tokenizer = processor.tokenizer
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    captions: list[str] = []
    for sequence in generated:
        new_tokens = sequence[input_width:].detach().cpu()
        caption = tokenizer.decode(new_tokens, skip_special_tokens=True)
        captions.append(_clean_caption(caption))
    return captions


def _record_key(example: dict[str, Any]) -> str:
    language = str(example.get("submission_language") or example.get("iso_lang") or "")
    return f"{language}:{example['id']}"


def _submission_record(example: dict[str, Any], predicted_caption: str) -> dict[str, Any]:
    return {
        **{column: example[column] for column in SUBMISSION_COLUMNS},
        "predicted_caption": predicted_caption,
    }


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if "predicted_caption" in record:
                record["predicted_caption"] = _clean_caption(str(record["predicted_caption"]))
            key = f"{record.get('submission_language') or record.get('iso_lang')}:{record.get('id')}"
            records[key] = record
    return records


def _write_checkpoint(
    path: Path,
    examples: list[dict[str, Any]],
    records_by_key: dict[str, dict[str, Any]],
) -> None:
    rows = [
        records_by_key[_record_key(example)]
        for example in examples
        if _record_key(example) in records_by_key
    ]
    write_jsonl(path, rows)


def _write_submission_files(
    *,
    output_dir: Path,
    examples: list[dict[str, Any]],
    records_by_key: dict[str, dict[str, Any]],
    version: int,
) -> list[Path]:
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        language = str(example["submission_language"]).strip()
        grouped.setdefault(language, []).append(records_by_key[_record_key(example)])

    paths: list[Path] = []
    for language, records in sorted(grouped.items()):
        path = submission_dir / f"{language}-{version}.jsonl"
        submission_rows = [
            {column: record[column] for column in (*SUBMISSION_COLUMNS, "predicted_caption")}
            for record in records
        ]
        write_jsonl(path, submission_rows)
        paths.append(path)
    return paths


def _write_zip(zip_path: Path, files: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_name(f"{zip_path.name}.tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, arcname=file_path.name)
    tmp_path.replace(zip_path)


def _select_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    dataset = load_from_disk(args.dataset_path)
    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Missing split '{args.split}'. Available: {available}")

    split_dataset = dataset[args.split]
    missing_columns = [
        column
        for column in (*SUBMISSION_COLUMNS, "submission_language", "image_path")
        if column not in split_dataset.column_names
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(split_dataset.column_names)
        raise ValueError(f"Missing columns: {missing}. Available: {available}")

    if "image" in split_dataset.column_names:
        split_dataset = split_dataset.remove_columns("image")

    examples = [dict(row) for row in split_dataset]
    if args.languages is not None:
        requested = {language.strip() for language in args.languages}
        examples = [
            example
            for example in examples
            if str(example.get("submission_language") or "").strip() in requested
            or str(example.get("iso_lang") or "").strip() in requested
        ]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No examples selected for captioning.")
    return examples


def _batch_indices(indices: list[int], batch_size: int) -> list[list[int]]:
    return [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = _select_examples(args)
    checkpoint_path = args.output_dir / "predictions.checkpoint.jsonl"
    records_by_key = _read_checkpoint(checkpoint_path)
    pending_indices = [
        index
        for index, example in enumerate(examples)
        if _record_key(example) not in records_by_key
    ]

    print(
        "Captioning plan: "
        + json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "dataset_path": args.dataset_path,
                "split": args.split,
                "languages": args.languages,
                "num_examples": len(examples),
                "completed": len(examples) - len(pending_indices),
                "pending": len(pending_indices),
                "batch_size": args.batch_size,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    if pending_indices:
        model, processor = load_model_and_processor(args.model_name_or_path, args.dtype)
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None

        index_batches = _batch_indices(pending_indices, args.batch_size)
        iterator = index_batches
        if tqdm is not None:
            iterator = tqdm(index_batches, desc="Generating captions", unit="batch")

        for batch in iterator:
            batch_examples = [examples[index] for index in batch]
            captions = _generate_batch(
                model=model,
                processor=processor,
                examples=batch_examples,
                prompt_template=args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                dtype_name=args.dtype,
            )
            for example, caption in zip(batch_examples, captions, strict=True):
                records_by_key[_record_key(example)] = {
                    **_submission_record(example, caption),
                    "submission_language": example["submission_language"],
                }
            _write_checkpoint(checkpoint_path, examples, records_by_key)

    missing = [
        _record_key(example)
        for example in examples
        if _record_key(example) not in records_by_key
    ]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions after generation.")

    submission_files = _write_submission_files(
        output_dir=args.output_dir,
        examples=examples,
        records_by_key=records_by_key,
        version=args.version,
    )
    zip_path = args.zip_file or (args.output_dir / f"{args.team_name}.zip")
    _write_zip(zip_path, submission_files)

    print(
        json.dumps(
            {
                "num_examples": len(examples),
                "submission_files": [str(path) for path in submission_files],
                "zip_file": str(zip_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
