#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

from datasets import Dataset, DatasetDict, Features, Image, Value


FEATURES = Features(
    {
        "id": Value("string"),
        "filename": Value("string"),
        "split": Value("string"),
        "culture": Value("string"),
        "language": Value("string"),
        "iso_lang": Value("string"),
        "target_caption": Value("string"),
        "submission_language": Value("string"),
        "image_path": Value("string"),
        "image": Image(decode=True),
    }
)

RAW_TO_HF_SPLIT = {
    "dev": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local HF dataset for the AmericasNLP captioning task."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/captioning"),
        help="Raw captioning directory containing dev/ and test/ language folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/captioning"),
        help="Where to save the Hugging Face dataset with save_to_disk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        has_files = any(path.iterdir())
        if has_files and not overwrite:
            raise FileExistsError(
                f"Output directory '{path}' is not empty. Use --overwrite to replace it."
            )
        if has_files and overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_image_path(language_dir: Path, filename: str) -> Path:
    candidates = [
        language_dir / filename,
        language_dir / "images" / Path(filename).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve image '{filename}'. Checked: {checked}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _build_raw_split(raw_dir: Path, raw_split: str) -> Dataset:
    split_dir = raw_dir / raw_split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing raw split directory: {split_dir}")

    rows: list[dict[str, Any]] = []
    for language_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        jsonl_path = language_dir / f"{language_dir.name}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing language JSONL file: {jsonl_path}")

        for raw_row in _read_jsonl(jsonl_path):
            filename = str(raw_row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"Row in {jsonl_path} is missing a non-empty filename.")

            image_path = _resolve_image_path(language_dir, filename)
            target_caption = raw_row.get("target_caption")
            rows.append(
                {
                    "id": str(raw_row.get("id") or "").strip(),
                    "filename": filename,
                    "split": str(raw_row.get("split") or raw_split).strip(),
                    "culture": str(raw_row.get("culture") or "").strip(),
                    "language": str(raw_row.get("language") or "").strip(),
                    "iso_lang": str(raw_row.get("iso_lang") or "").strip(),
                    "target_caption": (
                        "" if target_caption is None else str(target_caption).strip()
                    ),
                    "submission_language": language_dir.name,
                    "image_path": str(image_path),
                    "image": str(image_path),
                }
            )

    if not rows:
        raise ValueError(f"No rows found in raw split: {split_dir}")

    missing_required = [
        row["id"]
        for row in rows
        if not row["id"] or not row["language"] or not row["iso_lang"]
    ]
    if missing_required:
        raise ValueError(
            f"{len(missing_required)} rows in split '{raw_split}' are missing id, "
            "language, or iso_lang."
        )

    return Dataset.from_list(rows, features=FEATURES)


def _print_summary(dataset: DatasetDict) -> None:
    print(dataset)
    for split_name, split_dataset in dataset.items():
        language_counts = Counter(split_dataset["submission_language"])
        counts = ", ".join(
            f"{language}={count}" for language, count in sorted(language_counts.items())
        )
        print(f"{split_name}: {split_dataset.num_rows} rows ({counts})")


def main() -> None:
    args = parse_args()

    dataset = DatasetDict(
        {
            hf_split: _build_raw_split(args.raw_dir, raw_split)
            for raw_split, hf_split in RAW_TO_HF_SPLIT.items()
        }
    )

    _prepare_output_dir(args.output_dir, args.overwrite)
    dataset.save_to_disk(str(args.output_dir))

    print(f"Saved captioning dataset to: {args.output_dir}")
    _print_summary(dataset)


if __name__ == "__main__":
    main()
