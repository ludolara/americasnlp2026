#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from datasets import Dataset, DatasetDict, Features, Value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local HF dataset from AmericasNLP Wixarika-Spanish files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/wixarika_spanish_raw"),
        help="Directory containing train/dev .es and .hch files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wixarika_spanish_hf"),
        help="Where to save the Hugging Face dataset (save_to_disk).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim leading/trailing whitespace from each line (default: true).",
    )
    return parser.parse_args()


def _read_lines(path: Path, trim: bool) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if trim:
        return [line.strip() for line in lines]
    return lines


def _build_split(es_path: Path, wix_path: Path, split_name: str, trim: bool) -> Dataset:
    es_lines = _read_lines(es_path, trim=trim)
    wix_lines = _read_lines(wix_path, trim=trim)

    if len(es_lines) != len(wix_lines):
        raise ValueError(
            f"{split_name} split has mismatched files: "
            f"{es_path.name}={len(es_lines)} vs {wix_path.name}={len(wix_lines)}"
        )

    features = Features(
        {
            "es": Value("string"),
            "wix": Value("string"),
        }
    )

    rows = {
        "es": es_lines,
        "wix": wix_lines,
    }
    return Dataset.from_dict(rows, features=features)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        has_files = any(path.iterdir())
        if has_files and not overwrite:
            raise FileExistsError(
                f"Output directory '{path}' is not empty. "
                "Use --overwrite to replace it."
            )
        if has_files and overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir

    train_es = raw_dir / "train.es"
    train_wix = raw_dir / "train.hch"
    dev_es = raw_dir / "dev.es"
    dev_wix = raw_dir / "dev.hch"

    required = [train_es, train_wix, dev_es, dev_wix]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing required files: {missing_str}")

    train_dataset = _build_split(train_es, train_wix, "train", trim=args.trim)
    dev_dataset = _build_split(dev_es, dev_wix, "dev", trim=args.trim)

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": dev_dataset,
        }
    )

    _prepare_output_dir(output_dir, args.overwrite)
    dataset.save_to_disk(str(output_dir))

    print(f"Saved dataset to: {output_dir}")
    print(dataset)
    for split_name, split_ds in dataset.items():
        print(f"{split_name}: {split_ds.num_rows} rows")


if __name__ == "__main__":
    main()
