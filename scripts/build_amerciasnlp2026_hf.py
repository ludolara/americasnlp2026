#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import shutil

from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets


FEATURES = Features(
    {
        "es": Value("string"),
        "target": Value("string"),
        "language": Value("string"),
        "language_code": Value("string"),
        "source_type": Value("string"),
        "source_tag": Value("string"),
    }
)

TEST_BASENAME_CANDIDATES = ("2026_test", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local multilingual HF dataset for AmerciasNLP 2026."
    )
    parser.add_argument(
        "--wixarika-dir",
        type=Path,
        default=Path("data/raw/wixarika-spanish"),
        help="Directory containing Wixarika-Spanish files.",
    )
    parser.add_argument(
        "--bribri-dir",
        type=Path,
        default=Path("data/raw/bribri-spanish"),
        help="Directory containing Bribri-Spanish files.",
    )
    parser.add_argument(
        "--guarani-dir",
        type=Path,
        default=Path("data/raw/guarani-spanish"),
        help="Directory containing Guarani-Spanish files.",
    )
    parser.add_argument(
        "--maya-dir",
        type=Path,
        default=Path("data/raw/maya-spanish"),
        help="Directory containing Maya-Spanish files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amerciasnlp2026"),
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


def _dataset_from_rows(
    *,
    es_lines: list[str],
    target_lines: list[str],
    language: str,
    language_code: str,
    source_type: str,
    source_tags: list[str],
) -> Dataset:
    num_rows = len(es_lines)
    rows = {
        "es": es_lines,
        "target": target_lines,
        "language": [language] * num_rows,
        "language_code": [language_code] * num_rows,
        "source_type": [source_type] * num_rows,
        "source_tag": source_tags,
    }
    return Dataset.from_dict(rows, features=FEATURES)


def _build_parallel_split(
    es_path: Path,
    target_path: Path,
    *,
    split_name: str,
    language: str,
    language_code: str,
    trim: bool,
) -> Dataset:
    es_lines = _read_lines(es_path, trim=trim)
    target_lines = _read_lines(target_path, trim=trim)

    if len(es_lines) != len(target_lines):
        raise ValueError(
            f"{split_name} split has mismatched files: "
            f"{es_path.name}={len(es_lines)} vs {target_path.name}={len(target_lines)}"
        )

    return _dataset_from_rows(
        es_lines=es_lines,
        target_lines=target_lines,
        language=language,
        language_code=language_code,
        source_type="parallel",
        source_tags=[split_name] * len(es_lines),
    )


def _build_tsv_split(
    tsv_path: Path,
    *,
    split_name: str,
    language: str,
    language_code: str,
    source_type: str,
    trim: bool,
) -> Dataset:
    es_lines: list[str] = []
    target_lines: list[str] = []
    source_tags: list[str] = []

    with tsv_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            row = raw_line.rstrip("\n").split("\t")

            if trim:
                row = [cell.strip() for cell in row]

            if not any(row):
                continue

            if len(row) != 3:
                raise ValueError(
                    f"{split_name} TSV has malformed row at line {line_number}: "
                    f"expected 3 tab-separated columns, got {len(row)}"
                )

            source_tag, es, target = row
            source_tags.append(source_tag)
            es_lines.append(es)
            target_lines.append(target)

    return _dataset_from_rows(
        es_lines=es_lines,
        target_lines=target_lines,
        language=language,
        language_code=language_code,
        source_type=source_type,
        source_tags=source_tags,
    )


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


def _concat_parts(parts: list[Dataset], split_name: str) -> Dataset:
    if not parts:
        raise ValueError(f"No datasets were collected for split '{split_name}'.")
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def _resolve_optional_split(
    raw_dir: Path,
    *,
    basename_candidates: tuple[str, ...],
    target_suffix: str,
) -> tuple[Path, Path, str] | None:
    for basename in basename_candidates:
        es_path = raw_dir / f"{basename}.es"
        target_path = raw_dir / f"{basename}.{target_suffix}"
        if es_path.exists() and target_path.exists():
            return es_path, target_path, basename
        if es_path.exists() != target_path.exists():
            raise FileNotFoundError(
                f"Incomplete optional split '{basename}' in '{raw_dir}': "
                f"expected both {es_path.name} and {target_path.name}"
            )
    return None


def _print_summary(dataset: DatasetDict) -> None:
    print(dataset)
    for split_name, split_ds in dataset.items():
        language_counts = Counter(split_ds["language"])
        counts_str = ", ".join(
            f"{language}={count}" for language, count in sorted(language_counts.items())
        )
        print(f"{split_name}: {split_ds.num_rows} rows ({counts_str})")


def _print_random_examples(dataset: DatasetDict, num_examples: int = 10) -> None:
    candidates = [
        (split_name, row_idx)
        for split_name, split_ds in dataset.items()
        for row_idx in range(split_ds.num_rows)
    ]
    if not candidates:
        print("No examples available.")
        return

    sample_size = min(num_examples, len(candidates))
    sampled_examples = random.sample(candidates, k=sample_size)

    print(f"Random examples ({sample_size}):")
    for example_number, (split_name, row_idx) in enumerate(sampled_examples, start=1):
        example = dataset[split_name][row_idx]
        print(
            f"[{example_number}] split={split_name} "
            f"language={example['language']} "
            f"source_type={example['source_type']} "
            f"source_tag={example['source_tag']}"
        )
        print(f"  es: {example['es']}")
        print(f"  target: {example['target']}")
        print()


def main() -> None:
    args = parse_args()

    wixarika_dir = args.wixarika_dir
    bribri_dir = args.bribri_dir
    guarani_dir = args.guarani_dir
    maya_dir = args.maya_dir
    output_dir = args.output_dir

    required = [
        wixarika_dir / "train.es",
        wixarika_dir / "train.hch",
        wixarika_dir / "dev.es",
        wixarika_dir / "dev.hch",
        wixarika_dir / "extra.tsv",
        wixarika_dir / "synthetic.tsv",
        bribri_dir / "train.es",
        bribri_dir / "train.bzd",
        bribri_dir / "dev.es",
        bribri_dir / "dev.bzd",
        guarani_dir / "train.es",
        guarani_dir / "train.gn",
        guarani_dir / "dev.es",
        guarani_dir / "dev.gn",
        guarani_dir / "extra.tsv",
        guarani_dir / "synthetic.tsv",
        maya_dir / "train.es",
        maya_dir / "train.maya",
        maya_dir / "dev.es",
        maya_dir / "dev.maya",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing required files: {missing_str}")

    train_parts = [
        _build_parallel_split(
            wixarika_dir / "train.es",
            wixarika_dir / "train.hch",
            split_name="train",
            language="wixarika",
            language_code="wix",
            trim=args.trim,
        ),
        _build_tsv_split(
            wixarika_dir / "extra.tsv",
            split_name="wixarika extra",
            language="wixarika",
            language_code="wix",
            source_type="extra",
            trim=args.trim,
        ),
        _build_tsv_split(
            wixarika_dir / "synthetic.tsv",
            split_name="wixarika synthetic",
            language="wixarika",
            language_code="wix",
            source_type="synthetic",
            trim=args.trim,
        ),
        _build_parallel_split(
            bribri_dir / "train.es",
            bribri_dir / "train.bzd",
            split_name="train",
            language="bribri",
            language_code="bzd",
            trim=args.trim,
        ),
        _build_parallel_split(
            guarani_dir / "train.es",
            guarani_dir / "train.gn",
            split_name="train",
            language="guarani",
            language_code="gn",
            trim=args.trim,
        ),
        _build_tsv_split(
            guarani_dir / "extra.tsv",
            split_name="guarani extra",
            language="guarani",
            language_code="gn",
            source_type="extra",
            trim=args.trim,
        ),
        _build_tsv_split(
            guarani_dir / "synthetic.tsv",
            split_name="guarani synthetic",
            language="guarani",
            language_code="gn",
            source_type="synthetic",
            trim=args.trim,
        ),
        _build_parallel_split(
            maya_dir / "train.es",
            maya_dir / "train.maya",
            split_name="train",
            language="maya",
            language_code="maya",
            trim=args.trim,
        ),
    ]

    validation_parts = [
        _build_parallel_split(
            wixarika_dir / "dev.es",
            wixarika_dir / "dev.hch",
            split_name="dev",
            language="wixarika",
            language_code="wix",
            trim=args.trim,
        ),
        _build_parallel_split(
            bribri_dir / "dev.es",
            bribri_dir / "dev.bzd",
            split_name="dev",
            language="bribri",
            language_code="bzd",
            trim=args.trim,
        ),
        _build_parallel_split(
            guarani_dir / "dev.es",
            guarani_dir / "dev.gn",
            split_name="dev",
            language="guarani",
            language_code="gn",
            trim=args.trim,
        ),
        _build_parallel_split(
            maya_dir / "dev.es",
            maya_dir / "dev.maya",
            split_name="dev",
            language="maya",
            language_code="maya",
            trim=args.trim,
        ),
    ]

    test_parts: list[Dataset] = []
    optional_tests = [
        (
            wixarika_dir,
            "wixarika",
            "wix",
            "hch",
        ),
        (
            bribri_dir,
            "bribri",
            "bzd",
            "bzd",
        ),
        (
            guarani_dir,
            "guarani",
            "gn",
            "gn",
        ),
        (
            maya_dir,
            "maya",
            "maya",
            "maya",
        ),
    ]
    for raw_dir, language, language_code, target_suffix in optional_tests:
        optional_split = _resolve_optional_split(
            raw_dir,
            basename_candidates=TEST_BASENAME_CANDIDATES,
            target_suffix=target_suffix,
        )
        if optional_split is None:
            continue

        es_path, target_path, basename = optional_split
        test_parts.append(
            _build_parallel_split(
                es_path,
                target_path,
                split_name=basename,
                language=language,
                language_code=language_code,
                trim=args.trim,
            )
        )

    dataset_splits = {
        "train": _concat_parts(train_parts, "train"),
        "validation": _concat_parts(validation_parts, "validation"),
    }
    if test_parts:
        dataset_splits["test"] = _concat_parts(test_parts, "test")

    dataset = DatasetDict(dataset_splits)

    _prepare_output_dir(output_dir, args.overwrite)
    dataset.save_to_disk(str(output_dir))

    print(f"Saved dataset to: {output_dir}")
    _print_summary(dataset)
    _print_random_examples(dataset)


if __name__ == "__main__":
    main()
