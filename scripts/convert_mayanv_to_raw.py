#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


SPLITS = ("train", "dev", "test")
DEV_TEST_TOTAL_LIMIT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the MayanV directory layout into one combined raw corpus "
            "under 'maya-spanish'."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/MayanV"),
        help="Root directory containing MayanV language folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/maya-spanish"),
        help="Output directory for the combined raw corpus.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional list of language codes to include. Defaults to all languages found.",
    )
    parser.add_argument(
        "--target-suffix",
        default="maya",
        help="Target file suffix for the combined corpus (default: maya).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim leading/trailing whitespace from converted lines (default: true).",
    )
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _strip_spanish_prefix(line: str, language_code: str) -> str:
    prefix = f"#{language_code}#"
    if line.startswith(prefix):
        return line[len(prefix) :].lstrip()
    return line


def _normalize_lines(
    lines: list[str],
    *,
    trim: bool,
    strip_prefix: bool = False,
    language_code: str | None = None,
) -> list[str]:
    normalized: list[str] = []
    for line in lines:
        text = line
        if strip_prefix:
            if language_code is None:
                raise ValueError("language_code is required when strip_prefix is enabled.")
            text = _strip_spanish_prefix(text, language_code)
        if trim:
            text = text.strip()
        normalized.append(text)
    return normalized


def _is_missing_record(text: str) -> bool:
    return text == "" or text == '""'


def _is_punctuation_only_record(text: str) -> bool:
    non_space_chars = [char for char in text if not char.isspace()]
    return bool(non_space_chars) and all(not char.isalnum() for char in non_space_chars)


def _find_target_file(split_dir: Path) -> Path | None:
    candidates = sorted(path for path in split_dir.glob("data.*") if path.name != "data.es")
    if not candidates:
        return None
    if len(candidates) > 1:
        candidate_names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Expected one target file in '{split_dir}', found: {candidate_names}")
    return candidates[0]


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


def _discover_languages(source_root: Path, requested: list[str] | None) -> list[str]:
    available = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    if requested is None:
        return available

    missing = sorted(set(requested) - set(available))
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Unknown language codes under '{source_root}': {missing_str}")
    return requested


def _load_split_for_language(
    *,
    source_root: Path,
    language_code: str,
    split_name: str,
    trim: bool,
) -> tuple[list[str], list[str]] | None:
    split_dir = source_root / language_code / split_name
    es_input = split_dir / "data.es"
    target_input = _find_target_file(split_dir)

    if not es_input.exists() and target_input is None:
        return None
    if not es_input.exists() or target_input is None:
        raise FileNotFoundError(
            f"Incomplete split '{split_name}' for '{language_code}' in '{split_dir.parent}'."
        )

    expected_suffix = f".{language_code}"
    if target_input.suffix != expected_suffix:
        raise ValueError(
            f"Unexpected target suffix for '{language_code}' in '{split_dir}': "
            f"expected '{expected_suffix}', got '{target_input.suffix}'"
        )

    es_lines = _normalize_lines(
        _read_lines(es_input),
        trim=trim,
        strip_prefix=True,
        language_code=language_code,
    )
    target_lines = _normalize_lines(_read_lines(target_input), trim=trim)
    if len(es_lines) != len(target_lines):
        raise ValueError(
            f"Mismatched line counts for '{language_code}' split '{split_name}': "
            f"{es_input.name}={len(es_lines)} vs {target_input.name}={len(target_lines)}"
        )

    filtered_es_lines: list[str] = []
    filtered_target_lines: list[str] = []
    for es_line, target_line in zip(es_lines, target_lines, strict=True):
        if _is_missing_record(es_line) or _is_missing_record(target_line):
            continue
        if _is_punctuation_only_record(es_line):
            continue
        filtered_es_lines.append(es_line)
        filtered_target_lines.append(target_line)

    return filtered_es_lines, filtered_target_lines


def _collect_language_splits(
    *,
    source_root: Path,
    language_code: str,
    trim: bool,
) -> dict[str, tuple[list[str], list[str]]]:
    collected: dict[str, tuple[list[str], list[str]]] = {}
    for split_name in SPLITS:
        loaded = _load_split_for_language(
            source_root=source_root,
            language_code=language_code,
            split_name=split_name,
            trim=trim,
        )
        if loaded is not None:
            collected[split_name] = loaded
    return collected


def _slice_pair(
    pair: tuple[list[str], list[str]],
    limit: int,
) -> tuple[list[str], list[str]]:
    es_lines, target_lines = pair
    return es_lines[:limit], target_lines[:limit]


def _allocate_evenly(
    *,
    available_by_language: dict[str, int],
    total_limit: int,
    language_order: list[str],
) -> dict[str, int]:
    allocations = {language_code: 0 for language_code in available_by_language}
    remaining = min(total_limit, sum(available_by_language.values()))
    active_languages = [
        language_code
        for language_code in language_order
        if available_by_language.get(language_code, 0) > 0
    ]

    while remaining > 0 and active_languages:
        base_share, remainder = divmod(remaining, len(active_languages))
        for index, language_code in enumerate(active_languages):
            requested = base_share + (1 if index < remainder else 0)
            capacity = available_by_language[language_code] - allocations[language_code]
            granted = min(requested, capacity)
            allocations[language_code] += granted

        remaining = min(total_limit, sum(available_by_language.values())) - sum(allocations.values())
        active_languages = [
            language_code
            for language_code in active_languages
            if allocations[language_code] < available_by_language[language_code]
        ]

    return allocations


def _build_output_splits(
    *,
    source_root: Path,
    languages: list[str],
    trim: bool,
) -> tuple[
    dict[str, tuple[list[str], list[str]]],
    dict[str, list[tuple[str, int]]],
]:
    combined: dict[str, tuple[list[str], list[str]]] = {
        split_name: ([], []) for split_name in SPLITS
    }
    counts: dict[str, list[tuple[str, int]]] = {split_name: [] for split_name in SPLITS}
    language_splits_by_language = {
        language_code: _collect_language_splits(
            source_root=source_root,
            language_code=language_code,
            trim=trim,
        )
        for language_code in languages
    }
    non_itz_languages = [language_code for language_code in languages if language_code != "itz"]
    capped_split_allocations = {
        split_name: _allocate_evenly(
            available_by_language={
                language_code: len(language_splits_by_language[language_code].get(split_name, ([], []))[0])
                for language_code in non_itz_languages
            },
            total_limit=DEV_TEST_TOTAL_LIMIT,
            language_order=non_itz_languages,
        )
        for split_name in ("dev", "test")
    }

    for language_code in languages:
        language_splits = language_splits_by_language[language_code]

        if language_code == "itz":
            itz_train_es, itz_train_target = combined["train"]
            total_rows = 0
            for split_name in SPLITS:
                pair = language_splits.get(split_name)
                if pair is None:
                    continue
                es_lines, target_lines = pair
                itz_train_es.extend(es_lines)
                itz_train_target.extend(target_lines)
                total_rows += len(es_lines)
            if total_rows:
                counts["train"].append((language_code, total_rows))
            continue

        train_total = 0
        train_pair = language_splits.get("train")
        if train_pair is not None:
            train_es_lines, train_target_lines = train_pair
            combined_train_es, combined_train_target = combined["train"]
            combined_train_es.extend(train_es_lines)
            combined_train_target.extend(train_target_lines)
            train_total += len(train_es_lines)

        for split_name in SPLITS:
            if split_name == "train":
                continue
            pair = language_splits.get(split_name)
            if pair is None:
                continue
            assigned_count = capped_split_allocations[split_name].get(language_code, 0)
            es_lines, target_lines = _slice_pair(pair, assigned_count)
            remaining_es_lines, remaining_target_lines = pair[0][assigned_count:], pair[1][assigned_count:]

            split_es_lines, split_target_lines = combined[split_name]
            split_es_lines.extend(es_lines)
            split_target_lines.extend(target_lines)
            counts[split_name].append((language_code, len(es_lines)))

            if remaining_es_lines:
                combined_train_es, combined_train_target = combined["train"]
                combined_train_es.extend(remaining_es_lines)
                combined_train_target.extend(remaining_target_lines)
                train_total += len(remaining_es_lines)

        if train_total:
            counts["train"].append((language_code, train_total))

    return combined, counts


def _write_split(
    *,
    split_name: str,
    output_dir: Path,
    target_suffix: str,
    es_lines: list[str],
    target_lines: list[str],
) -> None:
    if len(es_lines) != len(target_lines):
        raise ValueError(
            f"Combined split '{split_name}' has mismatched line counts: "
            f"es={len(es_lines)} vs target={len(target_lines)}"
        )
    if not es_lines:
        return

    (output_dir / f"{split_name}.es").write_text("\n".join(es_lines) + "\n", encoding="utf-8")
    (output_dir / f"{split_name}.{target_suffix}").write_text(
        "\n".join(target_lines) + "\n",
        encoding="utf-8",
    )


def _write_readme(output_dir: Path, languages: list[str], target_suffix: str) -> None:
    readme = "\n".join(
        [
            "# maya-spanish",
            "",
            "Combined raw corpus converted from `data/raw/MayanV`.",
            "",
            "Details:",
            f"- Included languages: {', '.join(languages)}",
            f"- Target file suffix: `{target_suffix}`",
            "- Spanish source lines have their leading `#lang#` prefixes removed.",
            '- Records where either side is empty or `""` are dropped.',
            "- Records where the Spanish side is only punctuation or symbols are dropped.",
            f"- `itz` contributes all available examples to `train`, even without a train folder.",
            f"- Non-`itz` `dev` and `test` splits are each capped at {DEV_TEST_TOTAL_LIMIT} total examples.",
            "- Those non-`itz` `dev` and `test` examples are distributed as evenly as possible across languages, and leftover examples are moved to `train`.",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()

    source_root = args.source_root
    output_dir = args.output_dir
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    languages = _discover_languages(source_root, args.languages)
    if not languages:
        raise ValueError(f"No language directories found under '{source_root}'.")

    _prepare_output_dir(output_dir, overwrite=args.overwrite)
    _write_readme(output_dir, languages, args.target_suffix)

    print(f"Converting {len(languages)} languages from: {source_root}")
    print(f"Writing combined corpus to: {output_dir}")
    combined_splits, split_counts = _build_output_splits(
        source_root=source_root,
        languages=languages,
        trim=args.trim,
    )

    for split_name in SPLITS:
        split_es_lines, split_target_lines = combined_splits[split_name]
        _write_split(
            split_name=split_name,
            output_dir=output_dir,
            target_suffix=args.target_suffix,
            es_lines=split_es_lines,
            target_lines=split_target_lines,
        )

        language_counts = split_counts[split_name]
        if language_counts:
            counts_str = ", ".join(
                f"{language_code}={row_count}" for language_code, row_count in language_counts
            )
            print(f"{split_name}: total={len(split_es_lines)} ({counts_str})")


if __name__ == "__main__":
    main()
