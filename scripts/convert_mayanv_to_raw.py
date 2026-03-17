#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


SPLITS = ("train", "dev", "test")


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

    for split_name in SPLITS:
        split_es_lines: list[str] = []
        split_target_lines: list[str] = []
        language_counts: list[tuple[str, int]] = []

        for language_code in languages:
            loaded = _load_split_for_language(
                source_root=source_root,
                language_code=language_code,
                split_name=split_name,
                trim=args.trim,
            )
            if loaded is None:
                continue

            es_lines, target_lines = loaded
            split_es_lines.extend(es_lines)
            split_target_lines.extend(target_lines)
            language_counts.append((language_code, len(es_lines)))

        _write_split(
            split_name=split_name,
            output_dir=output_dir,
            target_suffix=args.target_suffix,
            es_lines=split_es_lines,
            target_lines=split_target_lines,
        )

        if language_counts:
            counts_str = ", ".join(
                f"{language_code}={row_count}" for language_code, row_count in language_counts
            )
            print(f"{split_name}: total={len(split_es_lines)} ({counts_str})")


if __name__ == "__main__":
    main()
