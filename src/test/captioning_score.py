from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sacrebleu.metrics import CHRF


DEFAULT_LANGUAGES = ("wixarika", "bribri", "guarani", "nahuatl")


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
        description="Score captioning predictions against labeled dev/validation captions."
    )
    parser.add_argument("--dataset-path", default="data/captioning")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--records-output-file", type=Path, default=None)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help=(
            "Language filter. Accepts submission names, ISO codes, culture names, "
            "language names, or comma-separated lists."
        ),
    )
    parser.add_argument("--heldout-percentage", type=float, default=0.1)
    parser.add_argument("--heldout-seed", type=int, default=42)
    args = parser.parse_args()
    args.languages = parse_language_filter(args.languages)
    if args.heldout_percentage is not None and not 0 < args.heldout_percentage < 1:
        raise ValueError("--heldout-percentage must be between 0 and 1.")
    return args


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
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _matches_language(example: dict[str, Any], requested: set[str] | None) -> bool:
    if requested is None:
        return True
    values = {
        str(example.get("iso_lang") or "").strip(),
        str(example.get("submission_language") or "").strip(),
        str(example.get("culture") or "").strip(),
        str(example.get("language") or "").strip(),
    }
    return bool(values & requested)


def _record_key(row: dict[str, Any]) -> str:
    language = str(row.get("submission_language") or row.get("iso_lang") or "").strip()
    return f"{language}:{row['id']}"


def _load_examples(
    *,
    dataset_path: str,
    split: str,
    languages: list[str] | None,
) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Missing split {split!r}. Available: {available}")

    split_dataset = dataset[split]
    required_columns = {
        "id",
        "filename",
        "split",
        "culture",
        "language",
        "iso_lang",
        "submission_language",
        "target_caption",
    }
    missing_columns = sorted(required_columns - set(split_dataset.column_names))
    if missing_columns:
        available = ", ".join(split_dataset.column_names)
        raise ValueError(f"Missing columns: {missing_columns}. Available: {available}")

    if "image" in split_dataset.column_names:
        split_dataset = split_dataset.remove_columns("image")

    requested = {language.strip() for language in languages} if languages else None
    examples = [
        dict(row)
        for row in split_dataset
        if str(row.get("target_caption") or "").strip()
        and _matches_language(dict(row), requested)
    ]
    if not examples:
        raise ValueError("No labeled examples matched the requested split/languages.")
    return examples


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in _read_jsonl(path):
        if "id" not in row or "predicted_caption" not in row:
            continue
        key = _record_key(row)
        if key in predictions:
            duplicates.append(key)
        predictions[key] = row
    if duplicates:
        sample = ", ".join(duplicates[:10])
        raise ValueError(f"Duplicate prediction keys in {path}: {sample}")
    return predictions


def _heldout_keys(
    examples: list[dict[str, Any]],
    *,
    percentage: float,
    seed: int,
) -> tuple[set[str], dict[str, list[str]]]:
    language_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        language_to_indices[str(example["language"]).strip()].append(index)

    rng = random.Random(seed)
    keys: set[str] = set()
    ids_by_language: dict[str, list[str]] = {}
    for language, indices in sorted(language_to_indices.items()):
        if len(indices) < 2:
            raise ValueError(f"Cannot sample held-out rows for {language!r}.")
        eval_count = max(1, int(round(len(indices) * percentage)))
        if eval_count >= len(indices):
            raise ValueError(f"Held-out percentage leaves no training rows for {language!r}.")
        sampled_indices = rng.sample(indices, k=eval_count)
        ids_by_language[language] = [str(examples[index]["id"]) for index in sampled_indices]
        keys.update(_record_key(examples[index]) for index in sampled_indices)
    return keys, ids_by_language


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric = CHRF(word_order=2)
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    sentence_scores = [
        metric.sentence_score(prediction, [reference]).score
        for prediction, reference in zip(predictions, references, strict=True)
    ]
    return {
        "num_examples": len(rows),
        "chrf_pp": round(metric.corpus_score(predictions, [references]).score, 4),
        "average_sentence_chrf_pp": round(sum(sentence_scores) / len(sentence_scores), 4),
    }


def _build_scored_rows(
    *,
    examples: list[dict[str, Any]],
    predictions_by_key: dict[str, dict[str, Any]],
    heldout_keys: set[str],
) -> list[dict[str, Any]]:
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    metric = CHRF(word_order=2)

    for example in examples:
        key = _record_key(example)
        prediction_row = predictions_by_key.get(key)
        if prediction_row is None:
            missing.append(key)
            continue

        prediction = str(prediction_row.get("predicted_caption") or "").strip()
        reference = str(example.get("target_caption") or "").strip()
        rows.append(
            {
                "key": key,
                "id": example["id"],
                "filename": example["filename"],
                "split": example["split"],
                "culture": example["culture"],
                "language": example["language"],
                "submission_language": example["submission_language"],
                "iso_lang": example["iso_lang"],
                "reference": reference,
                "prediction": prediction,
                "heldout": key in heldout_keys,
                "sentence_chrf_pp": round(
                    metric.sentence_score(prediction, [reference]).score,
                    4,
                ),
            }
        )

    if missing:
        sample = ", ".join(missing[:20])
        raise RuntimeError(f"Missing {len(missing)} predictions. Sample: {sample}")
    return rows


def _scope_summary(scope_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scope_rows:
        grouped[str(row["submission_language"])].append(row)

    per_language = []
    for submission_language, rows in sorted(grouped.items()):
        language_name = str(rows[0]["language"])
        per_language.append(
            {
                "submission_language": submission_language,
                "language": language_name,
                **_score_rows(rows),
            }
        )

    return {
        "total": _score_rows(scope_rows),
        "per_language": per_language,
    }


def main() -> None:
    args = parse_args()

    examples = _load_examples(
        dataset_path=args.dataset_path,
        split=args.split,
        languages=args.languages,
    )
    predictions_by_key = _load_predictions(args.predictions_file)
    heldout_keys, heldout_ids_by_language = _heldout_keys(
        examples,
        percentage=args.heldout_percentage,
        seed=args.heldout_seed,
    )
    scored_rows = _build_scored_rows(
        examples=examples,
        predictions_by_key=predictions_by_key,
        heldout_keys=heldout_keys,
    )

    heldout_rows = [row for row in scored_rows if row["heldout"]]
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "predictions_file": str(args.predictions_file),
        "languages": args.languages,
        "heldout_selection": {
            "percentage": args.heldout_percentage,
            "seed": args.heldout_seed,
            "ids_by_language": heldout_ids_by_language,
        },
        "full_dev": _scope_summary(scored_rows),
        "heldout": _scope_summary(heldout_rows),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if args.records_output_file is not None:
        _write_jsonl(args.records_output_file, scored_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
