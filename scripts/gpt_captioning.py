#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


SUBMISSION_COLUMNS = ("id", "filename", "split", "culture", "language", "iso_lang")
DEFAULT_LANGUAGES = ("bribri", "guarani", "nahuatl", "wixarika")
DEFAULT_PROMPT_TEMPLATE = (
    "Look at the image and write one concise, culturally appropriate caption "
    "directly in {language} (ISO {iso_lang}). Do not write in Spanish "
    "or English. Do not include labels, explanations, markdown, quotation marks, "
    "or translations. Return only the final caption in {language}."
)
DEFAULT_INSTRUCTIONS = (
    "You create image captions for an AmericasNLP submission. The target language "
    "is specified in the user prompt. Answer with exactly one caption and no "
    "surrounding text."
)
VALID_IMAGE_DETAILS = {"low", "high", "auto"}


class EmptyCaptionResponseError(RuntimeError):
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__("OpenAI response did not contain caption text.")
        self.response = response


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _parse_languages(raw_languages: list[str] | None, raw_language: str | None) -> list[str]:
    if raw_languages is None and raw_language:
        raw_languages = [raw_language]
    if raw_languages is None:
        env_languages = os.environ.get("LANGUAGES")
        raw_languages = [env_languages] if env_languages else None
    if raw_languages is None and os.environ.get("LANGUAGE"):
        raw_languages = [os.environ["LANGUAGE"]]
    if raw_languages is None:
        raw_languages = list(DEFAULT_LANGUAGES)

    languages: list[str] = []
    for raw in raw_languages:
        for language in raw.split(","):
            cleaned = language.strip()
            if cleaned and cleaned not in languages:
                languages.append(cleaned)
    if not languages:
        raise ValueError("At least one language must be provided.")
    return languages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption image test sets with the OpenAI Responses API."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path(os.environ.get("RAW_DIR", "data/raw/captioning")))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "test"))
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to caption. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument("--version", type=int, default=_env_int("VERSION", 0))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "results/v0_gpt_captioning")))
    parser.add_argument("--output-file", type=Path, default=Path(os.environ["OUTPUT_FILE"]) if os.environ.get("OUTPUT_FILE") else None)
    parser.add_argument("--checkpoint-file", type=Path, default=Path(os.environ["CHECKPOINT_FILE"]) if os.environ.get("CHECKPOINT_FILE") else None)
    parser.add_argument("--error-file", type=Path, default=Path(os.environ["ERROR_FILE"]) if os.environ.get("ERROR_FILE") else None)
    parser.add_argument("--zip-file", type=Path, default=Path(os.environ["ZIP_FILE"]) if os.environ.get("ZIP_FILE") else None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("REASONING_EFFORT", "medium"))
    parser.add_argument("--image-detail", default=os.environ.get("IMAGE_DETAIL", "high"))
    parser.add_argument("--max-output-tokens", type=int, default=_env_int("MAX_OUTPUT_TOKENS", 4096))
    parser.add_argument("--request-timeout", type=int, default=_env_int("REQUEST_TIMEOUT", 300))
    parser.add_argument("--max-retries", type=int, default=_env_int("MAX_RETRIES", 8))
    parser.add_argument("--retry-base-seconds", type=int, default=_env_int("RETRY_BASE_SECONDS", 2))
    parser.add_argument("--limit", type=int, default=_env_int("LIMIT", 0) or None)
    parser.add_argument("--prompt-template", default=os.environ.get("PROMPT_TEMPLATE") or DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--store", action=argparse.BooleanOptionalAction, default=_env_bool("STORE", False))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=_env_bool("DRY_RUN", False))
    args = parser.parse_args()
    args.languages = _parse_languages(args.languages, args.language)

    if args.version < 0 or args.version > 9:
        raise ValueError("--version must be a single digit from 0 to 9.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when set.")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive.")
    if args.image_detail == "original":
        print("IMAGE_DETAIL=original is not valid for Responses API; using high.", file=sys.stderr)
        args.image_detail = "high"
    if args.image_detail not in VALID_IMAGE_DETAILS:
        valid = ", ".join(sorted(VALID_IMAGE_DETAILS))
        raise ValueError(f"--image-detail must be one of: {valid}.")
    if len(args.languages) > 1:
        explicit_file_args = [
            name
            for name, value in (
                ("--output-file", args.output_file),
                ("--checkpoint-file", args.checkpoint_file),
                ("--error-file", args.error_file),
            )
            if value is not None
        ]
        if explicit_file_args:
            joined = ", ".join(explicit_file_args)
            raise ValueError(f"{joined} can only be used with one language.")
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
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_image_path(language_dir: Path, filename: str) -> Path:
    raw_path = Path(filename)
    candidates = [raw_path] if raw_path.is_absolute() else [
        language_dir / raw_path,
        language_dir / "images" / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve image {filename!r}. Checked: {checked}")


def _clean_caption(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        r"^(predicted_caption|caption|pie de foto|bribri|guarani|nahuatl|wixarika|maya|yucatec maya)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _record_key(row: dict[str, Any]) -> str:
    return str(row["id"])


def _submission_record(row: dict[str, Any], caption: str) -> dict[str, Any]:
    return {**{column: row[column] for column in SUBMISSION_COLUMNS}, "predicted_caption": caption}


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if "id" in record and "predicted_caption" in record:
            record["predicted_caption"] = _clean_caption(str(record["predicted_caption"]))
            records[str(record["id"])] = record
    return records


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _response_debug_summary(response: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "id": response.get("id"),
        "status": response.get("status"),
        "model": response.get("model"),
        "incomplete_details": response.get("incomplete_details"),
        "error": response.get("error"),
        "usage": response.get("usage"),
        "output": [],
    }
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        content_types = []
        for content in item.get("content", []):
            if isinstance(content, dict):
                content_types.append(content.get("type"))
        summary["output"].append(
            {
                "type": item.get("type"),
                "status": item.get("status"),
                "role": item.get("role"),
                "content_types": content_types,
            }
        )
    return summary


def _request_caption(
    *,
    api_key: str,
    args: argparse.Namespace,
    prompt: str,
    image_path: Path,
) -> str:
    payload: dict[str, Any] = {
        "model": args.model,
        "instructions": DEFAULT_INSTRUCTIONS,
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "store": args.store,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image_path),
                        "detail": args.image_detail,
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        body = response.read().decode("utf-8")

    response_json = json.loads(body)
    caption = _clean_caption(_extract_output_text(response_json))
    if not caption:
        raise EmptyCaptionResponseError(response_json)
    return caption


def _request_caption_with_retries(
    *,
    api_key: str,
    args: argparse.Namespace,
    prompt: str,
    image_path: Path,
) -> str:
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(args.max_retries + 1):
        try:
            return _request_caption(api_key=api_key, args=args, prompt=prompt, image_path=image_path)
        except EmptyCaptionResponseError as exc:
            if attempt >= args.max_retries:
                raise
            delay = args.retry_base_seconds * (2 ** attempt)
            summary = _response_debug_summary(exc.response)
            print(
                f"Retrying after empty response in {delay}s: {image_path} "
                f"{json.dumps(summary, ensure_ascii=False)}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in retryable_statuses or attempt >= args.max_retries:
                raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else args.retry_base_seconds * (2 ** attempt)
            print(f"Retrying after HTTP {exc.code} in {delay}s: {image_path}", file=sys.stderr, flush=True)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt >= args.max_retries:
                raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
            delay = args.retry_base_seconds * (2 ** attempt)
            print(f"Retrying after request error in {delay}s: {exc}", file=sys.stderr, flush=True)
            time.sleep(delay)

    raise AssertionError("unreachable")


def _write_zip(zip_path: Path, output_files: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_name(f"{zip_path.name}.tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for output_file in output_files:
            zf.write(output_file, arcname=output_file.name)
    tmp_path.replace(zip_path)


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        missing = [column for column in SUBMISSION_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Row {row.get('id')} is missing columns: {', '.join(missing)}")


def _language_paths(
    args: argparse.Namespace,
    language: str,
) -> tuple[Path, Path, Path, Path, Path]:
    language_dir = args.raw_dir / args.split / language
    input_file = language_dir / f"{language}.jsonl"
    output_file = args.output_file or args.output_dir / "submission" / f"{language}-{args.version}.jsonl"
    checkpoint_file = args.checkpoint_file or args.output_dir / f"{language}-{args.version}.checkpoint.jsonl"
    error_file = args.error_file or args.output_dir / f"{language}-{args.version}.errors.jsonl"
    return language_dir, input_file, output_file, checkpoint_file, error_file


def _caption_language(
    *,
    args: argparse.Namespace,
    api_key: str,
    language: str,
) -> dict[str, Any]:
    language_dir, input_file, output_file, checkpoint_file, error_file = _language_paths(args, language)
    rows = _read_jsonl(input_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows selected from {input_file}")
    _validate_rows(rows)

    records_by_key = _load_completed(checkpoint_file)
    for key, record in _load_completed(output_file).items():
        records_by_key.setdefault(key, record)

    pending = [row for row in rows if _record_key(row) not in records_by_key]
    print(
        json.dumps(
            {
                "language": language,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "image_detail": args.image_detail,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "checkpoint_file": str(checkpoint_file),
                "error_file": str(error_file),
                "num_examples": len(rows),
                "completed": len(rows) - len(pending),
                "pending": len(pending),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    if pending and not args.dry_run and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run or DRY_RUN=1 is set.")

    failures: list[str] = []
    for index, row in enumerate(pending, start=1):
        image_path = _resolve_image_path(language_dir, str(row["filename"]))
        prompt = args.prompt_template.format(**row)
        print(f"[{index}/{len(pending)}] captioning {row['id']} from {image_path}", file=sys.stderr, flush=True)
        try:
            caption = (
                f"DRY_RUN_CAPTION_FOR_{row['id']}"
                if args.dry_run
                else _request_caption_with_retries(api_key=api_key, args=args, prompt=prompt, image_path=image_path)
            )
        except Exception as exc:
            failures.append(_record_key(row))
            error_record: dict[str, Any] = {
                "id": row["id"],
                "filename": row["filename"],
                "split": row["split"],
                "culture": row["culture"],
                "language": row["language"],
                "iso_lang": row["iso_lang"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            if isinstance(exc, EmptyCaptionResponseError):
                error_record["response"] = _response_debug_summary(exc.response)
            _append_jsonl(error_file, error_record)
            print(f"Failed {row['id']}; logged details to {error_file}", file=sys.stderr, flush=True)
            continue

        records_by_key[_record_key(row)] = _submission_record(row, caption)
        completed_rows = [records_by_key[_record_key(done)] for done in rows if _record_key(done) in records_by_key]
        _write_jsonl(checkpoint_file, completed_rows)

    missing = [_record_key(row) for row in rows if _record_key(row) not in records_by_key]
    if missing:
        return {
            "language": language,
            "num_examples": len(rows),
            "submission_file": None,
            "checkpoint_file": str(checkpoint_file),
            "error_file": str(error_file),
            "missing": missing,
            "failures": failures,
        }

    output_rows = [records_by_key[_record_key(row)] for row in rows]
    _write_jsonl(output_file, output_rows)
    return {
        "language": language,
        "num_examples": len(output_rows),
        "submission_file": str(output_file),
        "checkpoint_file": str(checkpoint_file),
        "error_file": str(error_file),
        "missing": [],
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")

    results: list[dict[str, Any]] = []
    for language in args.languages:
        try:
            results.append(_caption_language(args=args, api_key=api_key, language=language))
        except Exception as exc:
            results.append(
                {
                    "language": language,
                    "num_examples": 0,
                    "submission_file": None,
                    "checkpoint_file": None,
                    "error_file": None,
                    "missing": ["<language failed before row processing>"],
                    "failures": [f"{type(exc).__name__}: {exc}"],
                }
            )
            print(f"Failed language {language}: {exc}", file=sys.stderr, flush=True)

    missing_results = [result for result in results if result["missing"]]
    if missing_results:
        raise RuntimeError(
            "Missing captions after generation: "
            + json.dumps(missing_results, ensure_ascii=False)
        )

    output_files = [Path(result["submission_file"]) for result in results if result["submission_file"]]
    if args.zip_file is not None:
        _write_zip(args.zip_file, output_files)

    print(
        json.dumps(
            {
                "languages": args.languages,
                "num_examples": sum(int(result["num_examples"]) for result in results),
                "submission_files": [str(path) for path in output_files],
                "checkpoint_files": [str(result["checkpoint_file"]) for result in results],
                "zip_file": str(args.zip_file) if args.zip_file is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
