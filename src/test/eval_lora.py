from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from train.data import ensure_chat_template, format_translation_prompt
from test.eval import (
    build_per_language_summary,
    build_records,
    extract_assistant_response,
    parse_language_filter,
    print_examples,
    summarize_records,
    write_jsonl,
)

EVAL_PROMPT_VERSION = "translation_prompt_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SacreBLEU evaluation for Aya Vision LoRA checkpoints."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="outputs/aya-vision-32b-americas",
        help="Path to the LoRA checkpoint, adapter directory, or merged model.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/americasnlp2026",
        help="Path to a Hugging Face dataset saved with save_to_disk.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of target languages to evaluate. Accepts space-separated "
            "values, comma-separated values, or both."
        ),
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default="es",
        help="Source text column name.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
        help="Reference text column name.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per example.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for generation.",
    )
    parser.add_argument(
        "--generation-budget",
        type=int,
        default=10,
        help=(
            "Number of candidate generations to sample per example. "
            "When greater than 1, the evaluator runs best-of-n selection "
            "using sentence-level chrF++ against the reference."
        ),
    )
    parser.add_argument(
        "--score-budgets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional candidate prefix sizes to score using the first N generations "
            "from each example. Defaults to the full generation budget and also 10 "
            "when generation_budget is at least 10."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples to score.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSONL path for per-example predictions and sentence-level chrF++ scores.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory where checkpointed inference JSON files and scores are saved.",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print a small qualitative sample of predictions after scoring.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of qualitative examples to print when --show-examples is set.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Model loading dtype used for Aya Vision inference.",
    )
    args = parser.parse_args()
    args.languages = parse_language_filter(args.languages)
    args.score_budgets = _normalize_score_budgets(
        args.score_budgets,
        args.generation_budget,
    )
    return args


def _normalize_score_budgets(
    raw_score_budgets: list[int] | None,
    generation_budget: int,
) -> list[int]:
    if generation_budget < 1:
        raise ValueError("--generation-budget must be at least 1.")

    if raw_score_budgets is None:
        raw_score_budgets = [generation_budget]
        if generation_budget > 10:
            raw_score_budgets.insert(0, 10)

    normalized_budgets: list[int] = []
    for budget in raw_score_budgets:
        if budget < 1:
            raise ValueError("--score-budgets values must be at least 1.")
        if budget > generation_budget:
            raise ValueError(
                "--score-budgets values cannot exceed --generation-budget."
            )
        if budget not in normalized_budgets:
            normalized_budgets.append(budget)

    return normalized_budgets


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
        f"Adapter at {adapter_path} contains zero tensors. Refusing to evaluate an "
        "empty LoRA export because that silently falls back to base-model behavior."
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


def _slugify_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "value"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _model_fingerprint(model_name_or_path: str) -> dict[str, Any]:
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        return {"path": model_name_or_path, "type": "external"}

    fingerprint_files = [
        model_path / "adapter_config.json",
        model_path / "adapter_model.safetensors",
    ]
    existing_files = [path for path in fingerprint_files if path.exists()]
    if not existing_files:
        existing_files = [
            path
            for path in (
                model_path / "config.json",
                model_path / "generation_config.json",
                model_path / "model.safetensors",
            )
            if path.exists()
        ]

    return {
        "path": str(model_path),
        "files": [_file_fingerprint(path) for path in existing_files],
    }


def _get_model_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_model_fingerprint", None)
    if cached is None:
        cached = _model_fingerprint(args.model_name_or_path)
        setattr(args, "_model_fingerprint", cached)
    return cached


def _build_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name_or_path": args.model_name_or_path,
        "model_fingerprint": _get_model_fingerprint(args),
        "dataset_path": args.dataset_path,
        "split": args.split,
        "languages": args.languages,
        "source_column": args.source_column,
        "target_column": args.target_column,
        "max_new_tokens": args.max_new_tokens,
        "generation_budget": args.generation_budget,
        "dtype": args.dtype,
        "limit": args.limit,
        "prompt_version": EVAL_PROMPT_VERSION,
    }


def _build_run_dir(args: argparse.Namespace) -> Path:
    run_identity = _build_run_identity(args)
    digest = hashlib.sha1(
        json.dumps(run_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    model_slug = _slugify_path_component(
        Path(args.model_name_or_path).name or args.model_name_or_path
    )
    dataset_slug = _slugify_path_component(
        Path(args.dataset_path).name or args.dataset_path
    )
    language_slug = _slugify_path_component(
        "-".join(args.languages) if args.languages else "all"
    )
    run_name = (
        f"{model_slug}__{dataset_slug}__{args.split}"
        f"__langs-{language_slug}__gb-{args.generation_budget}__{digest}"
    )
    return args.results_dir / run_name


def _ensure_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    num_examples: int,
) -> None:
    config_path = run_dir / "config.json"
    run_identity = _build_run_identity(args)
    config_payload = {
        "run_identity": run_identity,
        "results_dir": str(run_dir),
        "num_examples": num_examples,
    }

    if config_path.exists():
        existing = _read_json(config_path)
        if existing.get("run_identity") != run_identity:
            raise ValueError(
                "Existing results directory does not match the requested evaluation "
                f"arguments: {run_dir}"
            )
        if int(existing.get("num_examples", num_examples)) != num_examples:
            raise ValueError(
                "Existing results directory was created for a different dataset slice: "
                f"{run_dir}"
            )
        return

    _write_json(config_path, config_payload)


def _example_result_path(run_dir: Path, example_index: int) -> Path:
    return run_dir / "inference" / f"{example_index}.json"


def _prediction_candidates_for_budget(
    prediction_candidates: list[list[str]],
    score_budget: int,
) -> list[list[str]]:
    return [candidates[:score_budget] for candidates in prediction_candidates]


def _records_output_path(run_dir: Path, score_budget: int, full_budget: int) -> Path:
    if score_budget == full_budget:
        return run_dir / "records.json"
    return run_dir / f"records_top_{score_budget}.json"


def _output_file_for_budget(
    output_file: Path,
    score_budget: int,
    full_budget: int,
) -> Path:
    if score_budget == full_budget:
        return output_file
    return output_file.with_name(
        f"{output_file.stem}_top_{score_budget}{output_file.suffix}"
    )


def _build_example_payload(
    *,
    example_index: int,
    source: str,
    reference: str,
    language: str,
    prompt: str,
    candidates: list[str],
    generation_budget: int,
) -> dict[str, Any]:
    return {
        "example_index": example_index,
        "source": source,
        "reference": reference,
        "language": language,
        "prompt": prompt,
        "candidate_predictions": candidates,
        "num_saved_candidates": len(candidates),
        "generation_budget": generation_budget,
        "completed": len(candidates) >= generation_budget,
    }


def _save_example_result(
    run_dir: Path,
    *,
    example_index: int,
    source: str,
    reference: str,
    language: str,
    prompt: str,
    candidates: list[str],
    generation_budget: int,
) -> None:
    payload = _build_example_payload(
        example_index=example_index,
        source=source,
        reference=reference,
        language=language,
        prompt=prompt,
        candidates=candidates,
        generation_budget=generation_budget,
    )
    _write_json(_example_result_path(run_dir, example_index), payload)


def _load_or_initialize_candidate_store(
    run_dir: Path,
    *,
    sources: list[str],
    references: list[str],
    languages: list[str],
    prompts: list[str],
    generation_budget: int,
) -> list[list[str]]:
    candidate_store: list[list[str]] = []
    for example_index, (source, reference, language, prompt) in enumerate(
        zip(sources, references, languages, prompts, strict=True)
    ):
        example_path = _example_result_path(run_dir, example_index)
        if not example_path.exists():
            _save_example_result(
                run_dir,
                example_index=example_index,
                source=source,
                reference=reference,
                language=language,
                prompt=prompt,
                candidates=[],
                generation_budget=generation_budget,
            )
            candidate_store.append([])
            continue

        payload = _read_json(example_path)
        expected_fields = {
            "example_index": example_index,
            "source": source,
            "reference": reference,
            "language": language,
            "prompt": prompt,
        }
        for field_name, expected_value in expected_fields.items():
            actual_value = payload.get(field_name)
            if actual_value != expected_value:
                raise ValueError(
                    f"Saved inference file {example_path} has unexpected {field_name!r}."
                )

        raw_candidates = payload.get("candidate_predictions", [])
        if not isinstance(raw_candidates, list):
            raise ValueError(
                f"Saved inference file {example_path} must contain a list of candidates."
            )

        candidates = [str(value) for value in raw_candidates]
        if len(candidates) > generation_budget:
            raise ValueError(
                f"Saved inference file {example_path} has more than "
                f"{generation_budget} candidates."
            )
        candidate_store.append(candidates)

    return candidate_store


def _count_pending_generation_tasks(
    candidate_store: list[list[str]],
    generation_budget: int,
    batch_size: int,
) -> int:
    total_tasks = 0
    for round_index in range(generation_budget):
        pending_examples = sum(1 for candidates in candidate_store if len(candidates) <= round_index)
        if pending_examples > 0:
            total_tasks += (pending_examples + batch_size - 1) // batch_size
    return total_tasks


def _write_progress(
    run_dir: Path,
    *,
    generation_budget: int,
    candidate_store: list[list[str]],
    stage: str,
    last_updated_example_indices: list[int] | None = None,
    last_completed_generation_round: int | None = None,
) -> None:
    total_saved_candidates = sum(len(candidates) for candidates in candidate_store)
    num_examples = len(candidate_store)
    completed_examples = sum(
        1 for candidates in candidate_store if len(candidates) >= generation_budget
    )
    payload = {
        "stage": stage,
        "num_examples": num_examples,
        "generation_budget": generation_budget,
        "completed_examples": completed_examples,
        "pending_examples": num_examples - completed_examples,
        "total_saved_candidates": total_saved_candidates,
        "total_expected_candidates": num_examples * generation_budget,
        "last_completed_generation_round": last_completed_generation_round,
        "last_updated_example_index": (
            last_updated_example_indices[-1] if last_updated_example_indices else None
        ),
        "last_updated_example_indices": last_updated_example_indices,
    }
    _write_json(run_dir / "progress.json", payload)


def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer = ensure_chat_template(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(model_name_or_path: str, dtype_name: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model_dtype = _resolve_model_dtype(dtype_name)
    adapter_base_model_name = _load_adapter_base_model_name(model_name_or_path)
    load_path = adapter_base_model_name or model_name_or_path

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": model_dtype,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        load_path,
        **model_kwargs,
    )
    if adapter_base_model_name is not None:
        _assert_adapter_has_weights(model_name_or_path)
        model = PeftModel.from_pretrained(
            model,
            model_name_or_path,
        )
    elif not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.eval()
    return model


def generate_prediction_candidates(
    model_name_or_path: str,
    tokenizer: Any,
    prompts: list[str],
    sources: list[str],
    references: list[str],
    languages: list[str],
    batch_size: int,
    max_new_tokens: int,
    generation_budget: int,
    dtype_name: str,
    run_dir: Path,
) -> list[list[str]]:
    import torch

    if generation_budget < 1:
        raise ValueError("--generation-budget must be at least 1.")

    candidate_store = _load_or_initialize_candidate_store(
        run_dir,
        sources=sources,
        references=references,
        languages=languages,
        prompts=prompts,
        generation_budget=generation_budget,
    )
    pending_generation_tasks = _count_pending_generation_tasks(
        candidate_store,
        generation_budget,
        batch_size,
    )
    total_saved_candidates = sum(len(candidates) for candidates in candidate_store)
    print(
        "Inference checkpoint status: "
        + json.dumps(
            {
                "results_dir": str(run_dir),
                "saved_candidates": total_saved_candidates,
                "expected_candidates": len(candidate_store) * generation_budget,
                "remaining_generation_tasks": pending_generation_tasks,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    if pending_generation_tasks == 0:
        _write_progress(
            run_dir,
            generation_budget=generation_budget,
            candidate_store=candidate_store,
            stage="generated",
        )
        return candidate_store

    _write_progress(
        run_dir,
        generation_budget=generation_budget,
        candidate_store=candidate_store,
        stage="generating",
    )

    model = load_lora_model(model_name_or_path, dtype_name)
    input_device = _get_input_device(model)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=pending_generation_tasks,
            desc="Generating translations",
            unit="batch",
        )

    try:
        for round_index in range(generation_budget):
            pending_indices = [
                example_index
                for example_index, candidates in enumerate(candidate_store)
                if len(candidates) <= round_index
            ]
            if not pending_indices:
                continue

            batch_starts = range(0, len(pending_indices), batch_size)
            for batch_start in batch_starts:
                batch_indices = pending_indices[batch_start : batch_start + batch_size]
                prompt_batch = [prompts[example_index] for example_index in batch_indices]
                encoded = tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=generation_budget > 1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_width = encoded["input_ids"].shape[1]
                for offset, sequence in enumerate(generated):
                    example_index = batch_indices[offset]
                    new_tokens = sequence[prompt_width:].detach().cpu()
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    text = extract_assistant_response(text)
                    candidate_store[example_index].append(text)
                    _save_example_result(
                        run_dir,
                        example_index=example_index,
                        source=sources[example_index],
                        reference=references[example_index],
                        language=languages[example_index],
                        prompt=prompts[example_index],
                        candidates=candidate_store[example_index],
                        generation_budget=generation_budget,
                    )

                _write_progress(
                    run_dir,
                    generation_budget=generation_budget,
                    candidate_store=candidate_store,
                    stage="generating",
                    last_updated_example_indices=batch_indices,
                    last_completed_generation_round=round_index + 1,
                )
                if progress_bar is not None:
                    progress_bar.set_postfix_str(
                        f"round {round_index + 1}/{generation_budget}"
                    )
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _write_progress(
        run_dir,
        generation_budget=generation_budget,
        candidate_store=candidate_store,
        stage="generated",
        last_completed_generation_round=generation_budget,
    )
    return candidate_store


def _load_saved_prediction_candidates(
    run_dir: Path,
    num_examples: int,
    generation_budget: int,
) -> list[list[str]]:
    prediction_candidates: list[list[str]] = []
    for example_index in range(num_examples):
        example_path = _example_result_path(run_dir, example_index)
        if not example_path.exists():
            raise ValueError(f"Missing saved inference file: {example_path}")

        payload = _read_json(example_path)
        raw_candidates = payload.get("candidate_predictions", [])
        if not isinstance(raw_candidates, list):
            raise ValueError(
                f"Saved inference file {example_path} must contain a list of candidates."
            )

        candidates = [str(value) for value in raw_candidates]
        if len(candidates) != generation_budget:
            raise ValueError(
                f"Saved inference file {example_path} has {len(candidates)} candidates; "
                f"expected {generation_budget}."
            )
        prediction_candidates.append(candidates)

    return prediction_candidates


def _write_scored_outputs(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    evaluation_results: list[dict[str, Any]],
) -> None:
    for result in evaluation_results:
        _write_json(
            _records_output_path(
                run_dir,
                result["candidate_budget"],
                args.generation_budget,
            ),
            result["records"],
        )

    _write_json(
        run_dir / "summary.json",
        {
            "results_dir": str(run_dir),
            "model_name_or_path": args.model_name_or_path,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "generation_budget": args.generation_budget,
            "score_budgets": args.score_budgets,
            "dtype": args.dtype,
            "evaluations": [
                {
                    "candidate_budget": result["candidate_budget"],
                    "per_language": result["per_language"],
                    "total": result["total"],
                }
                for result in evaluation_results
            ],
        },
    )


def main() -> None:
    args = parse_args()

    from datasets import load_from_disk

    dataset = load_from_disk(args.dataset_path)
    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Missing split '{args.split}'. Available: {available}")

    split_dataset = dataset[args.split]
    missing_columns = [
        name
        for name in (args.source_column, args.target_column, "language")
        if name not in split_dataset.column_names
    ]
    if args.languages is not None and "language_code" not in split_dataset.column_names:
        missing_columns.append("language_code")
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(split_dataset.column_names)
        raise ValueError(f"Missing columns: {missing}. Available: {available}")

    if args.languages is not None:
        requested_languages = set(args.languages)
        available_languages = {
            str(value).strip()
            for value in split_dataset["language_code"]
            if str(value).strip()
        }
        missing_languages = sorted(requested_languages - available_languages)
        if missing_languages:
            missing = ", ".join(missing_languages)
            available = ", ".join(sorted(available_languages))
            raise ValueError(
                f"Requested languages not found in split '{args.split}': {missing}. "
                f"Available: {available}"
            )

        selected_indices = [
            index
            for index, value in enumerate(split_dataset["language_code"])
            if str(value).strip() in requested_languages
        ]
        split_dataset = split_dataset.select(selected_indices)

    if args.limit is not None:
        split_dataset = split_dataset.select(range(min(args.limit, len(split_dataset))))
    if len(split_dataset) == 0:
        raise ValueError(f"Split '{args.split}' is empty; nothing to score.")

    sources = [str(value).strip() for value in split_dataset[args.source_column]]
    references = [str(value).strip() for value in split_dataset[args.target_column]]
    tokenizer = load_tokenizer(args.model_name_or_path)
    target_names = [str(value).strip() for value in split_dataset["language"]]
    if any(not target_name for target_name in target_names):
        raise ValueError("All rows must include a non-empty `language` value.")

    prompts = [
        format_translation_prompt(
            source,
            target_name,
            tokenizer,
        )
        for source, target_name in zip(sources, target_names, strict=True)
    ]

    run_dir = _build_run_dir(args)
    _ensure_run_config(run_dir, args, len(prompts))

    prompt_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    print(
        "Evaluation plan: "
        + json.dumps(
            {
                "split": args.split,
                "languages": args.languages,
                "limit": args.limit,
                "num_examples": len(prompts),
                "batch_size": args.batch_size,
                "generation_budget": args.generation_budget,
                "score_budgets": args.score_budgets,
                "prompt_batches": prompt_batches,
                "expected_generation_tasks": prompt_batches * args.generation_budget,
                "results_dir": str(run_dir),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    generate_prediction_candidates(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        prompts=prompts,
        sources=sources,
        references=references,
        languages=target_names,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        generation_budget=args.generation_budget,
        dtype_name=args.dtype,
        run_dir=run_dir,
    )

    prediction_candidates = _load_saved_prediction_candidates(
        run_dir,
        len(prompts),
        args.generation_budget,
    )
    _write_progress(
        run_dir,
        generation_budget=args.generation_budget,
        candidate_store=prediction_candidates,
        stage="scoring",
        last_completed_generation_round=args.generation_budget,
    )

    evaluation_results: list[dict[str, Any]] = []
    for score_budget in args.score_budgets:
        budget_prediction_candidates = _prediction_candidates_for_budget(
            prediction_candidates,
            score_budget,
        )
        records = build_records(
            sources,
            references,
            target_names,
            budget_prediction_candidates,
        )
        evaluation_results.append(
            {
                "candidate_budget": score_budget,
                "records": records,
                "per_language": build_per_language_summary(records),
                "total": summarize_records(records),
            }
        )
    evaluation_results.sort(key=lambda result: int(result["candidate_budget"]))

    _write_scored_outputs(
        run_dir,
        args=args,
        evaluation_results=evaluation_results,
    )
    _write_progress(
        run_dir,
        generation_budget=args.generation_budget,
        candidate_store=prediction_candidates,
        stage="complete",
        last_completed_generation_round=args.generation_budget,
    )

    print(
        json.dumps(
            {
                "results_dir": str(run_dir),
                "model_name_or_path": args.model_name_or_path,
                "dataset_path": args.dataset_path,
                "split": args.split,
                "generation_budget": args.generation_budget,
                "score_budgets": args.score_budgets,
                "dtype": args.dtype,
                "evaluations": [
                    {
                        "candidate_budget": result["candidate_budget"],
                        "per_language": result["per_language"],
                        "total": result["total"],
                    }
                    for result in evaluation_results
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nEvaluation summary by candidate budget:")
    for result in evaluation_results:
        total = result["total"]
        print(
            f"top-{result['candidate_budget']}: "
            f"BLEU={total['bleu']:.2f} "
            f"chrF={total['chrf']:.2f} "
            f"chrF++={total['chrf_pp']:.2f}"
        )

    if args.output_file is not None:
        for result in evaluation_results:
            output_path = _output_file_for_budget(
                args.output_file,
                result["candidate_budget"],
                args.generation_budget,
            )
            write_jsonl(output_path, result["records"])
            print(f"Wrote per-example results to {output_path}")

    print(f"Wrote checkpointed inference and scores to {run_dir}")

    if args.show_examples:
        print_examples(
            max(
                evaluation_results,
                key=lambda result: int(result["candidate_budget"]),
            )["records"],
            args.num_examples,
        )


if __name__ == "__main__":
    main()
