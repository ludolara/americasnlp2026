from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from train.data import format_translation_prompt
from test.eval import (
    build_per_language_summary,
    build_records,
    extract_assistant_response,
    maybe_tqdm,
    parse_language_filter,
    print_examples,
    summarize_records,
    write_jsonl,
)
from test.eval_lora import (
    _count_pending_generation_tasks,
    _load_or_initialize_candidate_store,
    _load_saved_prediction_candidates,
    _normalize_score_budgets,
    _output_file_for_budget,
    _prediction_candidates_for_budget,
    _save_example_result,
    _slugify_path_component,
    _write_json,
    _write_progress,
)


def _detect_visible_gpu_count() -> int:
    env_candidates = (
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("SLURM_STEP_GPUS"),
        os.environ.get("SLURM_JOB_GPUS"),
    )
    for raw_value in env_candidates:
        if not raw_value:
            continue
        value = raw_value.strip()
        if not value or value == "-1" or value.lower() == "nodevfiles":
            continue
        devices = [part.strip() for part in value.split(",") if part.strip()]
        if devices:
            return len(devices)

    slurm_gpus_on_node = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm_gpus_on_node:
        digits = "".join(ch for ch in slurm_gpus_on_node if ch.isdigit())
        if digits:
            return max(int(digits), 1)

    try:
        import torch
    except ModuleNotFoundError:
        return 1

    if torch.cuda.is_available():
        return max(torch.cuda.device_count(), 1)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SacreBLEU evaluation for Aya Vision merged checkpoints with vLLM."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="outputs/aya-vision-32b-americas-merged",
        help="Path to the merged model directory used by vLLM.",
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
        default=64,
        help="Number of prompts submitted to vLLM per batch.",
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
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature used by vLLM.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help=(
            "Top-k sampling constraint forwarded to vLLM. "
            "Defaulting to 1 avoids garbage Aya Vision samples on vLLM 0.18.x "
            "while preserving the requested temperature."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p sampling constraint forwarded to vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="Number of GPUs to use for tensor parallel inference. Use 0 to auto-detect all visible GPUs.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Model loading dtype used by vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory vLLM should reserve for the model and KV cache.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model context length reserved by vLLM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed forwarded to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run vLLM in eager mode for debugging or compatibility checks.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable vLLM custom all-reduce kernels for multi-GPU stability.",
    )
    parser.add_argument(
        "--language-model-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run Aya Vision in text-only mode inside vLLM. "
            "This avoids the broken multimodal sampling path for translation-only evaluation."
        ),
    )
    args = parser.parse_args()
    if args.tensor_parallel_size <= 0:
        args.tensor_parallel_size = _detect_visible_gpu_count()
    args.languages = parse_language_filter(args.languages)
    args.score_budgets = _normalize_score_budgets(
        args.score_budgets,
        args.generation_budget,
    )
    return args


def _load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer
    from train.data import ensure_chat_template

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "use_fast": True,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            fix_mistral_regex=True,
            **tokenizer_kwargs,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
    tokenizer = ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _encode_prompts(tokenizer: Any, prompts: list[str]) -> list[list[int]]:
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
    )
    return [list(token_ids) for token_ids in encoded["input_ids"]]


def _build_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "vllm",
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "languages": args.languages,
        "source_column": args.source_column,
        "target_column": args.target_column,
        "max_new_tokens": args.max_new_tokens,
        "generation_budget": args.generation_budget,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "limit": args.limit,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "language_model_only": args.language_model_only,
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
    temperature_slug = _slugify_path_component(str(args.temperature).replace(".", "p"))
    top_k_slug = _slugify_path_component(str(args.top_k))
    run_name = (
        f"{model_slug}__{dataset_slug}__{args.split}"
        f"__langs-{language_slug}__vllm__temp-{temperature_slug}__topk-{top_k_slug}"
        f"__gb-{args.generation_budget}__{digest}"
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
        with config_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
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


def _build_sampling_params(args: argparse.Namespace, tokenizer: Any):
    from vllm import SamplingParams

    params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_tokens": args.max_new_tokens,
    }
    if args.top_p is not None:
        params["top_p"] = args.top_p
    if tokenizer.eos_token_id is not None:
        params["stop_token_ids"] = [int(tokenizer.eos_token_id)]
    return SamplingParams(**params)


def _build_llm(args: argparse.Namespace):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        from vllm import LLM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vLLM is not installed in the current environment. "
            "Install it before running test.eval_lora_vllm."
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "tokenizer": args.model_name_or_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.batch_size,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    return LLM(**llm_kwargs)


def generate_prediction_candidates(
    args: argparse.Namespace,
    tokenizer: Any,
    prompts: list[str],
    prompt_token_ids: list[list[int]],
    sources: list[str],
    references: list[str],
    languages: list[str],
    run_dir: Path,
) -> list[list[str]]:
    if args.generation_budget < 1:
        raise ValueError("--generation-budget must be at least 1.")

    candidate_store = _load_or_initialize_candidate_store(
        run_dir,
        sources=sources,
        references=references,
        languages=languages,
        prompts=prompts,
        generation_budget=args.generation_budget,
    )
    pending_generation_tasks = _count_pending_generation_tasks(
        candidate_store,
        args.generation_budget,
        args.batch_size,
    )
    total_saved_candidates = sum(len(candidates) for candidates in candidate_store)
    print(
        "Inference checkpoint status: "
        + json.dumps(
            {
                "results_dir": str(run_dir),
                "saved_candidates": total_saved_candidates,
                "expected_candidates": len(candidate_store) * args.generation_budget,
                "remaining_generation_tasks": pending_generation_tasks,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if pending_generation_tasks == 0:
        _write_progress(
            run_dir,
            generation_budget=args.generation_budget,
            candidate_store=candidate_store,
            stage="generated",
        )
        return candidate_store

    _write_progress(
        run_dir,
        generation_budget=args.generation_budget,
        candidate_store=candidate_store,
        stage="generating",
    )

    llm = _build_llm(args)
    sampling_params = _build_sampling_params(args, tokenizer)
    try:
        for round_index in range(args.generation_budget):
            pending_indices = [
                example_index
                for example_index, candidates in enumerate(candidate_store)
                if len(candidates) <= round_index
            ]
            if not pending_indices:
                continue

            batch_starts = range(0, len(pending_indices), args.batch_size)
            for batch_start in maybe_tqdm(
                batch_starts,
                total=(len(pending_indices) + args.batch_size - 1) // args.batch_size,
                desc=(
                    f"Generating translations with vLLM "
                    f"(round {round_index + 1}/{args.generation_budget})"
                ),
                unit="batch",
            ):
                batch_indices = pending_indices[batch_start : batch_start + args.batch_size]
                request_batch = [
                    prompt_token_ids[example_index]
                    for example_index in batch_indices
                ]
                outputs = llm.generate(
                    request_batch,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                if len(outputs) != len(batch_indices):
                    raise RuntimeError(
                        "vLLM returned an unexpected number of outputs: "
                        f"{len(outputs)} for {len(batch_indices)} prompts."
                    )

                for output, example_index in zip(outputs, batch_indices, strict=True):
                    if not output.outputs:
                        raise RuntimeError(
                            f"vLLM returned no candidate outputs for example {example_index}."
                        )
                    text = tokenizer.decode(
                        output.outputs[0].token_ids,
                        skip_special_tokens=False,
                    )
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
                        generation_budget=args.generation_budget,
                    )

                _write_progress(
                    run_dir,
                    generation_budget=args.generation_budget,
                    candidate_store=candidate_store,
                    stage="generating",
                    last_updated_example_indices=batch_indices,
                    last_completed_generation_round=round_index + 1,
                )
    finally:
        del llm

    _write_progress(
        run_dir,
        generation_budget=args.generation_budget,
        candidate_store=candidate_store,
        stage="generated",
        last_completed_generation_round=args.generation_budget,
    )
    return candidate_store


def _write_scored_outputs(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    evaluation_results: list[dict[str, Any]],
) -> None:
    for result in evaluation_results:
        output_path = run_dir / "records.json"
        if result["candidate_budget"] != args.generation_budget:
            output_path = run_dir / f"records_top_{result['candidate_budget']}.json"
        _write_json(output_path, result["records"])

    _write_json(
        run_dir / "summary.json",
        {
            "results_dir": str(run_dir),
            "backend": "vllm",
            "model_name_or_path": args.model_name_or_path,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "generation_budget": args.generation_budget,
            "score_budgets": args.score_budgets,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "language_model_only": args.language_model_only,
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
    tokenizer = _load_tokenizer(args.model_name_or_path)
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
    prompt_token_ids = _encode_prompts(tokenizer, prompts)

    run_dir = _build_run_dir(args)
    _ensure_run_config(run_dir, args, len(prompts))

    prompt_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    print(
        "Evaluation plan: "
        + json.dumps(
            {
                "backend": "vllm",
                "split": args.split,
                "languages": args.languages,
                "limit": args.limit,
                "num_examples": len(prompts),
                "batch_size": args.batch_size,
                "generation_budget": args.generation_budget,
                "score_budgets": args.score_budgets,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "disable_custom_all_reduce": args.disable_custom_all_reduce,
                "language_model_only": args.language_model_only,
                "prompt_batches": prompt_batches,
                "expected_generation_tasks": prompt_batches * args.generation_budget,
                "results_dir": str(run_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    prediction_candidates = generate_prediction_candidates(
        args=args,
        tokenizer=tokenizer,
        prompts=prompts,
        prompt_token_ids=prompt_token_ids,
        sources=sources,
        references=references,
        languages=target_names,
        run_dir=run_dir,
    )

    saved_prediction_candidates = _load_saved_prediction_candidates(
        run_dir,
        len(prompts),
        args.generation_budget,
    )
    if prediction_candidates != saved_prediction_candidates:
        raise RuntimeError(
            "In-memory predictions did not match the checkpointed inference results."
        )

    evaluation_results: list[dict[str, Any]] = []
    for score_budget in args.score_budgets:
        budget_candidates = _prediction_candidates_for_budget(
            prediction_candidates,
            score_budget,
        )
        records = build_records(
            sources,
            references,
            target_names,
            budget_candidates,
        )
        per_language_summary = build_per_language_summary(records)
        total_summary = summarize_records(records)
        evaluation_results.append(
            {
                "candidate_budget": score_budget,
                "records": records,
                "per_language": per_language_summary,
                "total": total_summary,
            }
        )

    _write_scored_outputs(
        run_dir,
        args=args,
        evaluation_results=evaluation_results,
    )

    print(
        json.dumps(
            {
                "backend": "vllm",
                "model_name_or_path": args.model_name_or_path,
                "dataset_path": args.dataset_path,
                "split": args.split,
                "generation_budget": args.generation_budget,
                "score_budgets": args.score_budgets,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "disable_custom_all_reduce": args.disable_custom_all_reduce,
                "language_model_only": args.language_model_only,
                "per_budget": [
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

    if args.output_file is not None:
        for result in evaluation_results:
            budget_output_file = _output_file_for_budget(
                args.output_file,
                result["candidate_budget"],
                args.generation_budget,
            )
            write_jsonl(budget_output_file, result["records"])
            print(f"Wrote per-example results to {budget_output_file}")

    if args.show_examples:
        print_examples(evaluation_results[-1]["records"], args.num_examples)


if __name__ == "__main__":
    main()
