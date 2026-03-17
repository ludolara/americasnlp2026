from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SacreBLEU evaluation on a dataset split."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="outputs/tiny-aya-base-wixarika-sft",
        help="Path to the fine-tuned model or model id.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/wixarika_spanish_hf",
        help="Path to a Hugging Face dataset saved with save_to_disk.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
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
        default="wix",
        help="Reference text column name.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="Spanish",
        help="Human-readable source language name for prompting.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="Wixarika",
        help="Human-readable target language name for prompting.",
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
        help="Batch size used for generation.",
    )
    parser.add_argument(
        "--generation-budget",
        type=int,
        default=1,
        help=(
            "Number of candidate generations to sample per example. "
            "When greater than 1, the evaluator runs best-of-n selection "
            "using sentence-level chrF++ against the reference."
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
        "--show-examples",
        action="store_true",
        help="Print a small qualitative sample of predictions after scoring.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of qualitative examples to print when --show-examples is set.",
    )
    return parser.parse_args()


def format_prompt(source: str, source_name: str, target_name: str) -> str:
    return (
        "<|user|>\n"
        f"Translate from {source_name} to {target_name}.\n\n"
        f"{source.strip()}\n"
        "<|assistant|>\n"
    )


def maybe_tqdm(
    iterable: Any,
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str | None = None,
) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, total=total, desc=desc, unit=unit)


def extract_assistant_response(text: str) -> str:
    cleaned = text.strip()
    if "<|assistant|>" in cleaned:
        cleaned = cleaned.rsplit("<|assistant|>", maxsplit=1)[-1].strip()
    if "<|user|>" in cleaned:
        cleaned = cleaned.split("<|user|>", maxsplit=1)[0].strip()
    return cleaned


def generate_prediction_candidates(
    model_name_or_path: str,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    generation_budget: int,
) -> list[list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if generation_budget < 1:
        raise ValueError("--generation-budget must be at least 1.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    candidates: list[list[str]] = [[] for _ in prompts]
    generation_tasks = [
        batch_start
        for _ in range(generation_budget)
        for batch_start in range(0, len(prompts), batch_size)
    ]
    for batch_start in maybe_tqdm(
        generation_tasks,
        total=len(generation_tasks),
        desc="Generating translations",
        unit="batch",
    ):
        prompt_batch = prompts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=generation_budget > 1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for offset, (sequence, prompt_len) in enumerate(
            zip(generated, prompt_lengths, strict=True)
        ):
            new_tokens = sequence[int(prompt_len) :].detach().cpu()
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = extract_assistant_response(text)
            candidates[batch_start + offset].append(text)

    return candidates


def build_records(
    sources: list[str],
    references: list[str],
    prediction_candidates: list[list[str]],
) -> list[dict[str, Any]]:
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)
    records: list[dict[str, Any]] = []
    for source, reference, candidates in maybe_tqdm(
        zip(sources, references, prediction_candidates, strict=True),
        total=len(prediction_candidates),
        desc="Scoring chrF++",
        unit="example",
    ):
        candidate_scores = [
            metric.sentence_score(prediction, [reference]).score
            for prediction in candidates
        ]
        best_index, best_score = max(
            enumerate(candidate_scores),
            key=lambda item: item[1],
        )
        records.append(
            {
                "source": source,
                "reference": reference,
                "prediction": candidates[best_index],
                "chrf_pp": best_score,
                "generation_budget": len(candidates),
                "selected_candidate_index": best_index,
                "candidate_predictions": candidates,
                "candidate_chrf_pp": candidate_scores,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_examples(records: list[dict[str, Any]], num_examples: int) -> None:
    count = min(num_examples, len(records))
    if count <= 0:
        return

    print(f"\nQualitative examples ({count}):")
    for index, row in enumerate(records[:count], start=1):
        print(f"\nExample {index} (chrF++: {row['chrf_pp']:.2f})")
        print(f"Source: {row['source']}")
        print(f"Reference: {row['reference']}")
        print(f"Prediction: {row['prediction']}")


def main() -> None:
    args = parse_args()

    from datasets import load_from_disk
    from sacrebleu.metrics import BLEU, CHRF

    dataset = load_from_disk(args.dataset_path)
    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Missing split '{args.split}'. Available: {available}")

    split_dataset = dataset[args.split]
    if args.limit is not None:
        split_dataset = split_dataset.select(range(min(args.limit, len(split_dataset))))
    if len(split_dataset) == 0:
        raise ValueError(f"Split '{args.split}' is empty; nothing to score.")

    missing_columns = [
        name
        for name in (args.source_column, args.target_column)
        if name not in split_dataset.column_names
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(split_dataset.column_names)
        raise ValueError(f"Missing columns: {missing}. Available: {available}")

    sources = [str(value).strip() for value in split_dataset[args.source_column]]
    references = [str(value).strip() for value in split_dataset[args.target_column]]
    prompts = [
        format_prompt(source, args.source_name, args.target_name) for source in sources
    ]

    prediction_candidates = generate_prediction_candidates(
        model_name_or_path=args.model_name_or_path,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        generation_budget=args.generation_budget,
    )

    records = build_records(sources, references, prediction_candidates)
    predictions = [row["prediction"] for row in records]
    bleu_score = BLEU().corpus_score(predictions, [references])
    chrf_score = CHRF().corpus_score(predictions, [references])
    chrf_pp_score = CHRF(word_order=2).corpus_score(predictions, [references])

    print(
        json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "dataset_path": args.dataset_path,
                "split": args.split,
                "num_examples": len(records),
                "generation_budget": args.generation_budget,
                "bleu": round(bleu_score.score, 2),
                "chrf": round(chrf_score.score, 2),
                "chrf_pp": round(chrf_pp_score.score, 2),
            },
            indent=2,
        )
    )

    if args.output_file is not None:
        write_jsonl(args.output_file, records)
        print(f"Wrote per-example results to {args.output_file}")

    if args.show_examples:
        print_examples(records, args.num_examples)


if __name__ == "__main__":
    main()
