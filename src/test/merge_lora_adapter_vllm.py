from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train.data import ensure_chat_template
from test.eval_lora import _load_adapter_base_model_name, _write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge an Aya Vision LoRA adapter into a vLLM-ready checkpoint."
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/aya-vision-32b-americas"),
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aya-vision-32b-americas-merged"),
        help="Directory where the merged model will be written.",
    )
    parser.add_argument(
        "--base-model-name-or-path",
        type=str,
        default=None,
        help="Optional explicit base model path. Defaults to adapter_config.json.",
    )
    parser.add_argument(
        "--load-dtype",
        type=str,
        default="base",
        choices=("base", "float16", "bfloat16", "float32"),
        help="Dtype used to load the base model before merge.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="base",
        choices=("base", "float16", "bfloat16", "float32"),
        help="Dtype used to save the merged model.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="device_map forwarded to Transformers when loading the base model.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="10GB",
        help="Maximum shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        help="Enable PEFT safe_merge when folding LoRA weights into the base model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def _torch_dtype_from_name(dtype_name: str):
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name == "base":
        return None
    return dtype_map[dtype_name]


def _dtype_name(dtype: Any) -> str:
    dtype_str = str(dtype)
    if dtype_str.startswith("torch."):
        return dtype_str.split(".", maxsplit=1)[1]
    return dtype_str


def _resolve_base_model_name_or_path(args: argparse.Namespace) -> str:
    if args.base_model_name_or_path:
        return args.base_model_name_or_path

    resolved = _load_adapter_base_model_name(str(args.adapter_path))
    if resolved:
        return resolved

    raise ValueError(
        "Could not resolve the base model path from the adapter. "
        "Pass --base-model-name-or-path explicitly."
    )


def _load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

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


def _runtime_asset_source(adapter_path: Path, base_model_name_or_path: str) -> str:
    adapter_assets = (
        adapter_path / "tokenizer.json",
        adapter_path / "tokenizer_config.json",
        adapter_path / "processor_config.json",
    )
    if all(path.exists() for path in adapter_assets):
        return str(adapter_path)
    return base_model_name_or_path


def main() -> None:
    args = parse_args()

    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter directory: {args.adapter_path}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite a non-empty output directory without --overwrite: "
            f"{args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name_or_path = _resolve_base_model_name_or_path(args)

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_dtype = _torch_dtype_from_name(args.load_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    if load_dtype is not None:
        model_kwargs["torch_dtype"] = load_dtype

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(args.adapter_path),
    )
    merged_model = peft_model.merge_and_unload(safe_merge=args.safe_merge)

    save_dtype = _torch_dtype_from_name(args.save_dtype)
    if save_dtype is not None:
        merged_model = merged_model.to(save_dtype)

    saved_tensor_dtypes = sorted(
        {
            _dtype_name(parameter.dtype)
            for parameter in merged_model.parameters()
        }
    )

    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    runtime_asset_source = _runtime_asset_source(
        args.adapter_path,
        base_model_name_or_path,
    )
    processor = AutoProcessor.from_pretrained(
        runtime_asset_source,
        trust_remote_code=True,
    )
    tokenizer = _load_tokenizer(runtime_asset_source)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    saved_config_dtype = None
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            saved_config_dtype = json.load(f).get("dtype")

    _write_json(
        args.output_dir / "merge_manifest.json",
        {
            "adapter_path": str(args.adapter_path.resolve()),
            "base_model_path": base_model_name_or_path,
            "output_dir": str(args.output_dir.resolve()),
            "dtype": args.load_dtype,
            "resolved_load_dtype": _dtype_name(next(base_model.parameters()).dtype),
            "save_dtype": args.save_dtype,
            "resolved_save_dtype": _dtype_name(next(merged_model.parameters()).dtype),
            "device_map": args.device_map,
            "max_shard_size": args.max_shard_size,
            "safe_merge": args.safe_merge,
            "copied_runtime_assets_from": runtime_asset_source,
            "saved_tensor_dtypes": saved_tensor_dtypes,
            "saved_config_dtype": saved_config_dtype,
        },
    )

    print(
        json.dumps(
            {
                "adapter_path": str(args.adapter_path),
                "base_model_path": base_model_name_or_path,
                "output_dir": str(args.output_dir),
                "resolved_load_dtype": _dtype_name(next(base_model.parameters()).dtype),
                "resolved_save_dtype": _dtype_name(next(merged_model.parameters()).dtype),
                "copied_runtime_assets_from": runtime_asset_source,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
