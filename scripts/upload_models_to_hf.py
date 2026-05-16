#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


DEFAULT_MODEL_NAMES = (
    "aya-vision-32b-americas",
    "aya-vision-32b-americas-captioning",
    "aya-vision-32b-americas-grpo-captioning",
)
DEFAULT_BASE_MODEL_ID = "CohereLabs/aya-vision-32b"
REQUIRED_MODEL_FILES = ("README.md", "adapter_config.json", "adapter_model.safetensors")


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the final LoRA adapter artifacts under outputs/ to Hugging Face model repos."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing the trained model output folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
        help="Output folder names to upload. Defaults to the three Aya Vision Americas adapters.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Hugging Face user or organization namespace. Defaults to the authenticated account name.",
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help=(
            "Hub model id written into the staged adapter metadata before upload. "
            f"Defaults to {DEFAULT_BASE_MODEL_ID!r}."
        ),
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also upload checkpoint-* subdirectories. By default only final adapter artifacts are uploaded.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Create new repos as private or public. Omit to use the Hub default.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload trained adapter",
        help="Commit message used for each Hub upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned uploads without creating repos or uploading files.",
    )
    return parser.parse_args()


def _resolve_namespace(api: HfApi, namespace: str | None) -> str:
    if namespace:
        return namespace

    try:
        account = api.whoami()
    except Exception as exc:  # noqa: BLE001 - preserve the Hub's auth error as context.
        raise RuntimeError(
            "Could not infer the Hugging Face namespace. Pass --namespace explicitly "
            "or authenticate with HF_TOKEN / `hf auth login`."
        ) from exc

    inferred = account.get("name")
    if not inferred:
        raise RuntimeError("Authenticated Hub account did not include a usable account name.")
    return str(inferred)


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing model output directory: {model_dir}")

    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"{model_dir} is missing required artifact files: {missing_list}")


def _checkpoint_ignore(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name.startswith("checkpoint-")}


def _link_or_copy(src: str, dst: str) -> str:
    if Path(src).name in {"adapter_config.json", "README.md"}:
        return shutil.copy2(src, dst)

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _rewrite_adapter_config(path: Path, base_model_id: str) -> str | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    previous = data.get("base_model_name_or_path")
    if previous == base_model_id:
        return None

    data["base_model_name_or_path"] = base_model_id
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(previous) if previous is not None else None


def _rewrite_readme(path: Path, previous_base_model_id: str | None, base_model_id: str) -> None:
    if not previous_base_model_id:
        return

    text = path.read_text(encoding="utf-8")
    text = text.replace(f"base_model: {previous_base_model_id}", f"base_model: {base_model_id}")
    text = text.replace(
        f"base_model:adapter:{previous_base_model_id}",
        f"base_model:adapter:{base_model_id}",
    )
    path.write_text(text, encoding="utf-8")


def _stage_model_dir(
    source_dir: Path,
    destination_dir: Path,
    *,
    include_checkpoints: bool,
    base_model_id: str,
) -> None:
    ignore = None if include_checkpoints else _checkpoint_ignore
    shutil.copytree(source_dir, destination_dir, copy_function=_link_or_copy, ignore=ignore)

    for adapter_config in destination_dir.rglob("adapter_config.json"):
        previous_base_model_id = _rewrite_adapter_config(adapter_config, base_model_id)
        readme = adapter_config.with_name("README.md")
        if readme.exists():
            _rewrite_readme(readme, previous_base_model_id, base_model_id)


def _iter_files(path: Path) -> Iterable[Path]:
    return (candidate for candidate in path.rglob("*") if candidate.is_file())


def _format_size(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    args = _parse_args()
    _load_dotenv(Path(".env"))

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    namespace = _resolve_namespace(api, args.namespace)

    model_dirs = [(model_name, args.outputs_dir / model_name) for model_name in args.models]
    for _, model_dir in model_dirs:
        _validate_model_dir(model_dir)

    with tempfile.TemporaryDirectory(prefix=".hf-upload-", dir=Path.cwd()) as tmpdir:
        staging_root = Path(tmpdir)
        staged_models: list[tuple[str, Path]] = []

        for model_name, model_dir in model_dirs:
            staged_dir = staging_root / model_name
            _stage_model_dir(
                model_dir,
                staged_dir,
                include_checkpoints=args.include_checkpoints,
                base_model_id=args.base_model_id,
            )
            staged_models.append((model_name, staged_dir))

        for model_name, staged_dir in staged_models:
            repo_id = f"{namespace}/{model_name}"
            files = list(_iter_files(staged_dir))
            total_bytes = sum(path.stat().st_size for path in files)
            print(f"{repo_id}: {len(files)} files, {_format_size(total_bytes)}")

            if args.dry_run:
                continue

            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=args.private,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=staged_dir,
                revision=args.revision,
                commit_message=args.commit_message,
            )
            print(f"Uploaded {repo_id}")

    if args.dry_run:
        print("Dry run complete; no Hub repositories were created or modified.")


if __name__ == "__main__":
    main()
