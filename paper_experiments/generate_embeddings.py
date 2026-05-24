"""
Generate per-column, all-columns, and column-sweep embeddings for the fraud dataset.

For the chosen model, up to 8 base .npy files are produced:

  embeddings/fraud/fraud_{model}_{dim}d_{col}.npy   (one per text column, 7 files)
  embeddings/fraud/fraud_{model}_{dim}d.npy         (all 7 columns concatenated)

With --sweeps, an additional 12 sweep files are produced (plus 3 symlinks):

  Forward sweep  (fwd, canonical order, n=1..7):
    embeddings/fraud/fraud_{model}_{dim}d_fwd_title_to_{col}.npy
      n=1  → symlink to the per-column title file (if it exists)
      n=7  → symlink to the all-columns file (if it exists)

  Backward sweep  (back, reverse-canonical order, n=1..7):
    embeddings/fraud/fraud_{model}_{dim}d_back_benefits_to_{col}.npy
      n=1  → symlink to the per-column benefits file (if it exists)
      n=2..7 → new embeddings (columns in reverse canonical order)

Each file is skipped if it already exists (pass --force to regenerate).

Supported models
----------------
  minilm          : sentence-transformers/all-MiniLM-L6-v2 (384d)
  bge-large       : BAAI/bge-large-en-v1.5 (1024d)
  openai-small    : OpenAI text-embedding-3-small (1536d)
  openai-large    : OpenAI text-embedding-3-large (3072d)

Model loading
-------------
  HF models (minilm, bge-large) are downloaded from HuggingFace Hub on first use
  and cached in the default HF cache directory.

  If HF Hub is unreachable, set HF_MODEL_CACHE_DIR in your .env to a directory
  containing pre-downloaded model subdirectories:
    <HF_MODEL_CACHE_DIR>/all-MiniLM-L6-v2/
    <HF_MODEL_CACHE_DIR>/bge-large-en-v1.5/

  OpenAI models require OPENAI_API_KEY in .env.

Credentials
-----------
  OPENAI_API_KEY       : required for openai-small / openai-large
  HF_MODEL_CACHE_DIR   : optional; path to local HF model cache (see above)

Public API
----------
  TEXT_COLUMNS, MODEL_CONFIGS
  get_cache_path(dataset, model, text_cols=None) -> Path
  get_sweep_cache_path(dataset, model, direction, cols) -> Path
  get_embedding_model(model_key)
  load_dataset(dataset) -> (X, y, meta)

CLI
---
  python generate_embeddings.py --model <model_key> [--sweeps] [--force]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_core.embeddings import Embeddings as LangChainEmbeddings  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_ROOT = SCRIPT_DIR / "embeddings"

DATASET = "fraud"

BGE_LARGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
MINILM_REVISION = "fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"

TEXT_COLUMNS: dict[str, list[str]] = {
    "fraud": [
        "title",
        "location",
        "department",
        "company_profile",
        "description",
        "requirements",
        "benefits",
    ],
}

FRAUD_COLUMNS = TEXT_COLUMNS[DATASET]

MODEL_CONFIGS: dict[str, dict] = {
    "minilm": {
        "label": "MiniLM",
        "provider": "hf",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": MINILM_REVISION,
        "local_subdir": "all-MiniLM-L6-v2",
        "dim": 384,
    },
    "bge-large": {
        "label": "BGE large en v1.5",
        "provider": "hf",
        "model_id": "BAAI/bge-large-en-v1.5",
        "revision": BGE_LARGE_REVISION,
        "local_subdir": "bge-large-en-v1.5",
        "dim": 1024,
    },
    "openai-small": {
        "label": "OpenAI small",
        "provider": "openai",
        "model_id": "text-embedding-3-small",
        "dim": 1536,
    },
    "openai-large": {
        "label": "OpenAI large",
        "provider": "openai",
        "model_id": "text-embedding-3-large",
        "dim": 3072,
    },
}


def get_cache_path(
    dataset: str, model: str, text_cols: Optional[list[str]] = None
) -> Path:
    canonical = TEXT_COLUMNS[dataset]
    dim = MODEL_CONFIGS[model]["dim"]
    base = f"{dataset}_{model}_{dim}d"
    if text_cols is None or set(text_cols) == set(canonical):
        name = f"{base}.npy"
    else:
        ordered = [c for c in canonical if c in text_cols]
        suffix = "_".join(ordered)
        name = f"{base}_{suffix}.npy"
    return EMBEDDINGS_ROOT / dataset / name


def get_sweep_cache_path(
    dataset: str, model: str, direction: str, cols: list[str]
) -> Path:
    """Return the canonical path for a sweep embedding file.

    ``cols`` must already be in the order they will be fed to the model
    (canonical order for forward, reverse canonical for backward).

    Example filenames::

        fraud_minilm_384d_fwd_title_to_benefits.npy
        fraud_minilm_384d_back_benefits_to_title.npy
    """
    dim = MODEL_CONFIGS[model]["dim"]
    name = f"{dataset}_{model}_{dim}d_{direction}_{cols[0]}_to_{cols[-1]}.npy"
    return EMBEDDINGS_ROOT / dataset / name


def get_embedding_model(model_key: str):
    load_dotenv(find_dotenv())
    config = MODEL_CONFIGS[model_key]

    if config["provider"] == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        cache_dir = os.getenv("HF_MODEL_CACHE_DIR")
        if cache_dir:
            local_path = Path(cache_dir) / config["local_subdir"]
            if not local_path.exists():
                raise FileNotFoundError(
                    f"HF_MODEL_CACHE_DIR is set but model not found at: {local_path}\n"
                    "Either unset HF_MODEL_CACHE_DIR to download from HF Hub, "
                    "or place model files at the path above."
                )
            return HuggingFaceEmbeddings(
                model_name=str(local_path),
                model_kwargs={"local_files_only": True},
            )
        return HuggingFaceEmbeddings(
            model_name=config["model_id"],
            model_kwargs={"revision": config["revision"]},
        )

    if config["provider"] == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=config["model_id"], api_key=api_key)

    raise ValueError(f"unknown provider {config['provider']!r}")


def load_dataset(dataset: str):
    if dataset == "fraud":
        from tabullm import load_fraud

        return load_fraud()
    raise ValueError(f"unknown dataset {dataset!r}")


def _check_credentials(model_key: str) -> None:
    load_dotenv(find_dotenv())
    config = MODEL_CONFIGS[model_key]
    if config["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Add it to your .env file.")


def _build_sweep_targets(
    model: str,
) -> list[tuple[str, Path, list[str], Path | None]]:
    """Return (label, sweep_path, cols_in_feed_order, symlink_target_or_None)."""
    N = len(FRAUD_COLUMNS)
    targets: list[tuple[str, Path, list[str], Path | None]] = []

    for n in range(1, N + 1):
        fwd_cols = FRAUD_COLUMNS[:n]
        fwd_path = get_sweep_cache_path(DATASET, model, "fwd", fwd_cols)
        if n == 1:
            sym_target: Path | None = get_cache_path(DATASET, model, [fwd_cols[0]])
        elif n == N:
            sym_target = get_cache_path(DATASET, model)
        else:
            sym_target = None
        targets.append((
            f"fwd {fwd_cols[0]}..{fwd_cols[-1]} ({n} col(s))",
            fwd_path,
            fwd_cols,
            sym_target,
        ))

        back_cols = list(reversed(FRAUD_COLUMNS[N - n:]))
        back_path = get_sweep_cache_path(DATASET, model, "back", back_cols)
        if n == 1:
            back_sym: Path | None = get_cache_path(DATASET, model, [back_cols[0]])
        else:
            back_sym = None
        targets.append((
            f"back {back_cols[0]}..{back_cols[-1]} ({n} col(s))",
            back_path,
            back_cols,
            back_sym,
        ))

    return targets


def _make_relative_symlink(link: Path, target: Path, force: bool) -> bool:
    if not target.exists():
        print(f"  skip symlink  {link.name}  (target not found: {target.name})")
        return False
    if link.exists() or link.is_symlink():
        if not force:
            dest = os.readlink(link) if link.is_symlink() else "?"
            print(f"  cache hit    {link.name}  → {dest}")
            return False
        link.unlink()
    link.symlink_to(target.name)
    print(f"  symlink      {link.name}  → {target.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS), required=True,
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--sweeps", action="store_true",
        help="Also generate forward/backward column-sweep embeddings.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all files even if they already exist.",
    )
    args = parser.parse_args()

    _check_credentials(args.model)

    config = MODEL_CONFIGS[args.model]
    print(f"Dataset : {DATASET}")
    print(f"Model   : {config['label']} ({config['dim']}d)")
    print()

    base_targets: list[tuple[str, Path, list[str]]] = []
    for col in FRAUD_COLUMNS:
        base_targets.append((col, get_cache_path(DATASET, args.model, [col]), [col]))
    base_targets.append(("all cols", get_cache_path(DATASET, args.model), FRAUD_COLUMNS))

    to_generate: list[tuple[str, Path, list[str]]] = []
    for label, path, cols in base_targets:
        if path.exists() and not args.force:
            shape = np.load(path, mmap_mode="r").shape
            print(f"  cache hit  {path.name}  shape={shape}")
        else:
            to_generate.append((label, path, cols))

    sweep_embed: list[tuple[str, Path, list[str]]] = []
    if args.sweeps:
        print()
        print("--- sweep targets ---")
        for label, path, cols, sym_target in _build_sweep_targets(args.model):
            if path.exists() and not args.force:
                if path.is_symlink():
                    dest = os.readlink(path)
                    print(f"  cache hit  {path.name}  → {dest}")
                else:
                    shape = np.load(path, mmap_mode="r").shape
                    print(f"  cache hit  {path.name}  shape={shape}")
                continue
            if sym_target is not None:
                _make_relative_symlink(path, sym_target, force=args.force)
            else:
                sweep_embed.append((label, path, cols))

    all_to_generate = to_generate + sweep_embed

    if not all_to_generate:
        print("\nAll files already cached. Pass --force to regenerate.")
        return

    from tabullm import TextColumnTransformer  # heavy; defer until needed

    X, y, meta = load_dataset(DATASET)
    print(f"\nLoaded {DATASET}: n={len(X)}")

    embedding_model = get_embedding_model(args.model)
    transformer = TextColumnTransformer(model=embedding_model)

    print()
    for label, path, cols in all_to_generate:
        print(f"  embedding  {label} ({len(cols)} col(s)) ...", flush=True)
        X_emb = np.asarray(transformer.fit_transform(X[cols]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, X_emb)
        print(f"  saved      {path.name}  shape={X_emb.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
