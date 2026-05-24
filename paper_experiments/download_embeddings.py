"""
Download pre-computed fraud embedding .npy files from Zenodo record 20290842.

Downloads:
  fraud_minilm_zenodo.zip        → extracts into embeddings/fraud/
  fraud_bge-large_zenodo.zip     → extracts into embeddings/fraud/
  fraud_openai-small_zenodo.zip  → extracts into embeddings/fraud/
  fraud_openai-large_zenodo.zip  → extracts into embeddings/fraud/

Each zip is deleted after successful extraction. Files already present are
skipped unless --force is given.

Usage:
  python download_embeddings.py [--model minilm bge-large] [--force]
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings" / "fraud"

ZENODO_BASE = "https://zenodo.org/records/20290842/files"
MODEL_ARCHIVES = {
    "minilm":        "fraud_minilm_zenodo.zip",
    "bge-large":     "fraud_bge-large_zenodo.zip",
    "openai-small":  "fraud_openai-small_zenodo.zip",
    "openai-large":  "fraud_openai-large_zenodo.zip",
}


def _progress(downloaded: int, total: int) -> None:
    mb = downloaded / 1_048_576
    if total > 0:
        pct = min(downloaded / total * 100, 100.0)
        total_mb = total / 1_048_576
        print(f"\r  {mb:.0f} / {total_mb:.0f} MB  ({pct:.0f}%)", end="", flush=True)
    else:
        print(f"\r  {mb:.0f} MB", end="", flush=True)


def download_and_extract(archive_name: str, force: bool) -> None:
    url = f"{ZENODO_BASE}/{archive_name}"
    zip_path = EMBEDDINGS_DIR / archive_name

    # Check if all .npy files from this archive already exist by probing the zip
    # before downloading; instead just use the simpler heuristic: if any .npy
    # is missing for this rep, re-download the whole archive.
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if zip_path.exists() and not force:
        print(f"  zip already present: {zip_path.name}  (pass --force to re-download)")
    else:
        print(f"Downloading {archive_name} ...")
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
        )
        with opener.open(url) as resp, open(zip_path, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_reported = -1
            block = 1 << 16  # 64 KB
            while chunk := resp.read(block):
                fh.write(chunk)
                downloaded += len(chunk)
                pct = int(downloaded / total * 100) if total else 0
                if pct >= last_reported + 5:  # print every 5%
                    _progress(downloaded, total)
                    last_reported = pct
        _progress(downloaded, total)
        print()

    print(f"Extracting {zip_path.name} → {EMBEDDINGS_DIR}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        npy_members = [m for m in members if m.endswith(".npy")]
        skipped = 0
        extracted = 0
        for member in npy_members:
            dest = EMBEDDINGS_DIR / Path(member).name
            if dest.exists() and not force:
                skipped += 1
                continue
            # Extract flat (strip any subdirectory in the zip entry)
            data = zf.read(member)
            dest.write_bytes(data)
            extracted += 1
        print(f"  extracted={extracted}  skipped={skipped}")

    zip_path.unlink()
    print(f"  deleted {zip_path.name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download fraud embeddings from Zenodo record 20290842."
    )
    parser.add_argument(
        "--model", nargs="+", choices=list(MODEL_ARCHIVES), default=list(MODEL_ARCHIVES),
        metavar="MODEL",
        help=(
            f"Which embedding model(s) to download. "
            f"Choices: {', '.join(MODEL_ARCHIVES)}. Default: all."
        ),
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download and overwrite existing files.")
    args = parser.parse_args()

    for model in args.model:
        archive = MODEL_ARCHIVES[model]
        try:
            download_and_extract(archive, force=args.force)
        except Exception as exc:
            print(f"\nERROR downloading {archive}: {exc}", file=sys.stderr)
            sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
