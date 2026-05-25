"""Download pre-computed paper experiment results from Zenodo.

Default behavior:
  - Resolve the latest published Zenodo record under concept DOI 10.5281/zenodo.18884001
  - Download file `results.zip` from that record
  - Extract into this folder (so `results/...` lands at the expected path)

Usage examples:
  python download_results.py
  python download_results.py --list-files
  python download_results.py --file results.zip --force
    python download_results.py --dest-dir /tmp/paper_results_test
  python download_results.py --record-id 12345678 --file my_results_bundle.zip
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import certifi

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONCEPT_DOI = "10.5281/zenodo.18884001"
DEFAULT_FILE = "results.zip"
API_BASE = "https://zenodo.org/api/records"


def _open_json(url: str) -> dict:
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
    )
    with opener.open(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_latest_record_id(concept_doi: str) -> int:
    query = urllib.parse.urlencode(
        {
            "q": f'conceptdoi:"{concept_doi}"',
            "sort": "mostrecent",
            "size": 1,
        }
    )
    url = f"{API_BASE}?{query}"
    payload = _open_json(url)
    hits = payload.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError(
            f"No Zenodo records found for concept DOI {concept_doi}."
        )
    return int(hits[0]["id"])


def fetch_record(record_id: int) -> dict:
    return _open_json(f"{API_BASE}/{record_id}")


def _file_key(file_obj: dict) -> str:
    return str(file_obj.get("key") or file_obj.get("filename") or "")


def _file_url(file_obj: dict) -> str:
    links = file_obj.get("links", {})
    url = links.get("self")
    if not url:
        raise RuntimeError(f"Missing download URL for file entry: {_file_key(file_obj)}")
    return str(url)


def _progress(downloaded: int, total: int) -> None:
    mb = downloaded / 1_048_576
    if total > 0:
        pct = min(downloaded / total * 100, 100.0)
        total_mb = total / 1_048_576
        print(f"\r  {mb:.0f} / {total_mb:.0f} MB  ({pct:.0f}%)", end="", flush=True)
    else:
        print(f"\r  {mb:.0f} MB", end="", flush=True)


def download_file(url: str, out_path: Path, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  already exists: {out_path.name}  (pass --force to re-download)")
        return

    print(f"Downloading {out_path.name} ...")
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
    )
    with opener.open(url) as resp, open(out_path, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_reported = -1
        block = 1 << 16  # 64 KB
        while chunk := resp.read(block):
            fh.write(chunk)
            downloaded += len(chunk)
            pct = int(downloaded / total * 100) if total else 0
            if pct >= last_reported + 5:
                _progress(downloaded, total)
                last_reported = pct
    _progress(downloaded, total)
    print()


def _safe_extract_member(zf: zipfile.ZipFile, member: str, extract_base: Path) -> None:
    if Path(member).is_absolute():
        raise RuntimeError(f"Unsafe zip member path (absolute): {member}")
    target = (extract_base / member).resolve()
    base = extract_base.resolve()
    if not target.is_relative_to(base):
        raise RuntimeError(f"Unsafe zip member path: {member}")
    if member.endswith("/"):
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(zf.read(member))


def extract_zip(zip_path: Path, extract_root: Path, force: bool) -> None:
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        if not members:
            print("  archive is empty")
            return

        has_results_prefix = all(m.startswith("results/") for m in members)
        extract_base = extract_root if has_results_prefix else (extract_root / "results")

        extracted = 0
        skipped = 0
        for member in members:
            out_path = (extract_base / member).resolve()
            if out_path.exists() and not force:
                skipped += 1
                continue
            _safe_extract_member(zf, member, extract_base)
            extracted += 1

    print(f"  extracted={extracted}  skipped={skipped}")


def select_files(record: dict, requested_files: list[str]) -> list[dict]:
    files = record.get("files", [])
    by_key = {_file_key(f): f for f in files}
    missing = [name for name in requested_files if name not in by_key]
    if missing:
        available = sorted(_file_key(f) for f in files)
        raise RuntimeError(
            "Requested file(s) not found in Zenodo record: "
            f"{', '.join(missing)}\nAvailable files: {available}"
        )
    return [by_key[name] for name in requested_files]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download paper experiment results from Zenodo."
    )
    parser.add_argument(
        "--concept-doi",
        default=DEFAULT_CONCEPT_DOI,
        help=(
            "Zenodo concept DOI used to resolve latest record "
            f"(default: {DEFAULT_CONCEPT_DOI})."
        ),
    )
    parser.add_argument(
        "--record-id",
        type=int,
        default=None,
        help="Explicit Zenodo record ID (overrides --concept-doi lookup).",
    )
    parser.add_argument(
        "--file",
        nargs="+",
        default=[DEFAULT_FILE],
        metavar="FILENAME",
        help=f"File(s) to download from the record (default: {DEFAULT_FILE}).",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List files in the resolved record and exit.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not auto-extract zip archives after download.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep downloaded archive files after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing downloaded/extracted files.",
    )
    parser.add_argument(
        "--dest-dir",
        default=str(SCRIPT_DIR),
        metavar="PATH",
        help=(
            "Destination directory for downloaded files and extraction root "
            f"(default: {SCRIPT_DIR})."
        ),
    )
    args = parser.parse_args()

    try:
        record_id = args.record_id or resolve_latest_record_id(args.concept_doi)
        record = fetch_record(record_id)
        files = record.get("files", [])
        dest_dir = Path(args.dest_dir).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Zenodo record ID: {record_id}")
        print(f"Record URL      : https://zenodo.org/records/{record_id}")
        print(f"Files in record : {len(files)}")
        print(f"Destination dir : {dest_dir}")

        if args.list_files:
            for file_obj in sorted(files, key=lambda f: _file_key(f)):
                print(f"  - {_file_key(file_obj)}")
            return

        selected = select_files(record, args.file)
        for file_obj in selected:
            key = _file_key(file_obj)
            url = _file_url(file_obj)
            local_path = dest_dir / Path(key).name
            download_file(url, local_path, force=args.force)

            is_zip = local_path.suffix.lower() == ".zip"
            if is_zip and not args.no_extract:
                extract_zip(local_path, extract_root=dest_dir, force=args.force)
                if not args.keep_archive:
                    local_path.unlink(missing_ok=True)
                    print(f"  deleted {local_path.name}")
            print()

        print("Done.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
