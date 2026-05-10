#!/usr/bin/env python3
"""Validate every converted RLLAB Lance bundle under a root directory.

This is the batch companion to ``scripts/validate_bundle.py``.  It is intended
for data/converted_19d-style directories before upload or pretrain merging.
Each bundle result is written as one JSONL row so long runs can be inspected or
resumed by humans without opening the full logs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_bundle import validate_bundle


DEFAULT_STATUS = Path("data/index/validate_converted_root_status.jsonl")
DEFAULT_SUMMARY = Path("data/index/validate_converted_root_summary.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data/converted_19d"),
        help="Converted bundle root. Defaults to data/converted_19d.",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
        help=f"JSONL status output. Default: {DEFAULT_STATUS}",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"JSON summary output. Default: {DEFAULT_SUMMARY}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel validation workers. Use 1 for deterministic low-I/O runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validate only the first N discovered bundles.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed bundle. Ignored when --workers > 1.",
    )
    parser.add_argument(
        "--skip-blob-bytes",
        action="store_true",
        help="Skip video byte reads and sha256/byte_size checks for a fast schema pass.",
    )
    args = parser.parse_args()

    summary = validate_converted_root(
        args.root,
        status_path=args.status,
        summary_path=args.summary,
        workers=args.workers,
        limit=args.limit,
        fail_fast=args.fail_fast,
        check_blob_bytes=not args.skip_blob_bytes,
    )
    print_human_summary(summary)
    return 0 if summary["failed"] == 0 else 1


def validate_converted_root(
    root: Path,
    *,
    status_path: Path | None = DEFAULT_STATUS,
    summary_path: Path | None = DEFAULT_SUMMARY,
    workers: int = 1,
    limit: int | None = None,
    fail_fast: bool = False,
    check_blob_bytes: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    bundles = discover_bundles(root)
    if limit is not None:
        bundles = bundles[: max(0, limit)]

    started = time.time()
    results: list[dict[str, Any]] = []
    if status_path is not None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text("", encoding="utf-8")

    if workers <= 1:
        for index, bundle in enumerate(bundles, start=1):
            result = validate_one(bundle, root, check_blob_bytes=check_blob_bytes)
            results.append(result)
            append_status(status_path, result)
            print_progress(index, len(bundles), result)
            if fail_fast and not result["ok"]:
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_bundle = {
                executor.submit(validate_one, bundle, root, check_blob_bytes=check_blob_bytes): bundle
                for bundle in bundles
            }
            for index, future in enumerate(as_completed(future_to_bundle), start=1):
                result = future.result()
                results.append(result)
                append_status(status_path, result)
                print_progress(index, len(bundles), result)

    elapsed = time.time() - started
    failed = [row for row in results if not row["ok"]]
    warnings = sum(len(row.get("warnings") or []) for row in results)
    summary = {
        "root": str(root),
        "total_discovered": len(bundles),
        "total_validated": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "warnings": warnings,
        "check_blob_bytes": check_blob_bytes,
        "elapsed_sec": round(elapsed, 3),
        "status_path": str(status_path) if status_path is not None else None,
        "failed_bundles": [
            {
                "bundle": row["bundle"],
                "errors": row.get("errors") or [],
            }
            for row in failed
        ],
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def discover_bundles(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"converted root does not exist: {root}")
    if (root / "manifest.json").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("manifest.json")})


def validate_one(bundle: Path, root: Path, *, check_blob_bytes: bool) -> dict[str, Any]:
    started = time.time()
    try:
        result = validate_bundle(bundle, check_blob_bytes=check_blob_bytes)
    except Exception as exc:  # pragma: no cover - defensive batch isolation
        result = {
            "ok": False,
            "summary": {"bundle": str(bundle.resolve())},
            "errors": [f"validator exception: {type(exc).__name__}: {exc}"],
            "warnings": [],
        }
    elapsed = time.time() - started
    summary = result.get("summary") or {}
    resolved_bundle = Path(summary.get("bundle") or bundle).resolve()
    try:
        rel_bundle = str(resolved_bundle.relative_to(root))
    except ValueError:
        rel_bundle = str(resolved_bundle)
    return {
        "bundle": str(resolved_bundle),
        "relative_bundle": rel_bundle,
        "ok": bool(result.get("ok")),
        "errors": result.get("errors") or [],
        "warnings": result.get("warnings") or [],
        "counts": summary.get("counts"),
        "rows": summary.get("rows"),
        "format": summary.get("format"),
        "schema_version": summary.get("schema_version"),
        "elapsed_sec": round(elapsed, 3),
    }


def append_status(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def print_progress(index: int, total: int, result: dict[str, Any]) -> None:
    status = "OK" if result["ok"] else "FAIL"
    print(f"[{index}/{total}] {status} {result['relative_bundle']}")
    for error in result.get("errors") or []:
        print(f"  error: {error}")


def print_human_summary(summary: dict[str, Any]) -> None:
    print(
        "validated {total_validated}/{total_discovered} bundles: "
        "{passed} passed, {failed} failed, {warnings} warnings "
        "({elapsed_sec:.3f}s)".format(**summary)
    )
    if summary.get("status_path"):
        print(f"status: {summary['status_path']}")


if __name__ == "__main__":
    sys.exit(main())
