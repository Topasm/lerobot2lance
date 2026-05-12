#!/usr/bin/env python3
"""Download and convert indexed 19D LeRobot datasets into local Lance bundles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from huggingface_hub import snapshot_download

from lerobot2lance import convert_lerobot_to_lance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index/hf_robotis_19d.json")
    parser.add_argument("--downloads", default="data/downloads_19d")
    parser.add_argument("--output", default="data/converted_19d")
    parser.add_argument("--status", default="data/index/convert_19d_status.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict-bg2-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of repos to convert in parallel.")
    parser.add_argument(
        "--hf-workers",
        type=int,
        default=4,
        help="Per-repo Hugging Face download workers. Total download concurrency is workers * hf-workers.",
    )
    parser.add_argument(
        "--lock-stale-minutes",
        type=float,
        default=360.0,
        help="Remove stale per-repo conversion locks older than this many minutes.",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    downloads_root = Path(args.downloads)
    output_root = Path(args.output)
    status_path = Path(args.status)
    downloads_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if args.strict_bg2_only:
        rows = [row for row in rows if row.get("robot_type") == "ffw_bg2_rev4"]
    if args.limit is not None:
        rows = rows[: args.limit]

    jobs = [
        make_job(row, downloads_root, output_root, status_path, args)
        for row in rows
        if args.overwrite or not target_is_done(output_root / slug_repo_id(str(row["repo_id"])))
    ]
    total = len(jobs)
    existing = len(rows) - total
    worker_count = max(1, args.workers)
    print(
        f"converting {total} pending dataset(s) from {index_path} "
        f"({existing} already done, workers={worker_count}, hf_workers={args.hf_workers})",
        flush=True,
    )
    if total == 0:
        return 0

    print_lock = threading.Lock()
    status_lock = threading.Lock()
    if worker_count == 1:
        for index, job in enumerate(jobs, 1):
            process_job(index, total, job, print_lock, status_lock)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(process_job, index, total, job, print_lock, status_lock)
                for index, job in enumerate(jobs, 1)
            ]
            for future in as_completed(futures):
                future.result()
    return 0


def slug_repo_id(repo_id: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "__" for ch in repo_id)
    return slug or "dataset"


def target_is_done(target_dir: Path) -> bool:
    manifest = target_dir / "manifest.json"
    episodes = target_dir / "data" / "episodes.lance"
    videos = target_dir / "data" / "videos.lance"
    if not (manifest.exists() and episodes.exists() and videos.exists()):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("format") == "rllab_published_lance_dataset_v2"
        and str(payload.get("schema_version")) == "2.0"
        and (payload.get("lance") or {}).get("blob_encoding") == "lance.blob.v2"
        and (payload.get("tables") or {}).get("episodes") == "data/episodes.lance"
        and (payload.get("tables") or {}).get("videos") == "data/videos.lance"
    )


def make_job(
    row: dict[str, Any],
    downloads_root: Path,
    output_root: Path,
    status_path: Path,
    args: argparse.Namespace,
) -> SimpleNamespace:
    repo_id = str(row["repo_id"])
    slug = slug_repo_id(repo_id)
    return SimpleNamespace(
        row=row,
        repo_id=repo_id,
        slug=slug,
        source_dir=downloads_root / slug,
        target_dir=output_root / slug,
        dataset_id=slug.replace("__", "-").replace("_", "-"),
        lock_path=output_root / ".locks" / f"{slug}.lock",
        status_path=status_path,
        overwrite=args.overwrite,
        keep_source=args.keep_source,
        hf_workers=max(1, args.hf_workers),
        lock_stale_seconds=max(0.0, args.lock_stale_minutes * 60.0),
    )


def process_job(
    index: int,
    total: int,
    job: SimpleNamespace,
    print_lock: threading.Lock,
    status_lock: threading.Lock,
) -> None:
    started = time.time()
    log(print_lock, f"\n[{index}/{total}] {job.repo_id}")
    try:
        with conversion_lock(job.lock_path, job.lock_stale_seconds) as acquired:
            if not acquired:
                log(print_lock, f"  locked elsewhere: {job.repo_id}")
                write_status_threadsafe(
                    status_lock,
                    job.status_path,
                    job.row,
                    job.target_dir,
                    "locked",
                    time.time() - started,
                )
                return
            if target_is_done(job.target_dir) and not job.overwrite:
                log(print_lock, f"  skip existing: {job.target_dir}")
                return

            snapshot_download(
                repo_id=job.repo_id,
                repo_type="dataset",
                local_dir=job.source_dir,
                max_workers=job.hf_workers,
            )
            report = convert_lerobot_to_lance(
                job.source_dir,
                job.target_dir,
                overwrite=True,
                output_layout="hf",
                dataset_id=job.dataset_id,
            )
            stamp_manifest(job.target_dir, job.row, job.dataset_id)
            if not job.keep_source:
                shutil.rmtree(job.source_dir, ignore_errors=True)
            log(print_lock, f"  wrote {job.target_dir}")
            write_status_threadsafe(
                status_lock,
                job.status_path,
                job.row,
                job.target_dir,
                "converted",
                time.time() - started,
                report,
            )
    except Exception as exc:  # noqa: BLE001
        log(print_lock, f"  ERROR {job.repo_id}: {exc}", error=True)
        write_status_threadsafe(
            status_lock,
            job.status_path,
            job.row,
            job.target_dir,
            "error",
            time.time() - started,
            error=repr(exc),
        )


@contextmanager
def conversion_lock(lock_path: Path, stale_seconds: float):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and stale_seconds > 0:
        age_seconds = time.time() - lock_path.stat().st_mtime
        if age_seconds > stale_seconds:
            lock_path.unlink(missing_ok=True)
    fd = None
    try:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            yield False
            return
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            f.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}) + "\n")
        yield True
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def log(lock: threading.Lock, message: str, *, error: bool = False) -> None:
    with lock:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


def stamp_manifest(target_dir: Path, row: dict[str, Any], dataset_id: str) -> None:
    for path in (target_dir / "manifest.json", target_dir / "meta" / "manifest.json"):
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dataset_id"] = dataset_id
        manifest["provenance"] = {
            **(manifest.get("provenance") or {}),
            "source_repo_id": row["repo_id"],
            "source_namespace": row["repo_id"].split("/", 1)[0],
            "source_robot_type": row.get("robot_type"),
            "source_robot_name": row.get("robot_name"),
            "pretrain_tier": pretrain_tier(row),
        }
        for legacy_key in ("source_dataset", "source_repo_id", "source_dataset_url"):
            manifest.pop(legacy_key, None)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pretrain_tier(row: dict[str, Any]) -> str:
    if row.get("robot_type") == "ffw_bg2_rev4":
        return "A_bg2_full_19d"
    if row.get("action_dim") == 19 and row.get("state_dim") == 19:
        return "A_other_19d"
    return "unknown"


def write_status(
    status_path: Path,
    row: dict[str, Any],
    target_dir: Path,
    status: str,
    elapsed_s: float,
    report: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "repo_id": row["repo_id"],
        "target": str(target_dir),
        "status": status,
        "elapsed_s": round(elapsed_s, 3),
        "robot_type": row.get("robot_type"),
        "total_frames": row.get("total_frames"),
    }
    if report is not None:
        payload["report"] = report
    if error is not None:
        payload["error"] = error
    with status_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_status_threadsafe(
    lock: threading.Lock,
    status_path: Path,
    row: dict[str, Any],
    target_dir: Path,
    status: str,
    elapsed_s: float,
    report: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with lock:
        write_status(status_path, row, target_dir, status, elapsed_s, report, error)


if __name__ == "__main__":
    raise SystemExit(main())
