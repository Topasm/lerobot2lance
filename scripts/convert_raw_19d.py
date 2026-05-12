#!/usr/bin/env python3
"""Convert validated raw 19D LeRobot snapshots into published Lance bundles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot2lance import convert_lerobot_to_lance


PUBLISHED_FORMAT = "rllab_published_lance_dataset_v2"
DEFAULT_VALIDATION = Path("data/index/raw_19d_validation.json")
DEFAULT_OUTPUT = Path("data/converted_19d")
DEFAULT_STATUS = Path("data/index/convert_raw_19d_status.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-review-names", action="store_true", default=True)
    parser.add_argument(
        "--exclude-review-names",
        action="store_false",
        dest="include_review_names",
        help="Skip datasets whose repo names look like test/upload placeholders.",
    )
    parser.add_argument("--strict-bg2-only", action="store_true")
    parser.add_argument("--lock-stale-minutes", type=float, default=360.0)
    args = parser.parse_args()

    payload = json.loads(args.validation.read_text(encoding="utf-8"))
    rows = list(payload.get("datasets") or [])
    jobs = [
        make_job(row, args)
        for row in rows
        if row.get("ok")
        and (args.include_review_names or row.get("quality_flag") == "ok")
        and (not args.strict_bg2_only or row.get("robot_type") == "ffw_bg2_rev4")
    ]
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]

    args.output.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)

    pending = [job for job in jobs if args.overwrite or not target_is_done(job.target_dir)]
    existing = len(jobs) - len(pending)
    print(
        f"converting {len(pending)} validated raw dataset(s) into {args.output} "
        f"({existing} already done, workers={max(1, args.workers)})",
        flush=True,
    )
    print(
        f"validation summary: {payload.get('summary')}",
        flush=True,
    )
    if not pending:
        return 0

    print_lock = threading.Lock()
    status_lock = threading.Lock()
    if args.workers <= 1:
        for index, job in enumerate(pending, 1):
            process_job(index, len(pending), job, print_lock, status_lock)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(process_job, index, len(pending), job, print_lock, status_lock)
                for index, job in enumerate(pending, 1)
            ]
            for future in as_completed(futures):
                future.result()
    return 0


def make_job(row: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    repo_id = str(row["repo_id"])
    folder = str(row.get("folder") or slug_repo_id(repo_id))
    return SimpleNamespace(
        row=row,
        repo_id=repo_id,
        slug=folder,
        source_dir=Path(row["raw_path"]),
        target_dir=args.output / folder,
        dataset_id=folder.replace("__", "-").replace("_", "-"),
        status_path=args.status,
        overwrite=bool(args.overwrite),
        lock_path=args.output / ".locks" / f"{folder}.lock",
        lock_stale_seconds=max(0.0, float(args.lock_stale_minutes) * 60.0),
    )


def process_job(
    index: int,
    total: int,
    job: SimpleNamespace,
    print_lock: threading.Lock,
    status_lock: threading.Lock,
) -> None:
    started = time.time()
    log(print_lock, f"\n[{index}/{total}] {job.repo_id} ({job.row.get('observed_frames')} frames)")
    try:
        with conversion_lock(job.lock_path, job.lock_stale_seconds) as acquired:
            if not acquired:
                write_status_threadsafe(status_lock, job, "locked", time.time() - started)
                log(print_lock, f"  locked elsewhere: {job.repo_id}")
                return
            if target_is_done(job.target_dir) and not job.overwrite:
                write_status_threadsafe(status_lock, job, "skipped_existing", time.time() - started)
                log(print_lock, f"  skip existing: {job.target_dir}")
                return
            report = convert_lerobot_to_lance(
                job.source_dir,
                job.target_dir,
                overwrite=True,
                output_layout="hf",
                dataset_id=job.dataset_id,
            )
            stamp_manifest(job.target_dir, job.row, job.dataset_id, report)
            write_status_threadsafe(status_lock, job, "converted", time.time() - started, report)
            log(print_lock, f"  wrote {job.target_dir}")
    except Exception as exc:  # noqa: BLE001
        write_status_threadsafe(status_lock, job, "error", time.time() - started, error=repr(exc))
        log(print_lock, f"  ERROR {job.repo_id}: {exc}", error=True)


def target_is_done(target_dir: Path) -> bool:
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        manifest.get("format") == PUBLISHED_FORMAT
        and str(manifest.get("schema_version") or "").startswith("2.")
        and (manifest.get("counts") or {}).get("frames")
        and (target_dir / "data" / "episodes.lance").exists()
        and (target_dir / "data" / "videos.lance").exists()
    )


def stamp_manifest(
    target_dir: Path,
    row: dict[str, Any],
    dataset_id: str,
    report: dict[str, Any],
) -> None:
    for path in (target_dir / "manifest.json", target_dir / "meta" / "manifest.json"):
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dataset_id"] = dataset_id
        manifest["provenance"] = {
            **(manifest.get("provenance") or {}),
            "source_repo_id": row["repo_id"],
            "source_dataset_url": row.get("source_url"),
            "source_namespace": str(row["repo_id"]).split("/", 1)[0],
            "source_robot_type": row.get("robot_type"),
            "source_robot_name": row.get("robot_name"),
            "source_total_episodes": row.get("info_total_episodes"),
            "source_total_frames": row.get("info_total_frames"),
            "source_observed_episodes": row.get("observed_episodes"),
            "source_observed_frames": row.get("observed_frames"),
            "source_indexed_frames": row.get("indexed_frames"),
            "raw_validation_status": row.get("status"),
            "quality_flag": row.get("quality_flag"),
            "instructionish": bool(row.get("instructionish")),
            "instructionish_tasks_sample": row.get("instructionish_tasks_sample") or [],
            "pretrain_tier": pretrain_tier(row),
        }
        for legacy_key in ("source_dataset", "source_repo_id", "source_dataset_url"):
            manifest.pop(legacy_key, None)
        manifest.setdefault("conversion_report", {}).update(
            {
                "episodes_written": report.get("episodes_written"),
                "frames_written": report.get("frames_written"),
                "media_written": report.get("media_written"),
            }
        )
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pretrain_tier(row: dict[str, Any]) -> str:
    if row.get("robot_type") == "ffw_bg2_rev4":
        return "A_bg2_full_19d"
    if row.get("state_dim") == 19 and row.get("action_dim") == 19:
        return "A_other_19d"
    return "unknown"


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
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}) + "\n")
        yield True
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def write_status_threadsafe(
    lock: threading.Lock,
    job: SimpleNamespace,
    status: str,
    elapsed_s: float,
    report: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with lock:
        payload = {
            "repo_id": job.repo_id,
            "source": str(job.source_dir),
            "target": str(job.target_dir),
            "status": status,
            "elapsed_s": round(elapsed_s, 3),
            "robot_type": job.row.get("robot_type"),
            "quality_flag": job.row.get("quality_flag"),
            "indexed_frames": job.row.get("indexed_frames"),
            "info_total_frames": job.row.get("info_total_frames"),
            "observed_frames": job.row.get("observed_frames"),
        }
        if report is not None:
            payload["converted_frames"] = report.get("frames_written")
            payload["converted_episodes"] = report.get("episodes_written")
            payload["converted_media"] = report.get("media_written")
            payload["report"] = report
        if error is not None:
            payload["error"] = error
        with job.status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log(lock: threading.Lock, message: str, *, error: bool = False) -> None:
    with lock:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


def slug_repo_id(repo_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "__" for ch in repo_id) or "dataset"


if __name__ == "__main__":
    raise SystemExit(main())
