#!/usr/bin/env python3
"""Download indexed 19D LeRobot HF datasets without converting them."""

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
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


DEFAULT_NAMESPACES = ("ROBOTIS", "RobotisSW", "Dongkkka")
EXTRA_REPOS = ("ubless607/ffw_bg2_rev4_pick-place",)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download raw HF LeRobot snapshots for 19D AI Worker/BG2 datasets."
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--index", default="data/index/hf_robotis_19d.json")
    parser.add_argument("--all-index", default="data/index/hf_robotis_index.json")
    parser.add_argument("--status", default="data/index/download_raw_19d_status.jsonl")
    parser.add_argument("--namespaces", nargs="*", default=list(DEFAULT_NAMESPACES))
    parser.add_argument("--extra-repo", action="append", default=list(EXTRA_REPOS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict-bg2-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-workers", type=int, default=4)
    parser.add_argument(
        "--retries",
        type=int,
        default=12,
        help="Retry each dataset download this many times for transient network/rate-limit errors.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=60.0,
        help="Seconds to sleep before retrying ordinary transient download errors.",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=900.0,
        help="Seconds to sleep before retrying Hugging Face 429/rate-limit errors.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the HF metadata index even if --index already exists.",
    )
    parser.add_argument(
        "--lock-stale-minutes",
        type=float,
        default=360.0,
        help="Remove stale per-repo download locks older than this many minutes.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    index_path = Path(args.index)
    all_index_path = Path(args.all_index)
    status_path = Path(args.status)
    raw_root.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reindex or not index_path.exists():
        rows = build_index(
            namespaces=args.namespaces,
            extra_repos=args.extra_repo,
            index_path=index_path,
            all_index_path=all_index_path,
        )
    else:
        rows = json.loads(index_path.read_text(encoding="utf-8"))

    if args.strict_bg2_only:
        rows = [row for row in rows if row.get("robot_type") == "ffw_bg2_rev4"]
    if args.limit is not None:
        rows = rows[: args.limit]

    jobs = [
        make_job(row, raw_root, status_path, args)
        for row in rows
        if args.overwrite or not raw_snapshot_done(raw_root / slug_repo_id(str(row["repo_id"])))
    ]
    existing = len(rows) - len(jobs)
    print(
        f"downloading {len(jobs)} pending raw dataset(s) into {raw_root} "
        f"({existing} already present, workers={args.workers}, hf_workers={args.hf_workers})",
        flush=True,
    )
    if not jobs:
        write_raw_readme(raw_root, rows, status_path)
        return 0

    print_lock = threading.Lock()
    status_lock = threading.Lock()
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, job in enumerate(jobs, 1):
            process_job(index, len(jobs), job, print_lock, status_lock)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_job, index, len(jobs), job, print_lock, status_lock)
                for index, job in enumerate(jobs, 1)
            ]
            for future in as_completed(futures):
                future.result()

    write_raw_readme(raw_root, rows, status_path)
    return 0


def build_index(
    *,
    namespaces: list[str],
    extra_repos: list[str],
    index_path: Path,
    all_index_path: Path,
) -> list[dict[str, Any]]:
    api = HfApi()
    all_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for namespace in namespaces:
        datasets = list(api.list_datasets(author=namespace))
        print(f"index {namespace}: {len(datasets)} dataset(s)", flush=True)
        for offset, dataset in enumerate(datasets, 1):
            repo_id = dataset.id
            if repo_id in seen:
                continue
            seen.add(repo_id)
            all_rows.append(fetch_meta_row(repo_id, namespace=namespace))
            if offset % 50 == 0:
                print(f"  {namespace}: {offset}/{len(datasets)}", flush=True)
    for repo_id in extra_repos:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        namespace = repo_id.split("/", 1)[0] if "/" in repo_id else ""
        all_rows.append(fetch_meta_row(repo_id, namespace=namespace))

    valid = [row for row in all_rows if row.get("ok")]
    selected = [
        row
        for row in valid
        if int(row.get("state_dim") or 0) == 19
        and int(row.get("action_dim") or 0) == 19
    ]
    selected.sort(key=lambda row: str(row["repo_id"]))
    all_index_path.write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "indexed": len(all_rows),
                "valid": len(valid),
                "selected_19d": len(selected),
                "selected_frames": sum(int(row.get("total_frames") or 0) for row in selected),
                "index": str(index_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return selected


def fetch_meta_row(repo_id: str, *, namespace: str) -> dict[str, Any]:
    row: dict[str, Any] = {"repo_id": repo_id, "namespace": namespace}
    try:
        meta_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        features = meta.get("features") or {}
        action_dim, action_shape = dim_from_feature(features.get("action"))
        state_dim, state_shape = dim_from_feature(features.get("observation.state"))
        cameras = [name for name in features if name.startswith("observation.images")]
        row.update(
            {
                "ok": True,
                "robot_type": meta.get("robot_type"),
                "robot_name": meta.get("robot_name"),
                "fps": meta.get("fps"),
                "total_episodes": meta.get("total_episodes"),
                "total_frames": meta.get("total_frames"),
                "action_dim": action_dim,
                "action_shape": action_shape,
                "state_dim": state_dim,
                "state_shape": state_shape,
                "cameras": cameras,
                "camera_shapes": {key: (features.get(key) or {}).get("shape") for key in cameras},
                "camera_names": {key: (features.get(key) or {}).get("names") for key in cameras},
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"ok": False, "error": repr(exc)})
    return row


def dim_from_feature(feature: Any) -> tuple[int | None, Any]:
    shape = (feature or {}).get("shape") if isinstance(feature, dict) else None
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return int(shape[0]), shape
    return None, shape


def make_job(row: dict[str, Any], raw_root: Path, status_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    repo_id = str(row["repo_id"])
    slug = slug_repo_id(repo_id)
    return {
        "row": row,
        "repo_id": repo_id,
        "target_dir": raw_root / slug,
        "lock_path": raw_root / ".locks" / f"{slug}.lock",
        "status_path": status_path,
        "overwrite": bool(args.overwrite),
        "hf_workers": max(1, int(args.hf_workers)),
        "lock_stale_seconds": max(0.0, float(args.lock_stale_minutes) * 60.0),
        "retries": max(0, int(args.retries)),
        "retry_sleep": max(0.0, float(args.retry_sleep)),
        "rate_limit_sleep": max(0.0, float(args.rate_limit_sleep)),
    }


def process_job(
    index: int,
    total: int,
    job: dict[str, Any],
    print_lock: threading.Lock,
    status_lock: threading.Lock,
) -> None:
    started = time.time()
    repo_id = job["repo_id"]
    target_dir = Path(job["target_dir"])
    log(print_lock, f"\n[{index}/{total}] {repo_id}")
    try:
        with download_lock(Path(job["lock_path"]), float(job["lock_stale_seconds"])) as acquired:
            if not acquired:
                log(print_lock, f"  locked elsewhere: {repo_id}")
                write_status_threadsafe(status_lock, job, "locked", time.time() - started)
                return
            if raw_snapshot_done(target_dir) and not job["overwrite"]:
                log(print_lock, f"  skip existing: {target_dir}")
                return
            for attempt in range(int(job["retries"]) + 1):
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        local_dir=target_dir,
                        max_workers=int(job["hf_workers"]),
                    )
                    if has_incomplete_files(target_dir):
                        raise RuntimeError(f"Snapshot still has incomplete files under {target_dir}")
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt >= int(job["retries"]) or not is_retryable_download_error(exc):
                        raise
                    sleep_s = (
                        float(job["rate_limit_sleep"])
                        if is_rate_limit_error(exc)
                        else float(job["retry_sleep"])
                    )
                    log(
                        print_lock,
                        f"  retry {attempt + 1}/{job['retries']} for {repo_id} "
                        f"after {sleep_s:.0f}s: {exc}",
                        error=True,
                    )
                    time.sleep(sleep_s)
            write_json(
                target_dir / "rllab_source.json",
                {
                    **job["row"],
                    "source_dataset_url": f"https://huggingface.co/datasets/{repo_id}",
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            log(print_lock, f"  downloaded {target_dir}")
            write_status_threadsafe(status_lock, job, "downloaded", time.time() - started)
    except Exception as exc:  # noqa: BLE001
        log(print_lock, f"  ERROR {repo_id}: {exc}", error=True)
        write_status_threadsafe(status_lock, job, "error", time.time() - started, error=repr(exc))


def is_rate_limit_error(exc: BaseException) -> bool:
    message = repr(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def is_retryable_download_error(exc: BaseException) -> bool:
    message = repr(exc).lower()
    return (
        is_rate_limit_error(exc)
        or "localentrynotfounderror" in message
        or "please check your internet connection" in message
        or "connection" in message
        or "connection reset" in message
        or "timeout" in message
        or "temporarily unavailable" in message
        or "incomplete files" in message
    )


def raw_snapshot_done(target_dir: Path) -> bool:
    return (
        (target_dir / "meta" / "info.json").exists()
        and (target_dir / "data").exists()
        and (target_dir / "rllab_source.json").exists()
        and not has_incomplete_files(target_dir)
    )


def has_incomplete_files(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    try:
        next(target_dir.rglob("*.incomplete"))
    except StopIteration:
        return False
    return True


def slug_repo_id(repo_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "__" for ch in repo_id) or "dataset"


@contextmanager
def download_lock(lock_path: Path, stale_seconds: float):
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
    job: dict[str, Any],
    status: str,
    elapsed_s: float,
    *,
    error: str | None = None,
) -> None:
    with lock:
        payload = {
            "repo_id": job["repo_id"],
            "target": str(job["target_dir"]),
            "status": status,
            "elapsed_s": round(elapsed_s, 3),
            "robot_type": job["row"].get("robot_type"),
            "total_frames": job["row"].get("total_frames"),
        }
        if error is not None:
            payload["error"] = error
        with Path(job["status_path"]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_raw_readme(raw_root: Path, rows: list[dict[str, Any]], status_path: Path) -> None:
    downloaded = []
    status_rows = latest_status(status_path)
    for row in rows:
        repo_id = str(row["repo_id"])
        target = raw_root / slug_repo_id(repo_id)
        if raw_snapshot_done(target):
            downloaded.append((row, target, status_rows.get(repo_id, {})))
    lines = [
        "# Raw 19D LeRobot Snapshots",
        "",
        "This folder stores raw Hugging Face dataset snapshots only. No Lance conversion has been run here.",
        "",
        f"- Raw root: `{raw_root}`",
        f"- Indexed 19D datasets: `{len(rows)}`",
        f"- Downloaded snapshots: `{len(downloaded)}`",
        "",
        "| Source dataset | Local folder | Robot type | Episodes | Frames | FPS | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row, target, status in downloaded:
        repo_id = str(row["repo_id"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{repo_id}](https://huggingface.co/datasets/{repo_id})",
                    f"`{target.name}`",
                    f"`{row.get('robot_type') or ''}`",
                    str(row.get("total_episodes") or ""),
                    str(row.get("total_frames") or ""),
                    str(row.get("fps") or ""),
                    f"`{status.get('status', 'present')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Resume",
            "",
            "```bash",
            "/scratch/e1816a02/.venv/bin/python scripts/download_hf_raw_19d.py --raw-root data/raw",
            "```",
            "",
        ]
    )
    (raw_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def latest_status(status_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not status_path.exists():
        return rows
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        repo_id = payload.get("repo_id")
        if repo_id:
            rows[str(repo_id)] = payload
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log(lock: threading.Lock, message: str, *, error: bool = False) -> None:
    with lock:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
