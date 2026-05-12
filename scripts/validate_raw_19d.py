#!/usr/bin/env python3
"""Validate downloaded raw 19D LeRobot snapshots before Lance conversion."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required: install into the scratch venv first") from exc


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_INDEX = Path("data/index/hf_robotis_19d.json")
DEFAULT_OUTPUT = Path("data/index/raw_19d_validation.json")
DEFAULT_STATUS = Path("data/index/raw_19d_validation_status.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=19)
    parser.add_argument("--action-dim", type=int, default=19)
    parser.add_argument("--strict-bg2-only", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root
    index_rows = load_index(args.index)
    if args.strict_bg2_only:
        index_rows = [row for row in index_rows if row.get("robot_type") == "ffw_bg2_rev4"]
    jobs = build_jobs(raw_root, index_rows)
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text("", encoding="utf-8")

    print(
        f"validating {len(jobs)} raw dataset(s) under {raw_root} "
        f"(workers={max(1, args.workers)})",
        flush=True,
    )
    started = time.time()
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, job in enumerate(jobs, 1):
            result = validate_one(job, args.state_dim, args.action_dim)
            results.append(result)
            append_status(args.status, result)
            print_progress(index, len(jobs), result)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_job = {
                executor.submit(validate_one, job, args.state_dim, args.action_dim): job
                for job in jobs
            }
            for index, future in enumerate(as_completed(future_to_job), 1):
                result = future.result()
                results.append(result)
                append_status(args.status, result)
                print_progress(index, len(jobs), result)

    results.sort(key=lambda row: str(row.get("repo_id") or row.get("raw_path")))
    summary = build_summary(results, started)
    args.output.write_text(
        json.dumps({"summary": summary, "datasets": results}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0 if summary["bad_datasets"] == 0 else 0


def load_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_jobs(raw_root: Path, index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for row in index_rows:
        repo_id = str(row["repo_id"])
        path = raw_root / slug_repo_id(repo_id)
        jobs.append({"row": row, "raw_path": path, "repo_id": repo_id})
        seen_paths.add(path.resolve())

    for raw_path in sorted(raw_root.iterdir() if raw_root.exists() else []):
        if raw_path.name.startswith(".") or not raw_path.is_dir():
            continue
        resolved = raw_path.resolve()
        if resolved in seen_paths:
            continue
        source_row = read_json_if_exists(raw_path / "rllab_source.json") or {}
        repo_id = source_row.get("repo_id") or raw_path.name.replace("__", "/")
        jobs.append({"row": source_row, "raw_path": raw_path, "repo_id": repo_id})
    return jobs


def validate_one(job: dict[str, Any], expected_state_dim: int, expected_action_dim: int) -> dict[str, Any]:
    started = time.time()
    raw_path = Path(job["raw_path"])
    row = dict(job.get("row") or {})
    repo_id = str(job.get("repo_id") or row.get("repo_id") or raw_path.name)
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "raw_path": str(raw_path),
        "folder": raw_path.name,
        "source_url": hf_dataset_url(repo_id),
        "indexed_episodes": row.get("total_episodes"),
        "indexed_frames": row.get("total_frames"),
        "status": "ok",
        "ok": True,
        "errors": [],
        "warnings": [],
        "quality_flag": quality_flag(repo_id),
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]

    if not raw_path.exists():
        errors.append("raw snapshot missing")
        return finish(result, started)
    incomplete = list(raw_path.rglob("*.incomplete"))
    if incomplete:
        errors.append(f"incomplete download files present: {len(incomplete)}")
    info_path = raw_path / "meta" / "info.json"
    info = read_json_if_exists(info_path)
    if not isinstance(info, dict):
        errors.append("meta/info.json missing or invalid JSON")
        return finish(result, started)

    features = info.get("features") or {}
    state_dim, state_shape = dim_from_feature(features.get("observation.state"))
    action_dim, action_shape = dim_from_feature(features.get("action"))
    result.update(
        {
            "robot_type": info.get("robot_type") or row.get("robot_type"),
            "robot_name": info.get("robot_name") or row.get("robot_name"),
            "fps": info.get("fps") or row.get("fps"),
            "info_total_episodes": info.get("total_episodes"),
            "info_total_frames": info.get("total_frames"),
            "state_dim": state_dim,
            "state_shape": state_shape,
            "action_dim": action_dim,
            "action_shape": action_shape,
            "camera_keys": [
                key
                for key, feature in features.items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            ],
        }
    )
    if state_dim != expected_state_dim:
        errors.append(f"state_dim {state_dim} != expected {expected_state_dim}")
    if action_dim != expected_action_dim:
        errors.append(f"action_dim {action_dim} != expected {expected_action_dim}")
    try:
        fps = float(result.get("fps") or 0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0:
        errors.append("fps missing or non-positive")

    tasks, task_warnings = load_tasks(raw_path)
    warnings.extend(task_warnings)
    meaningful_tasks = [task for task in tasks if is_instructionish(task)]
    result.update(
        {
            "tasks_count": len(tasks),
            "tasks_sample": tasks[:8],
            "instructionish": bool(meaningful_tasks),
            "instructionish_tasks_sample": meaningful_tasks[:8],
        }
    )

    try:
        layout = detect_layout(raw_path)
        result["layout"] = layout
        episode_rows = load_episode_rows(raw_path, layout)
        result["observed_episodes"] = len(episode_rows)
        if not episode_rows:
            errors.append("no episode metadata rows")
        observed_frames = validate_frame_tables(
            raw_path,
            layout,
            info,
            episode_rows,
            errors,
            warnings,
            expected_state_dim=expected_state_dim,
            expected_action_dim=expected_action_dim,
        )
        result["observed_frames"] = observed_frames
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")

    compare_count("episodes", result.get("info_total_episodes"), result.get("observed_episodes"), warnings)
    compare_count("frames", result.get("info_total_frames"), result.get("observed_frames"), warnings)
    compare_count("indexed_frames", result.get("indexed_frames"), result.get("observed_frames"), warnings)

    return finish(result, started)


def detect_layout(raw_path: Path) -> str:
    episodes_dir = raw_path / "meta" / "episodes"
    if episodes_dir.is_dir() and (
        any(episodes_dir.glob("**/*.parquet")) or any(episodes_dir.glob("**/*.jsonl"))
    ):
        return "v3"
    if (raw_path / "meta" / "episodes.jsonl").exists():
        return "v2_1"
    raise FileNotFoundError(
        "could not detect LeRobot layout: expected meta/episodes/ or meta/episodes.jsonl"
    )


def load_episode_rows(raw_path: Path, layout: str) -> list[dict[str, Any]]:
    if layout == "v2_1":
        rows = []
        for line in (raw_path / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    rows: list[dict[str, Any]] = []
    parquets = sorted((raw_path / "meta" / "episodes").glob("**/*.parquet"))
    if parquets:
        for path in parquets:
            pq.read_metadata(path)
            rows.extend(pq.read_table(path).to_pylist())
        return rows
    for path in sorted((raw_path / "meta" / "episodes").glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_frame_tables(
    raw_path: Path,
    layout: str,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    *,
    expected_state_dim: int,
    expected_action_dim: int,
) -> int:
    chunks_size = int(info.get("chunks_size") or 1000) or 1000
    data_path_template = info.get("data_path")
    observed_frames = 0
    checked_paths: set[Path] = set()

    if layout == "v2_1":
        template = data_path_template or (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        for meta_row in episode_rows:
            episode_index = int(meta_row["episode_index"])
            chunk_index = episode_index // chunks_size
            rel = template.format(
                episode_chunk=chunk_index,
                episode_index=episode_index,
                chunk_index=chunk_index,
                file_index=episode_index,
            )
            path = raw_path / rel
            if not path.exists():
                errors.append(f"episode parquet missing: {rel}")
                continue
            observed_frames += validate_parquet_metadata(
                path,
                checked_paths,
                errors,
                expected_state_dim=expected_state_dim,
                expected_action_dim=expected_action_dim,
            )
        return observed_frames

    template = data_path_template or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    for meta_row in episode_rows:
        episode_index = int(meta_row.get("episode_index", 0))
        default_chunk = episode_index // chunks_size
        chunk_index = int_from_row(meta_row, "data/chunk_index", "chunk_index", default=default_chunk)
        file_index = int_from_row(meta_row, "data/file_index", "file_index", default=chunk_index)
        rel = template.format(chunk_index=chunk_index, file_index=file_index, episode_index=episode_index)
        path = raw_path / rel
        if not path.exists():
            errors.append(f"file parquet missing: {rel}")
            continue
        if path not in checked_paths:
            observed_frames += validate_parquet_metadata(
                path,
                checked_paths,
                errors,
                expected_state_dim=expected_state_dim,
                expected_action_dim=expected_action_dim,
            )

    lengths = [int(row["length"]) for row in episode_rows if row.get("length") is not None]
    if lengths:
        length_sum = sum(lengths)
        if observed_frames and observed_frames != length_sum:
            warnings.append(
                f"unique data parquet rows {observed_frames} != summed episode lengths {length_sum}"
            )
        return length_sum
    return observed_frames


def validate_parquet_metadata(
    path: Path,
    checked_paths: set[Path],
    errors: list[str],
    *,
    expected_state_dim: int,
    expected_action_dim: int,
) -> int:
    checked_paths.add(path)
    try:
        metadata = pq.read_metadata(path)
        schema = pq.read_schema(path)
        schema_names = set(schema.names)
    except Exception as exc:  # noqa: BLE001
        record_error(errors, f"bad parquet {path}: {type(exc).__name__}: {exc}")
        return 0
    missing = {"action", "observation.state"} - schema_names
    if missing:
        record_error(errors, f"parquet {path} missing columns: {sorted(missing)}")
    else:
        validate_vector_width(
            path,
            schema,
            "observation.state",
            expected_state_dim,
            errors,
        )
        validate_vector_width(path, schema, "action", expected_action_dim, errors)
    return int(metadata.num_rows or 0)


def validate_vector_width(
    path: Path,
    schema: Any,
    column: str,
    expected_dim: int,
    errors: list[str],
) -> None:
    observed = vector_width_from_type(schema.field(column).type)
    if observed is None:
        try:
            table = pq.read_table(path, columns=[column])
            values = table[column].to_pylist()
            for value in values:
                if value is not None:
                    observed = len(value)
                    break
        except Exception as exc:  # noqa: BLE001
            record_error(errors, f"could not sample {column} from {path}: {type(exc).__name__}: {exc}")
            return
    if observed is not None and observed != expected_dim:
        record_error(errors, f"{path} {column} width {observed} != expected {expected_dim}")


def vector_width_from_type(dtype: Any) -> int | None:
    width = getattr(dtype, "list_size", None)
    if isinstance(width, int) and width > 0:
        return width
    return None


def record_error(errors: list[str], message: str) -> None:
    if len(errors) < 200:
        errors.append(message)
    elif len(errors) == 200:
        errors.append("additional errors truncated")


def load_tasks(raw_path: Path) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    values: list[str] = []
    jsonl = raw_path / "meta" / "tasks.jsonl"
    if jsonl.exists():
        try:
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("task", "instruction", "language_instruction", "prompt", "text"):
                    if row.get(key) is not None:
                        values.append(str(row[key]))
                        break
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not read tasks.jsonl: {type(exc).__name__}: {exc}")
    parquet = raw_path / "meta" / "tasks.parquet"
    if parquet.exists():
        try:
            table = pq.read_table(parquet)
            for key in ("task", "instruction", "language_instruction", "prompt", "text"):
                if key in table.column_names:
                    values.extend(str(item) for item in table[key].to_pylist() if item is not None)
                    break
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not read tasks.parquet: {type(exc).__name__}: {exc}")

    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique, warnings


PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^(task|take)[_-]?\d+[a-z_0-9-]*$", re.I),
    re.compile(r"^test\d*$", re.I),
    re.compile(r"^test[_-]?.*$", re.I),
    re.compile(r"^default[_-]?task$", re.I),
    re.compile(r"^(good|dong)$", re.I),
    re.compile(r"^\d+$"),
]


def is_instructionish(text: str) -> bool:
    stripped = text.strip()
    if any(pattern.match(stripped) for pattern in PLACEHOLDER_PATTERNS):
        return False
    if " " not in stripped and re.fullmatch(r"[A-Za-z0-9_.:/-]+", stripped):
        lowered = stripped.lower()
        if any(token in lowered for token in ("task", "test", "dataset", "robotislab-real")):
            return False
    return len(re.findall(r"[A-Za-z가-힣]+", stripped)) >= 2


def dim_from_feature(feature: Any) -> tuple[int | None, Any]:
    shape = (feature or {}).get("shape") if isinstance(feature, dict) else None
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return int(shape[0]), shape
    return None, shape


def int_from_row(row: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        if key in row and row[key] is not None:
            return int(row[key])
    return int(default)


def compare_count(label: str, expected: Any, observed: Any, warnings: list[str]) -> None:
    try:
        expected_int = int(expected)
        observed_int = int(observed)
    except (TypeError, ValueError):
        return
    if expected_int != observed_int:
        warnings.append(f"{label} count mismatch: expected {expected_int}, observed {observed_int}")


def finish(result: dict[str, Any], started: float) -> dict[str, Any]:
    result["elapsed_s"] = round(time.time() - started, 3)
    if result.get("errors"):
        result["ok"] = False
        result["status"] = "bad_raw"
        result["quality_flag"] = "bad_raw"
    return result


def build_summary(results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    good = [row for row in results if row.get("ok")]
    bad = [row for row in results if not row.get("ok")]
    instructionish = [row for row in good if row.get("instructionish")]
    return {
        "raw_datasets": len(results),
        "good_datasets": len(good),
        "bad_datasets": len(bad),
        "good_frames": sum(int(row.get("observed_frames") or 0) for row in good),
        "bad_indexed_frames": sum(int(row.get("indexed_frames") or 0) for row in bad),
        "instructionish_good_datasets": len(instructionish),
        "instructionish_good_frames": sum(
            int(row.get("observed_frames") or 0) for row in instructionish
        ),
        "elapsed_s": round(time.time() - started, 3),
    }


def quality_flag(repo_id: Any) -> str:
    name = str(repo_id or "").lower()
    review_tokens = (
        "test",
        "upload",
        "your-new-dataset",
        "token",
        "demo_dataset",
    )
    return "review_name" if any(token in name for token in review_tokens) else "ok"


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def slug_repo_id(repo_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "__" for ch in repo_id) or "dataset"


def hf_dataset_url(repo_id: str) -> str | None:
    return f"https://huggingface.co/datasets/{repo_id}" if "/" in repo_id else None


def append_status(path: Path, result: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_progress(index: int, total: int, result: dict[str, Any]) -> None:
    status = "OK" if result.get("ok") else "BAD"
    frames = result.get("observed_frames") or result.get("indexed_frames") or ""
    print(f"[{index}/{total}] {status} {result.get('repo_id')} frames={frames}", flush=True)
    errors = result.get("errors") or []
    for error in errors[:3]:
        print(f"  error: {error}", flush=True)
    if len(errors) > 3:
        print(f"  ... {len(errors) - 3} more error(s)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
