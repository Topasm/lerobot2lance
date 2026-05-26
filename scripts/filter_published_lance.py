#!/usr/bin/env python3
"""Filter a published RLLAB Lance bundle while preserving row provenance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_pretrain_19d_lance import (  # noqa: E402
    DEFAULT_MAX_FPS,
    DEFAULT_MIN_EPISODE_FRAMES,
    DEFAULT_MIN_FPS,
    LANCE_DATA_STORAGE_VERSION,
    PUBLISHED_FORMAT,
    build_info,
    build_splits,
    build_tasks,
    compute_lerobot_stats,
    create_scalar_indexes,
    episode_filter_reason,
    is_blob_field,
    render_readme,
    scan_batches,
    table_from_pylist_with_blob_columns,
    write_episodes_jsonl,
    write_json,
    write_stats_sidecars,
    write_tasks_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter an existing published Lance bundle by episode length/FPS."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-id", default="rllab-postech/pretrain_aiworker_bg2_lance")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-fps", type=float, default=DEFAULT_MIN_FPS)
    parser.add_argument("--max-fps", type=float, default=DEFAULT_MAX_FPS)
    parser.add_argument("--min-episode-frames", type=int, default=DEFAULT_MIN_EPISODE_FRAMES)
    parser.add_argument("--max-episode-frames", type=int, default=0)
    parser.add_argument("--episode-batch-size", type=int, default=64)
    parser.add_argument("--frame-batch-size", type=int, default=100_000)
    parser.add_argument("--video-batch-size", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    import lance

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PUBLISHED_FORMAT:
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")

    episodes_path = table_path(source, manifest, "episodes")
    train_episodes_path = table_path(source, manifest, "train_episodes", fallback="episodes")
    frames_path = table_path(source, manifest, "frames")
    videos_path = table_path(source, manifest, "videos")

    episode_map, filtered_reasons = build_filtered_episode_map(
        lance.dataset(str(episodes_path)),
        min_episode_frames=args.min_episode_frames,
        max_episode_frames=args.max_episode_frames,
        min_fps=args.min_fps,
        max_fps=args.max_fps,
    )
    if not episode_map:
        raise SystemExit("No episodes remained after filters.")

    print(
        json.dumps(
            {
                "source_episodes": len(episode_map) + sum(filtered_reasons.values()),
                "kept_episodes": len(episode_map),
                "filtered": dict(filtered_reasons),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    write_episode_table(
        lance,
        episodes_path,
        output / "data" / "episodes.lance",
        episode_map,
        batch_size=args.episode_batch_size,
    )
    write_episode_table(
        lance,
        train_episodes_path,
        output / "data" / "train_episodes.lance",
        episode_map,
        batch_size=args.episode_batch_size,
    )
    frame_rows = write_frame_table(
        lance,
        frames_path,
        output / "data" / "frames.lance",
        episode_map,
        batch_size=args.frame_batch_size,
    )
    video_rows = write_video_table(
        lance,
        videos_path,
        output / "data" / "videos.lance",
        episode_map,
        batch_size=args.video_batch_size,
    )

    indexes_created = create_scalar_indexes(lance, output)
    manifest = update_manifest(
        manifest,
        dataset_id=args.dataset_id,
        episodes=len(episode_map),
        frames=frame_rows,
        videos=video_rows,
        indexes_created=indexes_created,
        filters={
            "min_fps": args.min_fps,
            "max_fps": args.max_fps,
            "min_episode_frames": args.min_episode_frames,
            "max_episode_frames": args.max_episode_frames,
            "filtered_reasons": dict(filtered_reasons),
        },
    )
    sessions = build_sessions(output, manifest)
    source_rows = build_source_rows(sessions, manifest)

    write_json(output / "manifest.json", manifest)
    write_json(output / "meta" / "sessions.json", sessions)
    write_json(output / "meta" / "info.json", build_info(args.dataset_id, manifest, source_rows))
    tasks_payload = build_tasks(output / "data" / "episodes.lance")
    stats_payload = compute_lerobot_stats(output / "data" / "train_episodes.lance")
    write_tasks_jsonl(output / "meta" / "tasks.jsonl", tasks_payload)
    write_episodes_jsonl(output / "meta" / "episodes.jsonl", output / "data" / "episodes.lance")
    write_json(output / "meta" / "splits.json", build_splits(len(episode_map)))
    write_stats_sidecars(output / "meta", stats_payload)
    (output / "README.md").write_text(
        render_readme(args.dataset_id, manifest, sessions),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(episode_map),
                "frames": frame_rows,
                "videos": video_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def table_path(root: Path, manifest: dict[str, Any], name: str, *, fallback: str | None = None) -> Path:
    tables = manifest.get("tables") or {}
    entry = tables.get(name) or (tables.get(fallback) if fallback else None)
    if isinstance(entry, dict):
        rel = entry.get("path")
    else:
        rel = entry
    if not rel:
        raise ValueError(f"manifest.tables.{name} is missing")
    return root / str(rel)


def build_filtered_episode_map(
    episodes: Any,
    *,
    min_episode_frames: int,
    max_episode_frames: int,
    min_fps: float,
    max_fps: float,
) -> tuple[dict[int, int], Counter[str]]:
    columns = [name for name in ("episode_index", "length", "timestamps", "fps") if name in episodes.schema.names]
    mapping: dict[int, int] = {}
    reasons: Counter[str] = Counter()
    for row in scan_rows_local(episodes, columns=columns, batch_size=4096):
        source_episode = int(row["episode_index"])
        reason = filter_reason(
            row,
            min_episode_frames=min_episode_frames,
            max_episode_frames=max_episode_frames,
            min_fps=min_fps,
            max_fps=max_fps,
        )
        if reason:
            reasons[reason] += 1
            continue
        mapping[source_episode] = len(mapping)
    return mapping, reasons


def filter_reason(
    row: dict[str, Any],
    *,
    min_episode_frames: int,
    max_episode_frames: int,
    min_fps: float,
    max_fps: float,
) -> str | None:
    return episode_filter_reason(
        row,
        min_episode_frames=min_episode_frames,
        max_episode_frames=max_episode_frames,
        min_fps=min_fps,
        max_fps=max_fps,
    )


def write_episode_table(
    lance: Any,
    source_path: Path,
    target_path: Path,
    episode_map: dict[int, int],
    *,
    batch_size: int,
) -> int:
    ds = lance.dataset(str(source_path))
    total = 0
    mode = "overwrite"
    for batch in scan_batches(ds, batch_size=batch_size):
        rows = []
        for row in batch.to_pylist():
            old_episode = int(row["episode_index"])
            new_episode = episode_map.get(old_episode)
            if new_episode is None:
                continue
            row["episode_index"] = new_episode
            rows.append(row)
        if rows:
            write_rows(lance, rows, ds.schema, target_path, mode=mode)
            mode = "append"
            total += len(rows)
    return total


def write_frame_table(
    lance: Any,
    source_path: Path,
    target_path: Path,
    episode_map: dict[int, int],
    *,
    batch_size: int,
) -> int:
    ds = lance.dataset(str(source_path))
    total = 0
    mode = "overwrite"
    for batch in scan_batches(ds, batch_size=batch_size):
        rows = []
        for row in batch.to_pylist():
            old_episode = int(row["episode_index"])
            new_episode = episode_map.get(old_episode)
            if new_episode is None:
                continue
            row["episode_index"] = new_episode
            row["global_frame_index"] = total + len(rows)
            rows.append(row)
        if rows:
            write_rows(lance, rows, ds.schema, target_path, mode=mode)
            mode = "append"
            total += len(rows)
    return total


def write_video_table(
    lance: Any,
    source_path: Path,
    target_path: Path,
    episode_map: dict[int, int],
    *,
    batch_size: int,
) -> int:
    ds = lance.dataset(str(source_path))
    source_columns = set(ds.schema.names)
    blob_columns = {field.name for field in ds.schema if is_blob_field(field)}
    total = 0
    source_row_index = 0
    mode = "overwrite"
    for batch in scan_batches(ds, batch_size=batch_size):
        rows = []
        for row in batch.to_pylist():
            this_row_index = source_row_index
            source_row_index += 1
            old_episode = int(row["episode_index"])
            new_episode = episode_map.get(old_episode)
            if new_episode is None:
                continue
            row["episode_index"] = new_episode
            materialize_blobs(ds, row, source_columns, blob_columns, this_row_index)
            rows.append(row)
        if rows:
            write_rows(lance, rows, ds.schema, target_path, mode=mode)
            mode = "append"
            total += len(rows)
    return total


def write_rows(lance: Any, rows: list[dict[str, Any]], schema: Any, target_path: Path, *, mode: str) -> None:
    import pyarrow as pa

    blob_columns = {field.name for field in schema if is_blob_field(field)}
    table = table_from_pylist_with_blob_columns(
        pa,
        lance,
        rows,
        schema=schema,
        blob_columns=blob_columns,
    )
    lance.write_dataset(
        table,
        str(target_path),
        mode=mode,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )


def materialize_blobs(
    ds: Any,
    row: dict[str, Any],
    source_columns: set[str],
    blob_columns: set[str],
    source_row_index: int,
) -> None:
    for column in blob_columns:
        value = row.get(column)
        if value is None or isinstance(value, (bytes, bytearray, memoryview)):
            continue
        if column not in source_columns:
            row[column] = None
            continue
        handles = ds.take_blobs(column, indices=[source_row_index])
        if not handles:
            row[column] = None
            continue
        handle = handles[0]
        try:
            row[column] = handle.readall() if hasattr(handle, "readall") else handle.read()
        finally:
            handle.close()


def scan_rows_local(ds: Any, *, columns: list[str], batch_size: int) -> Any:
    for batch in ds.scanner(columns=columns, batch_size=batch_size).to_batches():
        yield from batch.to_pylist()


def update_manifest(
    manifest: dict[str, Any],
    *,
    dataset_id: str,
    episodes: int,
    frames: int,
    videos: int,
    indexes_created: list[dict[str, Any]],
    filters: dict[str, Any],
) -> dict[str, Any]:
    out = dict(manifest)
    out["dataset_id"] = dataset_id
    out["created_at"] = datetime.now(timezone.utc).isoformat()
    out["primary_training_table"] = "data/train_episodes.lance"
    out["quality_filters"] = filters
    out["counts"] = {
        **(out.get("counts") or {}),
        "episodes": episodes,
        "frames": frames,
        "videos": videos,
    }
    out["tables"] = {
        "episodes": {"path": "data/episodes.lance", "exists": True},
        "train_episodes": {"path": "data/train_episodes.lance", "exists": True},
        "primary_training": {"path": "data/train_episodes.lance", "exists": True},
        "frames": {"path": "data/frames.lance", "exists": True},
        "videos": {"path": "data/videos.lance", "exists": True},
    }
    out["indexes"] = {
        **(out.get("indexes") or {}),
        "created": indexes_created,
    }
    return out


def build_sessions(output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import lance

    episodes = lance.dataset(str(output / "data" / "episodes.lance"))
    videos = lance.dataset(str(output / "data" / "videos.lance"))
    ep_rows = episodes.to_table(
        columns=[
            "episode_index",
            "source_dataset",
            "source_dataset_url",
            "source_robot_type",
            "source_robot_name",
            "pretrain_tier",
            "quality_flag",
            "fps",
            "length",
        ]
    ).to_pylist()
    video_counts = Counter(
        row.get("source_dataset")
        for row in videos.to_table(columns=["source_dataset"]).to_pylist()
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ep_rows:
        grouped[str(row.get("source_dataset") or "unknown")].append(row)

    sessions = []
    for source_id, (source_dataset, rows) in enumerate(
        sorted(grouped.items(), key=lambda item: min(int(row["episode_index"]) for row in item[1]))
    ):
        first = min(rows, key=lambda row: int(row["episode_index"]))
        start = min(int(row["episode_index"]) for row in rows)
        end = max(int(row["episode_index"]) for row in rows) + 1
        sessions.append(
            {
                "source_id": source_id,
                "source_dataset": source_dataset,
                "source_url": first.get("source_dataset_url"),
                "robot_type": first.get("source_robot_type"),
                "robot_name": first.get("source_robot_name"),
                "pretrain_tier": first.get("pretrain_tier"),
                "quality_flag": first.get("quality_flag"),
                "fps": first.get("fps"),
                "episode_start": start,
                "episode_end": end,
                "episodes": len(rows),
                "frames": sum(int(row.get("length") or 0) for row in rows),
                "videos": int(video_counts.get(source_dataset, 0)),
            }
        )
    manifest["counts"]["sources"] = len(sessions)
    return sessions


def build_source_rows(sessions: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    camera_keys = []
    camera_columns = []
    for entry in (manifest.get("modalities") or {}).values():
        if isinstance(entry, dict) and entry.get("kind") == "video":
            camera_keys.append(entry.get("camera_key"))
            camera_columns.append(entry.get("camera_column"))
    return [
        {
            **session,
            "state_dim": 19,
            "action_dim": 19,
            "source_camera_keys": camera_keys,
            "source_camera_columns": camera_columns,
        }
        for session in sessions
    ]


if __name__ == "__main__":
    raise SystemExit(main())
