#!/usr/bin/env python3
"""Merge converted 19D AI Worker/BG2 Lance bundles into one pretrain bundle.

Input bundles are expected to already be in the `lerobot2lance --layout hf`
published layout:

    <bundle>/manifest.json
    <bundle>/data/episodes.lance
    <bundle>/data/frames.lance
    <bundle>/data/videos.lance

The merged output keeps the same published layout contract, but rewrites
episode/global frame indices so multiple source datasets can be trained as one
dataset without index collisions. Source provenance is recorded both in row
columns and in `meta/sessions.json` / README.md.

There is one media contract: trajectory tables contain numeric/text episode
data, and `data/videos.lance` is the only MP4 store. The builder always
re-materializes `data/videos.lance.video_blob` from source bundles and never
writes `*_video_blob` columns into `episodes.lance` or `train_episodes.lance`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any


PUBLISHED_FORMAT = "rllab_published_lance_dataset_v2"
PUBLISHED_LAYOUT = "rllab_published_dataset_v2"
DEFAULT_DATASET_ID = "rllab-postech/pretraining-aiworker-bg2-19d"
LANCE_DATA_STORAGE_VERSION = "2.2"
LANCE_BLOB_ENCODING = "lance.blob.v2"
PUBLISHED_BLOB_POLICY = "inline_bytes_only"
STATE_DIM = 19
ACTION_DIM = 19
SCALAR_INDEXES: dict[str, list[tuple[str, str]]] = {
    "episodes": [
        ("episode_index", "BTREE"),
        ("task_index", "BITMAP"),
        ("source_dataset", "BITMAP"),
    ],
    "train_episodes": [
        ("episode_index", "BTREE"),
        ("task_index", "BITMAP"),
        ("source_dataset", "BITMAP"),
    ],
    "frames": [
        ("global_frame_index", "BTREE"),
        ("episode_index", "BTREE"),
        ("frame_index", "BTREE"),
        ("task_index", "BITMAP"),
        ("is_bad_frame", "BITMAP"),
        ("source_dataset", "BITMAP"),
    ],
    "videos": [
        ("media_id", "BTREE"),
        ("episode_index", "BTREE"),
        ("camera_id", "BITMAP"),
        ("source_dataset", "BITMAP"),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a merged 19D AI Worker/BG2 pretrain Lance bundle."
    )
    parser.add_argument("--converted-root", default="data/converted_19d")
    parser.add_argument("--output", default="data/pretrain_aiworker_19d")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-bg2-only", action="store_true")
    parser.add_argument(
        "--include-review-names",
        action="store_true",
        help="Include source repos whose names look like TEST/upload scratch data.",
    )
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Rows per write batch.",
    )
    args = parser.parse_args()

    converted_root = Path(args.converted_root)
    output = Path(args.output)
    if not converted_root.exists():
        raise FileNotFoundError(f"Converted root not found: {converted_root}")

    sources = discover_sources(
        converted_root,
        strict_bg2_only=args.strict_bg2_only,
        include_review_names=args.include_review_names,
    )
    if args.limit_sources is not None:
        sources = sources[: args.limit_sources]
    if not sources:
        raise SystemExit(f"No eligible 19D converted bundles found under {converted_root}")

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    import lance
    import pyarrow as pa

    source_camera_keys, source_camera_columns = union_cameras(sources)
    camera_keys = source_camera_keys
    camera_columns = source_camera_columns
    episodes_schema = build_episodes_schema(pa)
    train_episodes_schema = build_train_episodes_schema(
        pa,
        source_camera_columns,
    )
    frames_schema = build_frames_schema(pa)
    videos_schema = build_videos_schema(pa, lance)

    sessions: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    episode_offset = 0
    frame_offset = 0
    video_count = 0

    for source_id, source in enumerate(sources):
        print(
            f"[{source_id + 1}/{len(sources)}] {source['source_repo_id']} "
            f"episodes={source['episodes']} frames={source['frames']}",
            flush=True,
        )
        bundle = Path(source["path"])
        episode_map = build_episode_map(bundle / "data" / "episodes.lance", episode_offset)
        write_remapped_table(
            lance=lance,
            source_path=bundle / "data" / "episodes.lance",
            target_path=output / "data" / "episodes.lance",
            target_schema=episodes_schema,
            batch_size=args.batch_size,
            transform=lambda row, n=source, sid=source_id, emap=episode_map: transform_episode_metadata(
                row,
                n,
                sid,
                emap,
                source_camera_columns,
            ),
            copy_blobs=False,
        )
        write_remapped_table(
            lance=lance,
            source_path=bundle / "data" / "episodes.lance",
            target_path=output / "data" / "train_episodes.lance",
            target_schema=train_episodes_schema,
            batch_size=args.batch_size,
            transform=lambda row, n=source, sid=source_id, emap=episode_map: transform_train_episode(
                row,
                n,
                sid,
                emap,
                source_camera_columns,
            ),
            copy_blobs=False,
        )
        if (bundle / "data" / "frames.lance").exists():
            local_frame_count = write_remapped_table(
                lance=lance,
                source_path=bundle / "data" / "frames.lance",
                target_path=output / "data" / "frames.lance",
                target_schema=frames_schema,
                batch_size=max(args.batch_size * 128, 1024),
                transform=lambda row, n=source, sid=source_id, emap=episode_map: transform_frame(
                    row, n, sid, emap
                ),
                global_frame_start=frame_offset,
                copy_blobs=False,
            )
        else:
            local_frame_count = 0
        if (bundle / "data" / "videos.lance").exists():
            local_video_count = write_remapped_table(
                lance=lance,
                source_path=bundle / "data" / "videos.lance",
                target_path=output / "data" / "videos.lance",
                target_schema=videos_schema,
                batch_size=args.batch_size,
                transform=lambda row, n=source, sid=source_id, emap=episode_map: transform_video(
                    row,
                    n,
                    sid,
                    emap,
                ),
                copy_blobs=True,
            )
        else:
            local_video_count = 0

        source_episode_count = len(episode_map)
        session = {
            "source_id": source_id,
            "source_dataset": source["source_repo_id"],
            "source_url": hf_dataset_url(source["source_repo_id"]),
            "local_path": str(bundle),
            "robot_type": source.get("robot_type"),
            "robot_name": source.get("robot_name"),
            "pretrain_tier": source.get("pretrain_tier"),
            "quality_flag": source.get("quality_flag"),
            "fps": source.get("fps"),
            "episode_start": episode_offset,
            "episode_end": episode_offset + source_episode_count,
            "episodes": source_episode_count,
            "frames": local_frame_count,
            "videos": local_video_count,
        }
        sessions.append(session)
        source_rows.append(
            {
                **session,
                "state_dim": source.get("state_dim"),
                "action_dim": source.get("action_dim"),
                "source_camera_keys": source.get("camera_keys") or [],
                "source_camera_columns": source.get("camera_columns") or [],
            }
        )
        episode_offset += source_episode_count
        frame_offset += local_frame_count
        video_count += local_video_count

    manifest = build_manifest(
        dataset_id=args.dataset_id,
        sources=sources,
        camera_keys=camera_keys,
        camera_columns=camera_columns,
        source_camera_keys=source_camera_keys,
        source_camera_columns=source_camera_columns,
        episodes=episode_offset,
        frames=frame_offset,
        videos=video_count,
        indexes_created=create_scalar_indexes(lance, output),
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "meta" / "manifest.json", manifest)
    write_json(output / "meta" / "sessions.json", sessions)
    write_json(output / "meta" / "sources.json", source_rows)
    write_json(output / "meta" / "info.json", build_info(args.dataset_id, manifest, source_rows))
    tasks_payload = build_tasks(output / "data" / "episodes.lance")
    stats_payload = compute_lerobot_stats(output / "data" / "train_episodes.lance")
    write_tasks_jsonl(output / "meta" / "tasks.jsonl", tasks_payload)
    write_episodes_jsonl(output / "meta" / "episodes.jsonl", output / "data" / "episodes.lance")
    write_json(output / "meta" / "splits.json", build_splits(episode_offset))
    write_stats_sidecars(output / "meta", stats_payload)
    (output / "README.md").write_text(render_readme(args.dataset_id, manifest, sessions), encoding="utf-8")

    print(f"wrote {output}", flush=True)
    print(
        json.dumps(
            {
                "datasets": len(sources),
                "episodes": episode_offset,
                "frames": frame_offset,
                "videos": video_count,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def discover_sources(
    converted_root: Path,
    *,
    strict_bg2_only: bool,
    include_review_names: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(converted_root.glob("*/manifest.json")):
        bundle = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != PUBLISHED_FORMAT:
            continue
        state_dim = registry_shape_dim(manifest, "modalities", "state.body")
        action_dim = registry_shape_dim(manifest, "actions", "action.body")
        if state_dim != 19:
            continue
        if action_dim != 19:
            continue
        info = read_json_if_exists(bundle / "meta" / "info.json")
        session = first_session(bundle)
        camera_keys, camera_columns = cameras_from_manifest(manifest)
        pretrain_tier = str(manifest.get("pretrain_tier") or "")
        robot_type = info.get("robot_type") or manifest.get("source_robot_type")
        robot_name = info.get("robot_name") or manifest.get("source_robot_name")
        is_bg2 = (
            robot_type == "ffw_bg2_rev4"
            or pretrain_tier.startswith("A_bg2_full")
        )
        if strict_bg2_only and not is_bg2:
            continue
        repo_id = (
            session.get("source_dataset")
            or session.get("source_repo_id")
            or info.get("repo_id")
            or manifest.get("source_repo_id")
            or manifest.get("dataset_id")
        )
        quality = quality_flag(repo_id)
        if quality != "ok" and not include_review_names:
            continue
        if not (bundle / "data" / "episodes.lance").exists():
            continue
        rows.append(
            {
                "path": str(bundle),
                "dataset_id": manifest.get("dataset_id") or bundle.name,
                "source_repo_id": repo_id,
                "robot_type": robot_type,
                "robot_name": robot_name,
                "pretrain_tier": pretrain_tier,
                "quality_flag": quality,
                "fps": (manifest.get("rates") or {}).get("fps") or info.get("fps"),
                "episodes": int((manifest.get("counts") or {}).get("episodes") or 0),
                "frames": int((manifest.get("counts") or {}).get("frames") or 0),
                "videos": int((manifest.get("counts") or {}).get("videos") or 0),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "camera_keys": camera_keys,
                "camera_columns": camera_columns,
            }
        )
    rows.sort(key=lambda row: (str(row.get("source_repo_id")), str(row.get("path"))))
    return rows


def union_cameras(sources: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    by_column: dict[str, str] = {}
    for source in sources:
        keys = list(source.get("camera_keys") or [])
        columns = list(source.get("camera_columns") or [])
        for index, column in enumerate(columns):
            key = keys[index] if index < len(keys) else column
            by_column.setdefault(str(column), str(key))
    columns_sorted = sorted(by_column)
    keys_sorted = [by_column[column] for column in columns_sorted]
    return keys_sorted, columns_sorted


def registry_shape_dim(manifest: dict[str, Any], section: str, name: str) -> int:
    entry = ((manifest.get(section) or {}).get(name) or {})
    shape = entry.get("shape") or []
    if isinstance(shape, list) and shape:
        return int(shape[0])
    return 0


def cameras_from_manifest(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    cameras: list[tuple[str, str]] = []
    for entry in (manifest.get("modalities") or {}).values():
        if not isinstance(entry, dict) or entry.get("kind") != "video":
            continue
        key = entry.get("camera_key") or entry.get("source_key")
        column = entry.get("camera_column")
        if key and column:
            cameras.append((str(key), str(column)))
    cameras.sort(key=lambda item: item[1])
    return [key for key, _ in cameras], [column for _, column in cameras]


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def first_session(bundle: Path) -> dict[str, Any]:
    path = bundle / "meta" / "sessions.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list) and sessions and isinstance(sessions[0], dict):
            return sessions[0]
    return {}


def build_tasks(episodes_path: Path) -> dict[str, Any]:
    import lance

    ds = lance.dataset(str(episodes_path))
    tasks: dict[int, dict[str, Any]] = {}
    for row in scan_rows(
        ds,
        columns=["episode_index", "task_index", "language_instruction"],
    ):
        task_index = as_int(row.get("task_index")) or 0
        task = tasks.setdefault(
            task_index,
            {
                "task_index": task_index,
                "language_instruction": None,
                "episode_count": 0,
            },
        )
        task["episode_count"] += 1
        language = row.get("language_instruction")
        if isinstance(language, str) and language and not task["language_instruction"]:
            task["language_instruction"] = language
    return {
        "schema_version": "2.0",
        "tasks": [tasks[index] for index in sorted(tasks)],
    }


def compute_lerobot_stats(table_path: Path) -> dict[str, Any]:
    import lance

    states: list[list[float]] = []
    actions: list[list[float]] = []
    ds = lance.dataset(str(table_path))
    for row in scan_rows(ds, columns=["observation_state", "actions"]):
        states.extend(vector_rows(row.get("observation_state")))
        actions.extend(vector_rows(row.get("actions")))
    return {
        "observation.state": vector_stats(states),
        "action": vector_stats(actions),
    }


def write_stats_sidecars(meta_dir: Path, stats: dict[str, Any]) -> None:
    stats_dir = meta_dir / "stats"
    write_json(
        stats_dir / "state_body.json",
        {
            "schema_version": "2.0",
            "modality": "state.body",
            "feature": "observation.state",
            **stats.get("observation.state", {}),
        },
    )
    write_json(
        stats_dir / "action_body.json",
        {
            "schema_version": "2.0",
            "action": "action.body",
            "feature": "action",
            **stats.get("action", {}),
        },
    )


def write_tasks_jsonl(path: Path, tasks_payload: dict[str, Any]) -> None:
    lines = [
        json.dumps(
            {
                "task_index": row["task_index"],
                "task": row.get("language_instruction"),
                "language_instruction": row.get("language_instruction"),
                "episode_count": row.get("episode_count", 0),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in tasks_payload.get("tasks", [])
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_episodes_jsonl(path: Path, episodes_path: Path) -> None:
    import lance

    ds = lance.dataset(str(episodes_path))
    lines = []
    for row in scan_rows(
        ds,
        columns=["episode_index", "task_index", "language_instruction", "length"],
    ):
        language = row.get("language_instruction")
        lines.append(
            json.dumps(
                {
                    "episode_index": as_int(row.get("episode_index")) or 0,
                    "task_index": as_int(row.get("task_index")) or 0,
                    "tasks": [language] if language else [],
                    "length": as_int(row.get("length")) or 0,
                    "split": split_for_episode(as_int(row.get("episode_index")) or 0, ds.count_rows()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_splits(episode_count: int) -> dict[str, Any]:
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for episode_index in range(episode_count):
        splits[split_for_episode(episode_index, episode_count)].append(episode_index)
    return {
        "schema_version": "2.0",
        "strategy": "deterministic_90_10_0_by_episode_index",
        "ratios": {"train": 0.9, "val": 0.1, "test": 0.0},
        "splits": {key: value for key, value in splits.items() if value},
    }


def split_for_episode(episode_index: int, episode_count: int) -> str:
    if episode_count < 10:
        return "train"
    train_count = int(math.floor(episode_count * 0.9))
    return "train" if episode_index < train_count else "val"


def build_episode_map(episodes_path: Path, episode_offset: int) -> dict[int, int]:
    import lance

    mapping: dict[int, int] = {}
    ds = lance.dataset(str(episodes_path))
    for row in scan_rows(ds, columns=["episode_index"]):
        source_episode = int(row["episode_index"])
        if source_episode not in mapping:
            mapping[source_episode] = episode_offset + len(mapping)
    return mapping


def write_remapped_table(
    *,
    lance: Any,
    source_path: Path,
    target_path: Path,
    target_schema: Any,
    batch_size: int,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
    global_frame_start: int | None = None,
    copy_blobs: bool = False,
) -> int:
    import pyarrow as pa

    ds = lance.dataset(str(source_path))
    source_columns = set(ds.schema.names)
    blob_columns = {field.name for field in target_schema if is_blob_field(field)}
    total_rows = 0
    mode = "append" if target_path.exists() else "overwrite"
    for batch in scan_batches(ds, batch_size=batch_size):
        rows = []
        for row in batch.to_pylist():
            source_row_index = total_rows
            out = transform(row)
            if global_frame_start is not None:
                out["source_global_frame_index"] = as_int(row.get("global_frame_index"))
                out["global_frame_index"] = global_frame_start + total_rows
            if copy_blobs:
                materialize_blobs(ds, out, source_columns, blob_columns, source_row_index)
            elif blob_columns:
                clear_blobs(out, blob_columns)
            rows.append(out)
            total_rows += 1
        if not rows:
            continue
        table = table_from_pylist_with_blob_columns(
            pa,
            lance,
            rows,
            schema=target_schema,
            blob_columns=blob_columns,
        )
        lance.write_dataset(
            table,
            str(target_path),
            mode=mode,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
        mode = "append"
    if total_rows:
        assert_lance_storage_version(lance, target_path)
    return total_rows


def assert_lance_storage_version(lance: Any, path: Path) -> None:
    ds = lance.dataset(str(path))
    if str(getattr(ds, "data_storage_version", "")) != LANCE_DATA_STORAGE_VERSION:
        raise RuntimeError(
            f"{path} was not written with Lance data_storage_version "
            f"{LANCE_DATA_STORAGE_VERSION}"
        )


def is_blob_field(field: Any) -> bool:
    if field.metadata and field.metadata.get(b"lance-encoding:blob") == b"true":
        return True
    return getattr(field.type, "extension_name", None) == "lance.blob.v2"


def table_from_pylist_with_blob_columns(
    pa: Any,
    lance: Any,
    rows: list[dict[str, Any]],
    *,
    schema: Any,
    blob_columns: set[str],
) -> Any:
    if not blob_columns:
        return pa.Table.from_pylist(rows, schema=schema)
    arrays = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if field.name in blob_columns:
            validate_inline_blob_values(values, field.name)
            arrays.append(lance.blob_array(values))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def validate_inline_blob_values(values: list[Any], column: str) -> None:
    for value in values:
        if value is None or isinstance(value, (bytes, bytearray, memoryview)):
            continue
        raise TypeError(
            f"{column} must contain inline bytes only for published v2 bundles; "
            f"got {type(value).__name__}"
        )


def create_scalar_indexes(lance: Any, output: Path) -> list[dict[str, Any]]:
    table_paths = {
        "episodes": output / "data" / "episodes.lance",
        "train_episodes": output / "data" / "train_episodes.lance",
        "frames": output / "data" / "frames.lance",
        "videos": output / "data" / "videos.lance",
    }
    created: list[dict[str, Any]] = []
    for table_name, path in table_paths.items():
        if not path.exists():
            continue
        ds = lance.dataset(str(path))
        available = set(ds.schema.names)
        grouped: dict[str, list[str]] = {}
        for column, index_type in SCALAR_INDEXES.get(table_name, []):
            if column not in available:
                continue
            try:
                ds.create_scalar_index(column, index_type=index_type)
            except Exception:
                continue
            grouped.setdefault(index_type, []).append(column)
        for index_type, columns in grouped.items():
            created.append(
                {
                    "table": f"data/{path.name}",
                    "index_type": index_type,
                    "columns": columns,
                    "status": "ready",
                }
            )
    return created


def clear_blobs(row: dict[str, Any], blob_columns: set[str]) -> None:
    for column in blob_columns:
        row[column] = None


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


def scan_rows(ds: Any, *, columns: list[str]) -> Iterable[dict[str, Any]]:
    for batch in scan_batches(ds, batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def scan_batches(ds: Any, *, batch_size: int, columns: list[str] | None = None) -> Iterable[Any]:
    scanner = ds.scanner(columns=columns, batch_size=batch_size)
    yield from scanner.to_batches()


def transform_episode_metadata(
    row: dict[str, Any],
    source: dict[str, Any],
    source_id: int,
    episode_map: dict[int, int],
    camera_columns: list[str],
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    episode_index = episode_map[source_episode]
    timestamps = row.get("timestamps") or []
    states = row.get("observation_state") or []
    actions = row.get("actions") or []
    return {
        "episode_index": episode_index,
        "task_index": as_int(row.get("task_index")) or 0,
        "fps": as_float(row.get("fps")),
        "length": as_int(row.get("length")) or 0,
        "timestamps": timestamps,
        "observation_state": states,
        "actions": actions,
        "language_instruction": row.get("language_instruction"),
        "camera_segments": camera_segments_for_row(
            row,
            source_id=source_id,
            episode_index=episode_index,
            camera_columns=camera_columns,
        ),
        "task_segments": task_segments_for_row(row, timestamps=timestamps),
        "trajectory_sha256": trajectory_sha256(timestamps, states, actions),
        "split": row.get("split") or "train",
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
        "session_id": row.get("session_id") or source.get("source_repo_id"),
        "embodiment_id": row.get("embodiment_id") or source.get("robot_type"),
        "source_robot_type": source.get("robot_type"),
        "source_robot_name": source.get("robot_name"),
        "pretrain_tier": source.get("pretrain_tier"),
        "quality_flag": source.get("quality_flag"),
    }


def transform_train_episode(
    row: dict[str, Any],
    source: dict[str, Any],
    source_id: int,
    episode_map: dict[int, int],
    camera_columns: list[str],
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    episode_index = episode_map[source_episode]
    timestamps = row.get("timestamps") or []
    states = row.get("observation_state") or []
    actions = row.get("actions") or []
    out = {
        "episode_index": episode_index,
        "task_index": as_int(row.get("task_index")) or 0,
        "fps": as_float(row.get("fps")),
        "length": as_int(row.get("length")) or 0,
        "timestamps": timestamps,
        "observation_state": states,
        "actions": actions,
        "language_instruction": row.get("language_instruction"),
        "camera_segments": camera_segments_for_row(
            row,
            source_id=source_id,
            episode_index=episode_index,
            camera_columns=camera_columns,
        ),
        "task_segments": task_segments_for_row(row, timestamps=timestamps),
        "trajectory_sha256": trajectory_sha256(timestamps, states, actions),
        "split": row.get("split") or "train",
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
        "session_id": row.get("session_id") or source.get("source_repo_id"),
        "embodiment_id": row.get("embodiment_id") or source.get("robot_type"),
        "source_robot_type": source.get("robot_type"),
        "source_robot_name": source.get("robot_name"),
        "pretrain_tier": source.get("pretrain_tier"),
        "quality_flag": source.get("quality_flag"),
    }
    return out


def transform_frame(
    row: dict[str, Any],
    source: dict[str, Any],
    source_id: int,
    episode_map: dict[int, int],
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    return {
        "episode_index": episode_map[source_episode],
        "frame_index": as_int(row.get("frame_index")) or 0,
        "global_frame_index": 0,
        "timestamp": as_float(row.get("timestamp")),
        "task_index": as_int(row.get("task_index")) or 0,
        "observation_state": row.get("observation_state") or [],
        "action": row.get("action") or [],
        "state_norm": as_float(row.get("state_norm")),
        "action_norm": as_float(row.get("action_norm")),
        "is_bad_frame": bool(row.get("is_bad_frame", False)),
        "split": row.get("split") or "train",
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
        "session_id": row.get("session_id") or source.get("source_repo_id"),
        "embodiment_id": row.get("embodiment_id") or source.get("robot_type"),
        "source_frame_index": as_int(row.get("frame_index")) or 0,
        "source_global_frame_index": as_int(row.get("global_frame_index")),
        "source_robot_type": source.get("robot_type"),
        "source_robot_name": source.get("robot_name"),
        "pretrain_tier": source.get("pretrain_tier"),
        "quality_flag": source.get("quality_flag"),
    }


def target_media_id(source_id: int, episode_index: int, camera_id: Any) -> str:
    return f"source_{source_id:05d}_episode_{episode_index:08d}_{camera_id or ''}"


def camera_segments_for_row(
    row: dict[str, Any],
    *,
    source_id: int,
    episode_index: int,
    camera_columns: list[str],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in row.get("camera_segments") or []:
        camera_column = segment.get("camera_column") or segment.get("camera_id")
        if camera_column not in camera_columns:
            continue
        segments.append(
            {
                **segment,
                "media_id": target_media_id(source_id, episode_index, camera_column),
            }
        )
    return segments


def task_segments_for_row(row: dict[str, Any], *, timestamps: list[Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("task_segments"), list) and row["task_segments"]:
        return row["task_segments"]
    length = as_int(row.get("length")) or len(timestamps)
    if length <= 0:
        return []
    return [
        {
            "task_index": as_int(row.get("task_index")) or 0,
            "language_instruction": row.get("language_instruction"),
            "start_frame": 0,
            "end_frame_exclusive": length,
            "start_timestamp": as_float(timestamps[0]) if timestamps else None,
            "end_timestamp_exclusive": end_timestamp_exclusive(timestamps),
        }
    ]


TRAJECTORY_HASH_MAGIC = b"RLLAB_TRAJECTORY_V1\x00"


def trajectory_sha256(
    timestamps: list[Any],
    states: list[Any],
    actions: list[Any],
) -> str:
    length = len(timestamps)
    if len(states) != length:
        raise ValueError(f"observation_state length {len(states)} != timestamps length {length}")
    if len(actions) != length:
        raise ValueError(f"actions length {len(actions)} != timestamps length {length}")
    state_dim = len(states[0]) if states and states[0] else 0
    action_dim = len(actions[0]) if actions and actions[0] else 0
    chunks: list[bytes] = [
        TRAJECTORY_HASH_MAGIC,
        struct.pack("<q", length),
        struct.pack("<i", state_dim),
        struct.pack("<i", action_dim),
    ]
    if length:
        chunks.append(
            struct.pack(
                f"<{length}d",
                *(
                    finite_float(t, f"timestamps[{index}]")
                    for index, t in enumerate(timestamps)
                ),
            )
        )
    if state_dim:
        flat_state: list[float] = []
        for row_index, row in enumerate(states):
            if len(row) != state_dim:
                raise ValueError(
                    f"observation_state row width {len(row)} != state_dim {state_dim}"
                )
            flat_state.extend(
                finite_float(v, f"observation_state[{row_index}]")
                for v in row
            )
        chunks.append(struct.pack(f"<{len(flat_state)}f", *flat_state))
    if action_dim:
        flat_action: list[float] = []
        for row_index, row in enumerate(actions):
            if len(row) != action_dim:
                raise ValueError(
                    f"action row width {len(row)} != action_dim {action_dim}"
                )
            flat_action.extend(
                finite_float(v, f"actions[{row_index}]")
                for v in row
            )
        chunks.append(struct.pack(f"<{len(flat_action)}f", *flat_action))
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} contains non-finite value {value!r}")
    return out


def end_timestamp_exclusive(timestamps: list[Any]) -> float | None:
    if not timestamps:
        return None
    if len(timestamps) == 1:
        return as_float(timestamps[0])
    last = as_float(timestamps[-1])
    prev = as_float(timestamps[-2])
    if last is None or prev is None:
        return last
    return last + (last - prev)


def transform_video(
    row: dict[str, Any],
    source: dict[str, Any],
    source_id: int,
    episode_map: dict[int, int],
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    episode_index = episode_map[source_episode]
    camera_id = row.get("camera_id") or ""
    source_struct = row.get("source") if isinstance(row.get("source"), dict) else {}
    return {
        "media_id": target_media_id(source_id, episode_index, camera_id),
        "episode_index": episode_index,
        "camera_id": camera_id,
        "camera_name": row.get("camera_name"),
        "source": {
            "uri": source_struct.get("uri") or row.get("source_uri") or row.get("uri"),
            "repo_id": source.get("source_repo_id"),
            "dataset_url": hf_dataset_url(source.get("source_repo_id")),
            "media_id": row.get("media_id"),
            "relative_path": row.get("source_relative_path") or row.get("relative_path"),
        },
        "source_uri": row.get("source_uri") or row.get("uri"),
        "relative_path": row.get("relative_path"),
        "video_blob": row.get("video_blob"),
        "from_timestamp": row.get("from_timestamp"),
        "to_timestamp": row.get("to_timestamp"),
        "num_frames": as_int(row.get("num_frames")),
        "chunk_index": as_int(row.get("chunk_index")),
        "file_index": as_int(row.get("file_index")),
        "sha256": row.get("sha256"),
        "byte_size": as_int(row.get("byte_size")),
        "width_pixels": as_int(row.get("width_pixels")),
        "height_pixels": as_int(row.get("height_pixels")),
        "fps": as_float(row.get("fps")),
        "codec": row.get("codec"),
        "split": row.get("split") or "train",
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
        "session_id": row.get("session_id") or source.get("source_repo_id"),
        "embodiment_id": row.get("embodiment_id") or source.get("robot_type"),
        "source_media_id": row.get("media_id"),
        "source_relative_path": row.get("relative_path"),
        "source_video_table": f"{source.get('path')}/data/videos.lance",
        "source_robot_type": source.get("robot_type"),
        "source_robot_name": source.get("robot_name"),
        "pretrain_tier": source.get("pretrain_tier"),
        "quality_flag": source.get("quality_flag"),
    }


def build_episodes_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64()),
            pa.field("fps", pa.float64()),
            pa.field("length", pa.int64()),
            pa.field("timestamps", pa.list_(pa.float64())),
            pa.field("observation_state", pa.large_list(pa.list_(pa.float32(), STATE_DIM))),
            pa.field("actions", pa.large_list(pa.list_(pa.float32(), ACTION_DIM))),
            pa.field("language_instruction", pa.string()),
            segment_fields(pa)[0],
            segment_fields(pa)[1],
            pa.field("trajectory_sha256", pa.string()),
            *source_fields(pa),
        ]
    )


def segment_fields(pa: Any) -> tuple[Any, Any]:
    return (
        pa.field(
            "camera_segments",
            pa.list_(
                pa.struct(
                    [
                        pa.field("camera_key", pa.string()),
                        pa.field("camera_column", pa.string()),
                        pa.field("media_id", pa.string()),
                        pa.field("from_timestamp", pa.float64()),
                        pa.field("to_timestamp", pa.float64()),
                        pa.field("frame_start", pa.int64()),
                        pa.field("frame_count", pa.int64()),
                    ]
                )
            ),
        ),
        pa.field(
            "task_segments",
            pa.list_(
                pa.struct(
                    [
                        pa.field("task_index", pa.int64()),
                        pa.field("language_instruction", pa.string()),
                        pa.field("start_frame", pa.int64()),
                        pa.field("end_frame_exclusive", pa.int64()),
                        pa.field("start_timestamp", pa.float64()),
                        pa.field("end_timestamp_exclusive", pa.float64()),
                    ]
                )
            ),
        ),
    )


def build_train_episodes_schema(pa: Any, camera_columns: list[str]) -> Any:
    fields = [
        pa.field("episode_index", pa.int64(), nullable=False),
        pa.field("task_index", pa.int64()),
        pa.field("fps", pa.float64()),
        pa.field("length", pa.int64()),
        pa.field("timestamps", pa.list_(pa.float64())),
        pa.field("observation_state", pa.large_list(pa.list_(pa.float32(), STATE_DIM))),
        pa.field("actions", pa.large_list(pa.list_(pa.float32(), ACTION_DIM))),
        pa.field("language_instruction", pa.string()),
        segment_fields(pa)[0],
        segment_fields(pa)[1],
        pa.field("trajectory_sha256", pa.string()),
    ]
    fields.extend(source_fields(pa))
    return pa.schema(fields)


def build_frames_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("global_frame_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
            pa.field("task_index", pa.int64()),
            pa.field("observation_state", pa.list_(pa.float32(), STATE_DIM)),
            pa.field("action", pa.list_(pa.float32(), ACTION_DIM)),
            pa.field("state_norm", pa.float32()),
            pa.field("action_norm", pa.float32()),
            pa.field("is_bad_frame", pa.bool_(), nullable=False),
            *source_fields(pa, include_frame=True),
        ]
    )


def build_videos_schema(pa: Any, lance: Any) -> Any:
    video_blob_field = lance.blob_field("video_blob")
    return pa.schema(
        [
            pa.field("media_id", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("camera_id", pa.string()),
            pa.field("camera_name", pa.string()),
            pa.field(
                "source",
                pa.struct(
                    [
                        pa.field("uri", pa.string()),
                        pa.field("repo_id", pa.string()),
                        pa.field("dataset_url", pa.string()),
                        pa.field("media_id", pa.string()),
                        pa.field("relative_path", pa.string()),
                    ]
                ),
            ),
            pa.field("source_uri", pa.string()),
            pa.field("relative_path", pa.string()),
            video_blob_field,
            pa.field("from_timestamp", pa.float64()),
            pa.field("to_timestamp", pa.float64()),
            pa.field("num_frames", pa.int64()),
            pa.field("chunk_index", pa.int64()),
            pa.field("file_index", pa.int64()),
            pa.field("sha256", pa.string()),
            pa.field("byte_size", pa.int64()),
            pa.field("width_pixels", pa.int64()),
            pa.field("height_pixels", pa.int64()),
            pa.field("fps", pa.float64()),
            pa.field("codec", pa.string()),
            *source_fields(pa, include_media=True),
        ]
    )


def source_fields(pa: Any, *, include_frame: bool = False, include_media: bool = False) -> list[Any]:
    fields = [
        pa.field("split", pa.string(), nullable=False),
        pa.field("source_id", pa.int64()),
        pa.field("source_dataset", pa.string()),
        pa.field("source_repo_id", pa.string()),
        pa.field("source_dataset_url", pa.string()),
        pa.field("source_local_path", pa.string()),
        pa.field("source_episode_index", pa.int64()),
        pa.field("session_id", pa.string()),
        pa.field("embodiment_id", pa.string()),
    ]
    if include_frame:
        fields.extend(
            [
                pa.field("source_frame_index", pa.int64()),
                pa.field("source_global_frame_index", pa.int64()),
            ]
        )
    if include_media:
        fields.extend(
            [
                pa.field("source_media_id", pa.string()),
                pa.field("source_relative_path", pa.string()),
                pa.field("source_video_table", pa.string()),
            ]
        )
    fields.extend(
        [
            pa.field("source_robot_type", pa.string()),
            pa.field("source_robot_name", pa.string()),
            pa.field("pretrain_tier", pa.string()),
            pa.field("quality_flag", pa.string()),
        ]
    )
    return fields


def build_manifest(
    *,
    dataset_id: str,
    sources: list[dict[str, Any]],
    camera_keys: list[str],
    camera_columns: list[str],
    source_camera_keys: list[str],
    source_camera_columns: list[str],
    episodes: int,
    frames: int,
    videos: int,
    indexes_created: list[dict[str, Any]],
) -> dict[str, Any]:
    fps_values = sorted({float(source["fps"]) for source in sources if source.get("fps") is not None})
    primary_fps = 10.0 if 10.0 in fps_values else (fps_values[0] if fps_values else None)
    fps_for_registry = float(primary_fps or 0.0)
    return {
        "format": PUBLISHED_FORMAT,
        "schema_version": "2.0",
        "source_format": PUBLISHED_FORMAT,
        "lance": {
            "data_storage_version": LANCE_DATA_STORAGE_VERSION,
            "blob_encoding": LANCE_BLOB_ENCODING,
            "published_blob_policy": PUBLISHED_BLOB_POLICY,
            "external_blob_uris_allowed": False,
            "requires_take_blobs": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "source": "merged_converted_19d_lance_bundles",
        "primary_training_table": "data/train_episodes.lance",
        "state_action_alignment": {
            "type": "same_frame_timestamp",
            "episode_timestamp_column": "timestamps",
            "frame_timestamp_column": "timestamp",
            "state_episode_column": "observation_state",
            "action_episode_column": "actions",
            "state_frame_column": "observation_state",
            "action_frame_column": "action",
            "index_rule": (
                "observation_state[i] and actions[i] are aligned to timestamps[i]; "
                "the builder preserves source timing and does not shift actions."
            ),
        },
        "modalities": build_modalities(camera_keys, camera_columns, fps_for_registry),
        "actions": build_actions(fps_for_registry),
        "training_targets": ["action.body"],
        "rates": {
            "fps": primary_fps,
            "fps_values": fps_values,
            "fps_mode": "single" if len(fps_values) <= 1 else "mixed",
            "modalities": {
                "state.body": primary_fps,
                **{video_modality_key(key): primary_fps for key in camera_keys},
            },
            "actions": {"action.body": primary_fps},
        },
        "capabilities": {
            "inline_video_blobs": bool(videos),
            "lance_blob_v2": bool(videos),
            "videos_table": bool(videos),
            "frames_table": bool(frames),
            "modality_registry_v2": True,
            "fixed_size_state_action": True,
            "action_semantics": True,
            "camera_segments": True,
            "task_segments": True,
            "trajectory_sha256": True,
            "per_modality_stats": True,
        },
        "reader_hints": {
            "prefer_registry": True,
            "video_lookup": "videos.media_id",
            "normalization": "meta/stats",
            "lazy_blob_columns": {"data/videos.lance": ["video_blob"]} if videos else {},
            "blob_read_api": "take_blobs",
            "default_projections": {
                "frames_training": [
                    "episode_index",
                    "frame_index",
                    "global_frame_index",
                    "timestamp",
                    "observation_state",
                    "action",
                    "task_index",
                    "is_bad_frame",
                ],
                "videos_metadata": [
                    "media_id",
                    "episode_index",
                    "camera_id",
                    "camera_name",
                    "from_timestamp",
                    "to_timestamp",
                    "num_frames",
                    "sha256",
                    "byte_size",
                ],
            },
        },
        "indexes": {
            "created": indexes_created,
            "recommended": [
                {
                    "table": "data/train_episodes.lance",
                    "index_type": "BTREE",
                    "columns": ["episode_index"],
                },
                {
                    "table": "data/frames.lance",
                    "index_type": "BTREE",
                    "columns": ["global_frame_index", "episode_index", "frame_index"],
                },
                {
                    "table": "data/videos.lance",
                    "index_type": "BTREE",
                    "columns": ["media_id", "episode_index"],
                },
                {
                    "table": "data/videos.lance",
                    "index_type": "BITMAP",
                    "columns": ["camera_id"],
                },
            ],
        },
        "primary_access_patterns": {
            "episode_sequence_loading": {
                "table": "train_episodes",
                "path": "data/train_episodes.lance",
            },
            "random_frame_sampling": {
                "table": "frames",
                "path": "data/frames.lance",
                "index_column": "global_frame_index",
            },
            "video_blob_lookup": {
                "table": "videos",
                "path": "data/videos.lance",
                "lookup_key": "media_id",
                "blob_column": "video_blob",
            },
        },
        "tables": {
            "episodes": {"path": "data/episodes.lance", "exists": True},
            "train_episodes": {"path": "data/train_episodes.lance", "exists": True},
            "primary_training": {"path": "data/train_episodes.lance", "exists": True},
            "frames": {"path": "data/frames.lance", "exists": bool(frames)},
            "videos": {"path": "data/videos.lance", "exists": bool(videos)},
        },
        "meta": {
            "info": "meta/info.json",
            "stats_dir": "meta/stats",
            "state_body_stats": "meta/stats/state_body.json",
            "action_body_stats": "meta/stats/action_body.json",
            "tasks_jsonl": "meta/tasks.jsonl",
            "episodes_jsonl": "meta/episodes.jsonl",
            "splits": "meta/splits.json",
            "sessions": "meta/sessions.json",
            "sources": "meta/sources.json",
        },
        "counts": {
            "episodes": episodes,
            "frames": frames,
            "videos": videos,
            "sources": len(sources),
        },
        "provenance": {
            "sessions": "meta/sessions.json",
            "sources": "meta/sources.json",
            "source_columns": [
                "source_id",
                "source_dataset",
                "source_repo_id",
                "source_dataset_url",
                "source_episode_index",
                "source_robot_type",
                "pretrain_tier",
            ],
            "media_reference_columns": [
                "source_uri",
                "source_local_path",
                "source_video_table",
                "source_media_id",
                "source_relative_path",
            ],
        },
    }


def video_modality_key(camera_key: str) -> str:
    if camera_key.startswith("observation.images."):
        return f"video.{camera_key.removeprefix('observation.images.')}"
    return f"video.{camera_key}"


def build_modalities(camera_keys: list[str], camera_columns: list[str], fps: float) -> dict[str, Any]:
    modalities: dict[str, Any] = {
        "state.body": {
            "kind": "state",
            "source_key": "observation.state",
            "table": "train_episodes",
            "path": "data/train_episodes.lance",
            "column": "observation_state",
            "frame_table": "frames",
            "frame_path": "data/frames.lance",
            "frame_column": "observation_state",
            "names_ref": "meta/info.json#/features/observation.state/names",
            "shape": [19],
            "shape_policy": "single",
            "rate_hz": fps,
            "stats": "meta/stats/state_body.json",
        }
    }
    for camera_key, camera_column in zip(camera_keys, camera_columns):
        modalities[video_modality_key(camera_key)] = {
            "kind": "video",
            "source_key": camera_key,
            "camera_key": camera_key,
            "camera_column": camera_column,
            "table": "videos",
            "path": "data/videos.lance",
            "media_id_column": "media_id",
            "blob_column": "video_blob",
            "segment_column": "camera_segments",
            "encoding": "rgb8_h264",
            "names_ref": f"meta/info.json#/features/{camera_key}/names",
            "shape_ref": f"meta/info.json#/features/{camera_key}/shape",
            "rate_hz": fps,
        }
    return modalities


def build_actions(fps: float) -> dict[str, Any]:
    return {
        "action.body": {
            "kind": "action",
            "source_key": "action",
            "table": "train_episodes",
            "path": "data/train_episodes.lance",
            "column": "actions",
            "frame_table": "frames",
            "frame_path": "data/frames.lance",
            "frame_column": "action",
            "names_ref": "meta/info.json#/features/action/names",
            "shape": [19],
            "shape_policy": "single",
            "rate_hz": fps,
            "stats": "meta/stats/action_body.json",
            "alignment": "same_frame_timestamp",
            "semantics": {
                "command_type": "joint_position",
                "absolute_or_delta": "absolute",
                "units": "mixed",
                "control_frame": "robot_base",
                "applies_to_interval": "[t_i, t_{i+1})",
                "normalized": False,
            },
        }
    }


def build_info(dataset_id: str, manifest: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [19],
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": [19],
            "names": ["action"],
        },
    }
    camera_entries = [
        entry
        for entry in (manifest.get("modalities") or {}).values()
        if isinstance(entry, dict) and entry.get("kind") == "video"
    ]
    for entry in camera_entries:
        key = entry.get("camera_key")
        if key:
            features[str(key)] = {
                "dtype": "video",
                "shape": None,
                "names": ["height", "width", "channels"],
            }
    return {
        "codebase_version": "rllab_published_lance_dataset_v2",
        "repo_id": dataset_id,
        "fps": (manifest.get("rates") or {}).get("fps"),
        "fps_values": (manifest.get("rates") or {}).get("fps_values") or [],
        "total_episodes": manifest["counts"]["episodes"],
        "total_frames": manifest["counts"]["frames"],
        "total_videos": manifest["counts"]["videos"],
        "total_source_datasets": len(sources),
        "robot_type": "aiworker_19d_mixture",
        "features": features,
    }


def render_readme(dataset_id: str, manifest: dict[str, Any], sessions: list[dict[str, Any]]) -> str:
    lines = [
        f"# {dataset_id}",
        "",
        "Merged 19D AI Worker/BG2 pretraining dataset in RLLAB published Lance layout.",
        "",
        "## Tables",
        "",
        "| Table | Purpose |",
        "| --- | --- |",
        "| `data/episodes.lance` | Published episode table, one row per episode, no video blob columns. |",
        "| `data/train_episodes.lance` | Training trajectory table named by `manifest.json.primary_training_table`; no video blob columns. |",
        "| `data/frames.lance` | Frame-level QA/index table with remapped global frame indices. |",
        "| `data/videos.lance` | Canonical media table; `video_blob` stores one MP4 per episode/camera. |",
        "",
        "## Summary",
        "",
        f"- Source datasets: {len(sessions)}",
        f"- Episodes: {manifest['counts']['episodes']}",
        f"- Frames: {manifest['counts']['frames']}",
        f"- Videos: {manifest['counts']['videos']}",
        "- Action/state dim: 19 / 19",
        f"- FPS values: {', '.join(str(v) for v in (manifest.get('rates') or {}).get('fps_values') or [])}",
        f"- Format: {manifest['format']} / schema {manifest['schema_version']}",
        f"- Blob storage: {manifest['lance']['blob_encoding']} ({manifest['lance']['published_blob_policy']})",
        "",
        "Each Lance row",
        "carries provenance columns such as `source_dataset`, `source_repo_id`,",
        "`source_dataset_url`, `source_local_path`, `source_episode_index`,",
        "`source_robot_type`, and `pretrain_tier`. `data/videos.lance` also keeps",
        "`source_video_table`, `source_media_id`, and `source_relative_path` while",
        "storing the canonical copied MP4 blob used by viewers and training.",
        "",
        "## Sources",
        "",
        "| Source dataset | Robot type | Tier | Episodes | Frames | Videos | FPS | Quality |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sessions:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_link(row.get("source_dataset"), row.get("source_url")),
                    f"`{row.get('robot_type') or ''}`",
                    f"`{row.get('pretrain_tier') or ''}`",
                    str(row.get("episodes") or 0),
                    str(row.get("frames") or 0),
                    str(row.get("videos") or 0),
                    str(row.get("fps") or ""),
                    f"`{row.get('quality_flag') or ''}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "```bash",
            "PYTHONPATH=. ./.venv/bin/python scripts/build_pretrain_19d_lance.py --overwrite",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def hf_dataset_url(repo_id: Any) -> str:
    return f"https://huggingface.co/datasets/{repo_id}" if repo_id else ""


def md_link(label: Any, url: Any) -> str:
    label_s = md_escape(str(label or ""))
    url_s = str(url or "")
    return f"[{label_s}]({url_s})" if url_s else label_s


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def quality_flag(repo_id: Any) -> str:
    name = str(repo_id or "").lower()
    review_tokens = ("test", "upload", "your-new-dataset", "with-token", "latched", "latch")
    if any(token in name for token in review_tokens):
        return "review_name"
    return "ok"


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def vector_rows(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for item in value:
        if isinstance(item, list):
            rows.append([float(v) for v in item])
    return rows


def vector_stats(vectors: list[list[float]]) -> dict[str, list[float]]:
    if not vectors:
        return {"mean": [], "std": [], "min": [], "max": [], "count": []}
    dim = max(len(vector) for vector in vectors)
    columns = [
        [float(vector[index]) for vector in vectors if index < len(vector) and math.isfinite(float(vector[index]))]
        for index in range(dim)
    ]
    means = [mean(column) for column in columns]
    return {
        "mean": means,
        "std": [std(column, value) for column, value in zip(columns, means)],
        "min": [min(column) if column else 0.0 for column in columns],
        "max": [max(column) if column else 0.0 for column in columns],
        "count": [len(column) for column in columns],
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float], value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((item - value) ** 2 for item in values) / len(values)
    return math.sqrt(variance)


if __name__ == "__main__":
    raise SystemExit(main())
