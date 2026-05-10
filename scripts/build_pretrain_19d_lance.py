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

By default this script does not copy Lance video blobs. Lance blob columns scan
back as descriptor handles (for example {"position": 0, "size": 1234}), not raw
bytes, so a training/pretrain merge should keep numeric/text trajectory data and
point back to the source bundles for media. Use `--copy-video-blobs` only when
you explicitly want a viewer bundle that re-materializes MP4 bytes with
`Dataset.take_blobs()`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
from typing import Any


PUBLISHED_FORMAT = "rllab_published_lance_dataset_v1"
PUBLISHED_LAYOUT = "rllab_published_dataset_v1"
DEFAULT_DATASET_ID = "rllab-postech/pretraining-aiworker-bg2-19d"


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
    parser.add_argument(
        "--copy-video-blobs",
        action="store_true",
        help=(
            "Re-materialize episode/video blobs with Lance take_blobs(). "
            "Default is source-reference media with no copied MP4 bytes."
        ),
    )
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Rows per write batch. Keep modest when --copy-video-blobs is enabled.",
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
    camera_keys = source_camera_keys if args.copy_video_blobs else []
    camera_columns = source_camera_columns if args.copy_video_blobs else []
    episodes_schema = build_episodes_schema(pa)
    train_episodes_schema = build_train_episodes_schema(
        pa,
        source_camera_columns,
        include_video_blobs=args.copy_video_blobs,
    )
    frames_schema = build_frames_schema(pa)
    videos_schema = build_videos_schema(pa, include_video_blobs=args.copy_video_blobs)

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
                include_video_blobs=args.copy_video_blobs,
            ),
            copy_blobs=args.copy_video_blobs,
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
                    include_video_blobs=args.copy_video_blobs,
                ),
                copy_blobs=args.copy_video_blobs,
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
        copy_video_blobs=args.copy_video_blobs,
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "meta" / "manifest.json", manifest)
    write_json(output / "meta" / "sessions.json", sessions)
    write_json(output / "meta" / "sources.json", source_rows)
    write_json(output / "meta" / "info.json", build_info(args.dataset_id, manifest, source_rows))
    (output / "README.md").write_text(render_readme(args.dataset_id, manifest, sessions), encoding="utf-8")

    print(f"wrote {output}", flush=True)
    print(
        json.dumps(
            {
                "datasets": len(sources),
                "episodes": episode_offset,
                "frames": frame_offset,
                "videos": video_count,
                "copy_video_blobs": bool(args.copy_video_blobs),
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
        if int(manifest.get("state_dim") or 0) != 19:
            continue
        if int(manifest.get("action_dim") or 0) != 19:
            continue
        pretrain_tier = str(manifest.get("pretrain_tier") or "")
        is_bg2 = (
            manifest.get("source_robot_type") == "ffw_bg2_rev4"
            or pretrain_tier.startswith("A_bg2_full")
        )
        if strict_bg2_only and not is_bg2:
            continue
        repo_id = manifest.get("source_repo_id") or manifest.get("source_dataset") or manifest.get("dataset_id")
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
                "robot_type": manifest.get("source_robot_type"),
                "robot_name": manifest.get("source_robot_name"),
                "pretrain_tier": pretrain_tier,
                "quality_flag": quality,
                "fps": manifest.get("fps"),
                "episodes": int(manifest.get("total_episodes") or manifest.get("counts", {}).get("episodes") or 0),
                "frames": int(manifest.get("total_frames") or manifest.get("counts", {}).get("frames") or 0),
                "videos": int(
                    manifest.get("total_video_segments")
                    or manifest.get("total_videos")
                    or manifest.get("counts", {}).get("media")
                    or 0
                ),
                "state_dim": int(manifest.get("state_dim") or 0),
                "action_dim": int(manifest.get("action_dim") or 0),
                "camera_keys": manifest.get("camera_keys") or [],
                "camera_columns": manifest.get("camera_columns") or [],
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
    blob_columns = {
        field.name
        for field in target_schema
        if field.metadata and field.metadata.get(b"lance-encoding:blob") == b"true"
    }
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
        table = pa.Table.from_pylist(rows, schema=target_schema)
        lance.write_dataset(table, str(target_path), mode=mode)
        mode = "append"
    return total_rows


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
            row[column] = handle.readall()
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
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    return {
        "episode_index": episode_map[source_episode],
        "task_index": as_int(row.get("task_index")) or 0,
        "fps": as_float(row.get("fps")),
        "length": as_int(row.get("length")) or 0,
        "timestamps": row.get("timestamps") or [],
        "observation_state": row.get("observation_state") or [],
        "actions": row.get("actions") or [],
        "language_instruction": row.get("language_instruction"),
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
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
    include_video_blobs: bool,
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    out = {
        "episode_index": episode_map[source_episode],
        "task_index": as_int(row.get("task_index")) or 0,
        "fps": as_float(row.get("fps")),
        "length": as_int(row.get("length")) or 0,
        "timestamps": row.get("timestamps") or [],
        "observation_state": row.get("observation_state") or [],
        "actions": row.get("actions") or [],
        "language_instruction": row.get("language_instruction"),
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
        "source_robot_type": source.get("robot_type"),
        "source_robot_name": source.get("robot_name"),
        "pretrain_tier": source.get("pretrain_tier"),
        "quality_flag": source.get("quality_flag"),
    }
    for camera_column in camera_columns:
        if include_video_blobs:
            out[f"{camera_column}_video_blob"] = row.get(f"{camera_column}_video_blob")
        out[f"{camera_column}_from_timestamp"] = row.get(f"{camera_column}_from_timestamp")
        out[f"{camera_column}_to_timestamp"] = row.get(f"{camera_column}_to_timestamp")
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
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
        "source_frame_index": as_int(row.get("frame_index")) or 0,
        "source_global_frame_index": as_int(row.get("global_frame_index")),
        "source_robot_type": source.get("robot_type"),
        "source_robot_name": source.get("robot_name"),
        "pretrain_tier": source.get("pretrain_tier"),
        "quality_flag": source.get("quality_flag"),
    }


def transform_video(
    row: dict[str, Any],
    source: dict[str, Any],
    source_id: int,
    episode_map: dict[int, int],
    include_video_blobs: bool,
) -> dict[str, Any]:
    source_episode = int(row["episode_index"])
    episode_index = episode_map[source_episode]
    camera_id = row.get("camera_id") or ""
    return {
        "media_id": f"source_{source_id:05d}_episode_{episode_index:08d}_{camera_id}",
        "episode_id": f"episode_{episode_index:08d}",
        "episode_index": episode_index,
        "camera_id": camera_id,
        "camera_name": row.get("camera_name"),
        "media_type": row.get("media_type"),
        "uri": row.get("uri"),
        "relative_path": row.get("relative_path"),
        "video_blob": row.get("video_blob") if include_video_blobs else None,
        "video_path": row.get("video_path"),
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
        "source_id": source_id,
        "source_dataset": source.get("source_repo_id"),
        "source_repo_id": source.get("source_repo_id"),
        "source_dataset_url": hf_dataset_url(source.get("source_repo_id")),
        "source_local_path": source.get("path"),
        "source_episode_index": source_episode,
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
            pa.field("observation_state", pa.list_(pa.list_(pa.float32()))),
            pa.field("actions", pa.list_(pa.list_(pa.float32()))),
            pa.field("language_instruction", pa.string()),
            *source_fields(pa),
        ]
    )


def build_train_episodes_schema(pa: Any, camera_columns: list[str], *, include_video_blobs: bool) -> Any:
    fields = [
        pa.field("episode_index", pa.int64(), nullable=False),
        pa.field("task_index", pa.int64()),
        pa.field("fps", pa.float64()),
        pa.field("length", pa.int64()),
        pa.field("timestamps", pa.list_(pa.float64())),
        pa.field("observation_state", pa.list_(pa.list_(pa.float32()))),
        pa.field("actions", pa.list_(pa.list_(pa.float32()))),
        pa.field("language_instruction", pa.string()),
    ]
    for camera_column in camera_columns:
        if include_video_blobs:
            fields.append(
                pa.field(
                    f"{camera_column}_video_blob",
                    pa.large_binary(),
                    metadata={b"lance-encoding:blob": b"true"},
                )
            )
        fields.extend(
            [
                pa.field(f"{camera_column}_from_timestamp", pa.float64()),
                pa.field(f"{camera_column}_to_timestamp", pa.float64()),
            ]
        )
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
            pa.field("observation_state", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
            pa.field("state_norm", pa.float32()),
            pa.field("action_norm", pa.float32()),
            pa.field("is_bad_frame", pa.bool_(), nullable=False),
            *source_fields(pa, include_frame=True),
        ]
    )


def build_videos_schema(pa: Any, *, include_video_blobs: bool) -> Any:
    video_blob_field = (
        pa.field("video_blob", pa.large_binary(), metadata={b"lance-encoding:blob": b"true"})
        if include_video_blobs
        else pa.field("video_blob", pa.large_binary())
    )
    return pa.schema(
        [
            pa.field("media_id", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("camera_id", pa.string()),
            pa.field("camera_name", pa.string()),
            pa.field("media_type", pa.string()),
            pa.field("uri", pa.string()),
            pa.field("relative_path", pa.string()),
            video_blob_field,
            pa.field("video_path", pa.string()),
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
        pa.field("source_id", pa.int64()),
        pa.field("source_dataset", pa.string()),
        pa.field("source_repo_id", pa.string()),
        pa.field("source_dataset_url", pa.string()),
        pa.field("source_local_path", pa.string()),
        pa.field("source_episode_index", pa.int64()),
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
    copy_video_blobs: bool,
) -> dict[str, Any]:
    fps_values = sorted({float(source["fps"]) for source in sources if source.get("fps") is not None})
    primary_fps = 10.0 if 10.0 in fps_values else (fps_values[0] if fps_values else None)
    return {
        "format": PUBLISHED_FORMAT,
        "schema_version": "1.0",
        "published_layout": PUBLISHED_LAYOUT,
        "published_data_dir": "data",
        "source_format": PUBLISHED_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "source": "merged_converted_19d_lance_bundles",
        "source_dataset": [source.get("source_repo_id") for source in sources],
        "source_session_count": len(sources),
        "primary_training_table": "data/train_episodes.lance",
        "training_row_unit": "episode",
        "training_index_column": "episode_index",
        "source_episode_column": "source_episode_index",
        "video_frame_offset_column": None,
        "frame_table": "data/frames.lance" if frames else None,
        "media_table": "data/videos.lance" if videos else None,
        "state_column": "observation_state",
        "action_column": "actions",
        "training_columns": {
            "state": "observation_state",
            "action": "actions",
        },
        "camera_keys": camera_keys,
        "camera_columns": camera_columns,
        "fps": primary_fps,
        "fps_values": fps_values,
        "fps_mode": "single" if len(fps_values) <= 1 else "mixed",
        "state_dim": 19,
        "action_dim": 19,
        "source_camera_keys": source_camera_keys,
        "source_camera_columns": source_camera_columns,
        "media_mode": "videos_table" if copy_video_blobs else "source_reference",
        "training_ready": bool(copy_video_blobs),
        "training_ready_notes": (
            "Current rllab-training can read this bundle directly."
            if copy_video_blobs
            else "Image training with current rllab-training requires --copy-video-blobs or a source-reference media loader."
        ),
        "blob_storage": {
            "episodes": "metadata_only",
            "train_episodes": "video_blob_columns" if copy_video_blobs else "metadata_only_source_reference",
            "videos": "video_blob_column" if copy_video_blobs and videos else "null_source_reference",
            "source_reference": None if copy_video_blobs else "meta/sources.json + data/videos.lance source_* columns",
        },
        "tables": {
            "episodes": {"path": "data/episodes.lance", "exists": True},
            "train_episodes": {"path": "data/train_episodes.lance", "exists": True},
            "primary_training": {"path": "data/train_episodes.lance", "exists": True},
            "frames": {"path": "data/frames.lance", "exists": bool(frames)},
            "videos": {"path": "data/videos.lance", "exists": bool(videos)},
        },
        "counts": {
            "episodes": episodes,
            "frames": frames,
            "media": videos,
        },
        "total_episodes": episodes,
        "total_frames": frames,
        "total_videos": videos,
        "total_video_segments": videos,
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
                "source_local_path",
                "source_video_table",
                "source_media_id",
                "source_relative_path",
            ],
        },
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
    for key in manifest["camera_keys"]:
        features[key] = {
            "dtype": "video",
            "shape": None,
            "names": ["height", "width", "channels"],
        }
    return {
        "codebase_version": "rllab_published_lance_dataset_v1",
        "repo_id": dataset_id,
        "fps": manifest.get("fps"),
        "fps_values": manifest.get("fps_values") or [],
        "total_episodes": manifest["total_episodes"],
        "total_frames": manifest["total_frames"],
        "total_videos": manifest["total_videos"],
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
        "| `data/train_episodes.lance` | Training table named by `manifest.json.primary_training_table`; video blobs are present only with `--copy-video-blobs`. |",
        "| `data/frames.lance` | Frame-level QA/index table with remapped global frame indices. |",
        "| `data/videos.lance` | Source media index. `video_blob` is null unless built with `--copy-video-blobs`. |",
        "",
        "## Summary",
        "",
        f"- Source datasets: {manifest['source_session_count']}",
        f"- Episodes: {manifest['total_episodes']}",
        f"- Frames: {manifest['total_frames']}",
        f"- Videos: {manifest['total_videos']}",
        f"- Action/state dim: {manifest['action_dim']} / {manifest['state_dim']}",
        f"- FPS values: {', '.join(str(v) for v in manifest.get('fps_values') or [])}",
        f"- Media mode: {manifest['media_mode']}",
        f"- Training ready for current rllab-training: {manifest['training_ready']}",
        f"- Episode blob storage: {manifest['blob_storage']['episodes']}",
        f"- Train episode blob storage: {manifest['blob_storage']['train_episodes']}",
        "",
        "By default the merged bundle does not duplicate MP4 bytes. Each Lance row",
        "carries provenance columns such as `source_dataset`, `source_repo_id`,",
        "`source_dataset_url`, `source_local_path`, `source_episode_index`,",
        "`source_robot_type`, and `pretrain_tier`. `data/videos.lance` also keeps",
        "`source_video_table`, `source_media_id`, and `source_relative_path` so a",
        "viewer can resolve media from the original converted bundles.",
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


if __name__ == "__main__":
    raise SystemExit(main())
