"""Convert a LeRobot v2.1 or v3.0 dataset on disk into a Lance session bundle
consumed by downstream Lance-native viewers and trainers.

Both LeRobot layouts (v2.1 single-Parquet-per-episode, v3 sharded file
Parquets) are auto-detected and produce the same Lance bundle shape. The
source `meta/info.json` is copied to the target so downstream tools can
surface per-camera codec / pix_fmt / resolution metadata without re-reading
the original LeRobot tree.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


CONVERSION_REPORT_KEYS = (
    "source",
    "target",
    "layout_detected",
    "episodes_written",
    "frames_written",
    "media_written",
    "fps",
    "cameras",
)


def convert_lerobot_to_lance(
    source: str | Path,
    target: str | Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    include_frames: bool = True,
    include_video_blobs: bool = True,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read a LeRobot dataset rooted at ``source`` and write a Lance bundle
    under ``target``. Returns a report dict (``CONVERSION_REPORT_KEYS``)."""

    source = Path(source)
    target = Path(target)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import lance
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "LeRobot→Lance conversion requires pyarrow and lance "
            "(install the [lance] extra)."
        ) from exc

    info = _load_info(source)
    layout = _detect_layout(source)
    fps = float(info.get("fps") or 0)
    if fps <= 0:
        raise ValueError("info.json must include a positive fps")
    camera_keys = _video_features(info)
    task_lookup = _load_tasks(source)
    episode_meta_rows = _load_episode_meta_rows(source, layout)
    if limit is not None and limit > 0:
        episode_meta_rows = episode_meta_rows[:limit]

    if not episode_meta_rows:
        raise ValueError("No episode metadata rows found in source dataset")

    stale_table_names = ("episodes.lance", "frames.lance", "media.lance", "videos.lance")
    if any((target / name).exists() for name in stale_table_names):
        if not overwrite:
            raise FileExistsError(
                f"Target already contains Lance tables (use overwrite=True): {target}"
            )
        for name in stale_table_names:
            shutil.rmtree(target / name, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    chunks_size = int(info.get("chunks_size") or 1000) or 1000
    cameras_norm = [_normalize_camera_key(key) for key in camera_keys]

    frame_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    media_rows: list[dict[str, Any]] = []
    seen_video_paths: set[str] = set()

    total = len(episode_meta_rows)
    for ordinal, meta_row in enumerate(episode_meta_rows):
        try:
            episode_index = int(meta_row["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"episode metadata row missing episode_index: {meta_row}"
            ) from exc
        chunk_index = episode_index // chunks_size

        ep_frames = _read_episode_frames(source, layout, info, episode_index, meta_row)
        if not ep_frames:
            raise ValueError(f"No frame rows found for episode {episode_index}")

        timestamps: list[float] = []
        states: list[list[float]] = []
        actions: list[list[float]] = []
        first_task_index: int | None = None
        for frame in ep_frames:
            timestamps.append(float(frame.get("timestamp") or 0.0))
            states.append(_as_float_list(frame.get("observation.state")))
            actions.append(_as_float_list(frame.get("action")))
            if first_task_index is None and frame.get("task_index") is not None:
                first_task_index = int(frame["task_index"])

        episode_task_index = (
            int(meta_row.get("task_index"))
            if meta_row.get("task_index") is not None
            else first_task_index
        )
        if episode_task_index is None:
            episode_task_index = _task_index_from_meta(meta_row, task_lookup)

        language = meta_row.get("language_instruction") or _caption_from_meta_tasks(meta_row)
        if not language and episode_task_index is not None:
            language = _task_text_for_index(task_lookup, episode_task_index)

        episode_row: dict[str, Any] = {
            "episode_index": episode_index,
            "task_index": episode_task_index if episode_task_index is not None else 0,
            "fps": fps,
            "length": len(ep_frames),
            "timestamps": timestamps,
            "observation_state": states,
            "actions": actions,
            "language_instruction": language,
        }

        if include_frames:
            for frame in ep_frames:
                frame_index = int(frame.get("frame_index", frame.get("index", 0)))
                state = _as_float_list(frame.get("observation.state"))
                action = _as_float_list(frame.get("action"))
                frame_rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "global_frame_index": int(
                            frame.get(
                                "index",
                                episode_index * len(ep_frames) + frame_index,
                            )
                        ),
                        "timestamp": float(frame.get("timestamp") or 0.0),
                        "task_index": int(
                            frame.get("task_index") or episode_task_index or 0
                        ),
                        "observation_state": state,
                        "action": action,
                        "state_norm": _vector_norm(state),
                        "action_norm": _vector_norm(action),
                        "is_bad_frame": False,
                    }
                )

        for camera_key, camera_norm in zip(camera_keys, cameras_norm):
            feature_info = (info.get("features") or {}).get(camera_key) or {}
            video_path = _video_path(
                source,
                info,
                layout,
                episode_index,
                chunk_index,
                camera_key,
                meta_row,
            )
            if video_path is None or not video_path.exists():
                episode_row[f"{camera_norm}_video_blob"] = None
                episode_row[f"{camera_norm}_from_timestamp"] = None
                episode_row[f"{camera_norm}_to_timestamp"] = None
                continue
            blob = video_path.read_bytes()
            if include_video_blobs:
                episode_row[f"{camera_norm}_video_blob"] = blob
            else:
                episode_row[f"{camera_norm}_video_blob"] = None
            episode_row[f"{camera_norm}_from_timestamp"] = 0.0
            episode_row[f"{camera_norm}_to_timestamp"] = (len(ep_frames) - 1) / fps

            video_key = str(video_path.resolve())
            if video_key not in seen_video_paths:
                seen_video_paths.add(video_key)
                media_rows.append(
                    _media_row(
                        info=info,
                        camera_key=camera_key,
                        camera_norm=camera_norm,
                        video_path=video_path,
                        source=source,
                        blob=blob,
                        episode_index=episode_index,
                        chunk_index=chunk_index,
                        fps=fps,
                        num_frames=len(ep_frames),
                        feature_info=feature_info,
                    )
                )

        episode_rows.append(episode_row)
        if progress_callback:
            progress_callback(
                "episode_converted",
                {
                    "episode_index": episode_index,
                    "completed": ordinal + 1,
                    "total": total,
                },
            )

    episodes_schema = _build_episodes_schema(pa, cameras_norm)
    episodes_table = pa.Table.from_pylist(episode_rows, schema=episodes_schema)
    lance.write_dataset(episodes_table, str(target / "episodes.lance"), mode="overwrite")

    if include_frames and frame_rows:
        frames_schema = _build_frames_schema(pa)
        frames_table = pa.Table.from_pylist(frame_rows, schema=frames_schema)
        lance.write_dataset(frames_table, str(target / "frames.lance"), mode="overwrite")

    if media_rows:
        media_schema = _build_media_schema(pa)
        media_table = pa.Table.from_pylist(media_rows, schema=media_schema)
        lance.write_dataset(media_table, str(target / "media.lance"), mode="overwrite")

    target_meta = target / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)
    (target_meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _write_manifest(
        target,
        source=source,
        layout=layout,
        fps=fps,
        camera_keys=camera_keys,
        cameras_norm=cameras_norm,
        include_frames=include_frames and bool(frame_rows),
        has_media=bool(media_rows),
        state_dim=_nested_dim(states),
        action_dim=_nested_dim(actions),
        episodes_written=len(episode_rows),
        frames_written=len(frame_rows) if include_frames else 0,
        media_written=len(media_rows),
    )

    return {
        "source": str(source),
        "target": str(target),
        "layout_detected": layout,
        "episodes_written": len(episode_rows),
        "frames_written": len(frame_rows) if include_frames else 0,
        "media_written": len(media_rows),
        "fps": fps,
        "cameras": cameras_norm,
    }


# ---------------------------------------------------------------- file readers


def _load_info(source: Path) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot dataset must contain {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _detect_layout(source: Path) -> str:
    """Return ``"v3"`` if the v3 sharded episodes Parquet/JSONL exists,
    otherwise ``"v2_1"`` if a single ``meta/episodes.jsonl`` exists, else
    raise."""
    episodes_dir = source / "meta" / "episodes"
    if episodes_dir.is_dir() and (
        any(episodes_dir.glob("**/*.parquet"))
        or any(episodes_dir.glob("**/*.jsonl"))
    ):
        return "v3"
    if (source / "meta" / "episodes.jsonl").exists():
        return "v2_1"
    raise FileNotFoundError(
        f"Could not detect LeRobot layout under {source}: expected meta/episodes/ "
        "(v3) or meta/episodes.jsonl (v2.1)."
    )


def _load_tasks(source: Path) -> dict[int, str]:
    tasks_path = source / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}
    lookup: dict[int, str] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "task_index" in row and "task" in row:
            lookup[int(row["task_index"])] = str(row["task"])
    return lookup


def _video_features(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    return sorted(
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _load_episode_meta_rows(source: Path, layout: str) -> list[dict[str, Any]]:
    if layout == "v2_1":
        path = source / "meta" / "episodes.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    # v3: try Parquet under meta/episodes/, fall back to JSONL under same dir
    parquets = sorted((source / "meta" / "episodes").glob("**/*.parquet"))
    if parquets:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyarrow required to read v3 episodes parquet") from exc
        rows: list[dict[str, Any]] = []
        for path in parquets:
            rows.extend(list(pq.read_table(path).to_pylist()))
        return rows
    jsonls = sorted((source / "meta" / "episodes").glob("**/*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in jsonls:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if rows:
        return rows
    raise FileNotFoundError(
        f"No episode metadata files found under {source / 'meta' / 'episodes'}"
    )


def _read_episode_frames(
    source: Path,
    layout: str,
    info: dict[str, Any],
    episode_index: int,
    meta_row: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow required to read frame Parquet") from exc

    chunks_size = int(info.get("chunks_size") or 1000) or 1000
    chunk_index = episode_index // chunks_size

    if layout == "v2_1":
        template = info.get("data_path") or (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        rel = template.format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
            chunk_index=chunk_index,
            file_index=episode_index,
        )
        path = source / rel
        if not path.exists():
            raise FileNotFoundError(f"v2.1 episode Parquet missing: {path}")
        return list(pq.read_table(path).to_pylist())

    # v3: file Parquet may contain multiple episodes; slice by episode_index
    chunk_idx = _int_from_row(meta_row, "data/chunk_index", "chunk_index", default=chunk_index)
    file_index = _int_from_row(meta_row, "data/file_index", "file_index", default=chunk_idx)
    template = info.get("data_path") or (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    rel = template.format(chunk_index=chunk_idx, file_index=file_index)
    path = source / rel
    if not path.exists():
        raise FileNotFoundError(f"v3 file Parquet missing: {path}")
    table = pq.read_table(path)
    return [row for row in table.to_pylist() if int(row.get("episode_index", -1)) == episode_index]


def _video_path(
    source: Path,
    info: dict[str, Any],
    layout: str,
    episode_index: int,
    default_chunk_index: int,
    camera_key: str,
    meta_row: dict[str, Any],
) -> Path | None:
    template = info.get("video_path")
    if not template:
        return None
    chunk_index = default_chunk_index
    file_index = episode_index
    if layout == "v3":
        chunk_index = _int_from_row(
            meta_row,
            f"videos/{camera_key}/chunk_index",
            default=default_chunk_index,
        )
        file_index = _int_from_row(
            meta_row,
            f"videos/{camera_key}/file_index",
            default=episode_index,
        )
    rel = template.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        chunk_index=chunk_index,
        file_index=file_index,
        video_key=camera_key,
    )
    return source / rel


# ---------------------------------------------------------------- normalizers


def _int_from_row(row: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return int(value)
    return int(default)


def _normalize_camera_key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", value)


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _vector_norm(value: list[float]) -> float:
    return float(sum(item * item for item in value) ** 0.5)


def _nested_dim(value: list[list[float]]) -> int:
    for row in value:
        if row:
            return len(row)
    return 0


def _video_shape(feature_info: dict[str, Any]) -> tuple[int | None, int | None]:
    shape = feature_info.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return None, None


def _video_codec(feature_info: dict[str, Any]) -> str | None:
    nested = feature_info.get("info")
    if isinstance(nested, dict):
        codec = nested.get("video.codec") or nested.get("codec")
        if codec:
            return str(codec)
    codec = feature_info.get("video.codec") or feature_info.get("codec")
    return str(codec) if codec else None


def _media_row(
    *,
    info: dict[str, Any],
    camera_key: str,
    camera_norm: str,
    video_path: Path,
    source: Path,
    blob: bytes,
    episode_index: int,
    chunk_index: int,
    fps: float,
    num_frames: int,
    feature_info: dict[str, Any],
) -> dict[str, Any]:
    height, width = _video_shape(feature_info)
    rel = str(video_path.relative_to(source))
    return {
        "media_id": f"episode_{episode_index:06d}_{camera_norm}",
        "episode_id": f"episode_{episode_index:06d}",
        "episode_index": episode_index,
        "camera_id": camera_norm,
        "camera_name": camera_key,
        "media_type": "video",
        "uri": rel,
        "relative_path": rel,
        "video_path": None,
        "from_timestamp": 0.0,
        "to_timestamp": (num_frames - 1) / fps if num_frames > 0 else None,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "byte_size": len(blob),
        "num_frames": int(num_frames),
        "fps": float(fps),
        "width_pixels": width,
        "height_pixels": height,
        "codec": _video_codec(feature_info),
        "chunk_index": int(chunk_index),
        "file_index": int(episode_index),
        "video_blob": blob,
    }


def _write_manifest(
    target: Path,
    *,
    source: Path,
    layout: str,
    fps: float,
    camera_keys: list[str],
    cameras_norm: list[str],
    include_frames: bool,
    has_media: bool,
    state_dim: int,
    action_dim: int,
    episodes_written: int,
    frames_written: int,
    media_written: int,
) -> None:
    tables = {
        "episodes": "episodes.lance",
        "primary_training": "episodes.lance",
    }
    if include_frames:
        tables["frames"] = "frames.lance"
    if has_media:
        tables["media"] = "media.lance"

    manifest = {
        "format": "rllab_lance_session_v1",
        "source_format": f"lerobot_{layout}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "primary_training_table": "episodes.lance",
        "training_row_unit": "episode",
        "training_index_column": "episode_index",
        "source_episode_column": None,
        "video_frame_offset_column": None,
        "frame_table": "frames.lance" if include_frames else None,
        "media_table": "media.lance" if has_media else None,
        "state_column": "observation_state",
        "action_column": "actions",
        "training_columns": {
            "state": "observation_state",
            "action": "actions",
        },
        "camera_keys": camera_keys,
        "camera_columns": cameras_norm,
        "fps": float(fps),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "media_mode": "episode_blob",
        "blob_storage": {
            "episodes": "video_blob_columns",
            "media": "video_blob_column" if has_media else "absent",
        },
        "tables": tables,
        "counts": {
            "episodes": int(episodes_written),
            "frames": int(frames_written),
            "media": int(media_written),
        },
    }
    text = json.dumps(manifest, indent=2, sort_keys=True)
    (target / "manifest.json").write_text(text, encoding="utf-8")


def _caption_from_meta_tasks(row: dict[str, Any]) -> str | None:
    tasks = row.get("tasks")
    if isinstance(tasks, (list, tuple)) and tasks and isinstance(tasks[0], str):
        return tasks[0]
    return None


def _task_index_from_meta(row: dict[str, Any], lookup: dict[int, str]) -> int | None:
    if not lookup:
        return None
    caption = _caption_from_meta_tasks(row)
    if caption is None:
        return None
    for index, text in lookup.items():
        if text == caption:
            return int(index)
    return None


def _task_text_for_index(lookup: dict[int, str], task_index: int) -> str | None:
    return lookup.get(int(task_index))


# ---------------------------------------------------------------- schemas


def _build_episodes_schema(pa: Any, cameras_norm: list[str]) -> Any:
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
    for camera_norm in cameras_norm:
        fields.append(
            pa.field(
                f"{camera_norm}_video_blob",
                pa.large_binary(),
                metadata={b"lance-encoding:blob": b"true"},
            )
        )
        fields.append(pa.field(f"{camera_norm}_from_timestamp", pa.float64()))
        fields.append(pa.field(f"{camera_norm}_to_timestamp", pa.float64()))
    return pa.schema(fields)


def _build_frames_schema(pa: Any) -> Any:
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
        ]
    )


def _build_media_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("media_id", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("camera_id", pa.string()),
            pa.field("camera_name", pa.string(), nullable=False),
            pa.field("media_type", pa.string()),
            pa.field("uri", pa.string()),
            pa.field("relative_path", pa.string()),
            pa.field(
                "video_blob",
                pa.large_binary(),
                metadata={b"lance-encoding:blob": b"true"},
            ),
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
        ]
    )
