"""Convert a LeRobot v2.1 or v3.0 dataset on disk into an RLLAB Lance bundle
consumed by downstream Lance-native viewers and trainers.

Both LeRobot layouts (v2.1 single-Parquet-per-episode, v3 sharded file
Parquets) are auto-detected and produce the same Lance bundle shape. The
source `meta/info.json` is copied to the target so downstream tools can
surface per-camera codec / pix_fmt / resolution metadata without re-reading
the original LeRobot tree.

The canonical media contract is intentionally singular: episode rows contain
numeric/text trajectory data and camera timestamp ranges, while playable MP4
bytes live in the media table (`media.lance` for flat local layout,
`data/videos.lance` for HF/published layout). Episode-level `*_video_blob`
columns are not written by this converter.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any


RLLAB_SESSION_FORMAT = "rllab_lance_session_v2"
RLLAB_PUBLISHED_FORMAT = "rllab_published_lance_dataset_v2"
RLLAB_SCHEMA_VERSION = "2.0"
RLLAB_PUBLISHED_LAYOUT = "rllab_published_dataset_v2"
LANCE_DATA_STORAGE_VERSION = "2.2"
LANCE_BLOB_ENCODING = "lance.blob.v2"
PUBLISHED_BLOB_POLICY = "inline_bytes_only"

# Keep a single videos.lance fragment under HF LFS object limits so very large
# bundles stay pushable. 2 GB is well under HF's per-object cap and gives good
# parallel-download granularity.
VIDEOS_MAX_BYTES_PER_FILE = 2 * 1024 * 1024 * 1024
VIDEOS_MAX_ROWS_PER_FILE = 4096

# episodes.lance / frames.lance are numeric-only; cap rows per file so very long
# pretrain merges still shard cleanly.
TRAJ_MAX_ROWS_PER_FILE = 100_000

# Scalar indexes typed by column cardinality. BTREE for high-cardinality
# join/lookup keys, BITMAP for low-cardinality categorical filters. The earlier
# "BTREE on everything" plan was over-indexed for low-cardinality columns and
# missed `frames.global_frame_index` entirely.
SCALAR_INDEXES: dict[str, list[tuple[str, str]]] = {
    "episodes": [
        ("episode_index",  "BTREE"),
        ("task_index",     "BITMAP"),
        ("split",          "BITMAP"),
        ("source_dataset", "BITMAP"),
        ("session_id",     "BITMAP"),
        ("embodiment_id",  "BITMAP"),
    ],
    "frames": [
        ("global_frame_index", "BTREE"),
        ("episode_index",      "BTREE"),
        ("frame_index",        "BTREE"),
        ("task_index",         "BITMAP"),
        ("is_bad_frame",       "BITMAP"),
        ("split",              "BITMAP"),
        ("source_dataset",     "BITMAP"),
        ("session_id",         "BITMAP"),
        ("embodiment_id",      "BITMAP"),
    ],
    "videos": [
        ("media_id",       "BTREE"),
        ("episode_index",  "BTREE"),
        ("camera_id",      "BITMAP"),
        ("source_dataset", "BITMAP"),
        ("session_id",     "BITMAP"),
        ("embodiment_id",  "BITMAP"),
    ],
}

# Backwards-compat shim for callers that still expect the column list shape.
SCALAR_INDEX_COLUMNS: dict[str, list[str]] = {
    table: [col for col, _type in entries] for table, entries in SCALAR_INDEXES.items()
}

CONVERSION_REPORT_KEYS = (
    "source",
    "target",
    "output_layout",
    "dataset_id",
    "layout_detected",
    "episodes_written",
    "frames_written",
    "media_written",
    "fps",
    "cameras",
)

FFW_BG2_REV4_JOINT_ORDER = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "gripper_l_joint1",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
]

def _ffw_joint_spec(
    component: str,
    side: str,
    role: str,
    urdf_joint_type: str,
    axis: list[float],
    lower: float,
    upper: float,
    *,
    velocity: float = 4.8,
    effort: float = 1000.0,
    mimic_driver: bool = False,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "component": component,
        "side": side,
        "role": role,
        "urdf_joint_type": urdf_joint_type,
        "axis": axis,
        "lower": lower,
        "upper": upper,
        "velocity": velocity,
        "effort": effort,
    }
    if mimic_driver:
        spec["mimic_driver"] = True
    return spec


# Derived from:
# - rllab-data-collection/config/bg2_topics.yaml target_order
# - ai_worker/ffw_description/urdf/ffw_bg2_rev4_follower/ffw_bg2_follower.urdf
# The converter stores these as manifest metadata so downstream training code can
# distinguish arms, grippers, head, and lift without hard-coding column indices.
FFW_BG2_REV4_JOINT_SPECS: dict[str, dict[str, Any]] = {
    "arm_l_joint1": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [0.0, 1.0, 0.0], -3.14, 3.14),
    "arm_l_joint2": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [1.0, 0.0, 0.0], 0.0, 3.14),
    "arm_l_joint3": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [0.0, 0.0, 1.0], -3.14, 3.14),
    "arm_l_joint4": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [0.0, 1.0, 0.0], -2.9361, 1.0786),
    "arm_l_joint5": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [0.0, 0.0, 1.0], -3.14, 3.14),
    "arm_l_joint6": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [0.0, 1.0, 0.0], -1.57, 1.57),
    "arm_l_joint7": _ffw_joint_spec("left_arm", "left", "arm", "revolute", [1.0, 0.0, 0.0], -1.8201, 1.5804),
    "gripper_l_joint1": _ffw_joint_spec(
        "left_gripper",
        "left",
        "gripper",
        "revolute",
        [1.0, 0.0, 0.0],
        0.0,
        1.1,
        velocity=6.5,
        mimic_driver=True,
    ),
    "arm_r_joint1": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [0.0, 1.0, 0.0], -3.14, 3.14),
    "arm_r_joint2": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [1.0, 0.0, 0.0], -3.14, 0.0),
    "arm_r_joint3": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [0.0, 0.0, 1.0], -3.14, 3.14),
    "arm_r_joint4": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [0.0, 1.0, 0.0], -2.9361, 1.0786),
    "arm_r_joint5": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [0.0, 0.0, 1.0], -3.14, 3.14),
    "arm_r_joint6": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [0.0, 1.0, 0.0], -1.57, 1.57),
    "arm_r_joint7": _ffw_joint_spec("right_arm", "right", "arm", "revolute", [1.0, 0.0, 0.0], -1.5804, 1.8201),
    "gripper_r_joint1": _ffw_joint_spec(
        "right_gripper",
        "right",
        "gripper",
        "revolute",
        [1.0, 0.0, 0.0],
        0.0,
        1.1,
        velocity=6.5,
        mimic_driver=True,
    ),
    "head_joint1": _ffw_joint_spec("head", "center", "head", "revolute", [0.0, 1.0, 0.0], -0.2317, 0.6951),
    "head_joint2": _ffw_joint_spec("head", "center", "head", "revolute", [0.0, 0.0, 1.0], -0.35, 0.35),
    "lift_joint": _ffw_joint_spec("lift", "center", "lift", "prismatic", [0.0, 0.0, 1.0], -0.5, 0.0),
}


def convert_lerobot_to_lance(
    source: str | Path,
    target: str | Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    include_frames: bool = True,
    output_layout: str = "hf",
    dataset_id: str | None = None,
    session_id: str | None = None,
    embodiment_id: str | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read a LeRobot dataset rooted at ``source`` and write a Lance bundle
    under ``target``. Returns a report dict (``CONVERSION_REPORT_KEYS``)."""

    source = Path(source)
    target = Path(target)
    output_layout = output_layout.lower().replace("-", "_")
    if output_layout not in {"session", "hf"}:
        raise ValueError("output_layout must be 'session' or 'hf'")
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

    table_root = target / "data" if output_layout == "hf" else target
    video_table_name = "videos.lance" if output_layout == "hf" else "media.lance"
    stale_table_names = ("episodes.lance", "frames.lance", "media.lance", "videos.lance")
    stale_roots = [table_root]
    alternate_root = target if output_layout == "hf" else target / "data"
    if alternate_root != table_root:
        stale_roots.append(alternate_root)
    if any((root / name).exists() for root in stale_roots for name in stale_table_names):
        if not overwrite:
            raise FileExistsError(
                f"Target already contains Lance tables (use overwrite=True): {target}"
            )
        for root in stale_roots:
            for name in stale_table_names:
                shutil.rmtree(root / name, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    chunks_size = int(info.get("chunks_size") or 1000) or 1000
    cameras_norm = [_normalize_camera_key(key) for key in camera_keys]

    frame_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    media_rows: list[dict[str, Any]] = []
    source_repo_id = _source_repo_id(info, source)
    split_by_episode: dict[int, str] = {}
    resolved_session_id = session_id or source_repo_id or dataset_id
    next_global_frame_index = 0

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

        task_segments = _task_segments_from_frames(
            ep_frames,
            timestamps=timestamps,
            default_task_index=episode_task_index if episode_task_index is not None else 0,
            default_language=language,
            task_lookup=task_lookup,
        )
        episode_split = str(meta_row.get("split") or "train")
        split_by_episode[episode_index] = episode_split
        episode_row: dict[str, Any] = {
            "episode_index": episode_index,
            "task_index": episode_task_index if episode_task_index is not None else 0,
            "fps": fps,
            "length": len(ep_frames),
            "timestamps": timestamps,
            "observation_state": states,
            "actions": actions,
            "language_instruction": language,
            "camera_segments": [],
            "task_segments": task_segments,
            "trajectory_sha256": _trajectory_sha256(timestamps, states, actions),
            "split": episode_split,
            "source_dataset": source_repo_id,
            "source_episode_index": episode_index,
            "session_id": resolved_session_id,
            "embodiment_id": embodiment_id,
        }

        if include_frames:
            for frame_index, frame in enumerate(ep_frames):
                state = _as_float_list(frame.get("observation.state"))
                action = _as_float_list(frame.get("action"))
                frame_rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "global_frame_index": next_global_frame_index,
                        "timestamp": float(frame.get("timestamp") or 0.0),
                        "task_index": int(
                            frame.get("task_index") or episode_task_index or 0
                        ),
                        "observation_state": state,
                        "action": action,
                        "state_norm": _vector_norm(state),
                        "action_norm": _vector_norm(action),
                        "is_bad_frame": False,
                        "split": episode_split,
                        "source_dataset": source_repo_id,
                        "session_id": resolved_session_id,
                        "embodiment_id": embodiment_id,
                    }
                )
                next_global_frame_index += 1

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
                continue
            blob = video_path.read_bytes()
            media_id = _media_id(episode_index, camera_norm)
            episode_row["camera_segments"].append(
                _camera_segment(
                    camera_key=camera_key,
                    camera_norm=camera_norm,
                    media_id=media_id,
                    num_frames=len(ep_frames),
                    fps=fps,
                )
            )

            media_rows.append(
                _media_row(
                    info=info,
                    camera_key=camera_key,
                    camera_norm=camera_norm,
                    video_path=video_path,
                    source=source,
                    source_repo_id=source_repo_id,
                    blob=blob,
                    episode_index=episode_index,
                    chunk_index=chunk_index,
                    fps=fps,
                    num_frames=len(ep_frames),
                    session_id=resolved_session_id,
                    embodiment_id=embodiment_id,
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

    indexes_built: dict[str, list[dict[str, str]]] = {}

    state_dim = _episode_vector_dim(episode_rows, "observation_state")
    action_dim = _episode_vector_dim(episode_rows, "actions")
    info = _with_inferred_ffw_feature_names(
        info,
        source_repo_id=source_repo_id,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    episodes_schema = _build_episodes_schema(
        pa,
        cameras_norm,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    episodes_table = pa.Table.from_pylist(episode_rows, schema=episodes_schema)
    indexes_built["episodes"] = _write_lance_table(
        lance,
        episodes_table,
        table_root / "episodes.lance",
        max_rows_per_file=TRAJ_MAX_ROWS_PER_FILE,
        scalar_indexes=SCALAR_INDEXES["episodes"],
    )

    if include_frames and frame_rows:
        frames_schema = _build_frames_schema(
            pa,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        frames_table = pa.Table.from_pylist(frame_rows, schema=frames_schema)
        indexes_built["frames"] = _write_lance_table(
            lance,
            frames_table,
            table_root / "frames.lance",
            max_rows_per_file=TRAJ_MAX_ROWS_PER_FILE,
            scalar_indexes=SCALAR_INDEXES["frames"],
        )

    if media_rows or output_layout == "hf":
        media_schema = _build_media_schema(pa, lance)
        media_table = _table_from_pylist_with_blob_columns(
            pa,
            lance,
            media_rows,
            schema=media_schema,
            blob_columns={"video_blob"},
        )
        indexes_built["videos"] = _write_lance_table(
            lance,
            media_table,
            table_root / video_table_name,
            max_bytes_per_file=VIDEOS_MAX_BYTES_PER_FILE,
            max_rows_per_file=VIDEOS_MAX_ROWS_PER_FILE,
            scalar_indexes=SCALAR_INDEXES["videos"],
        )

    target_meta = target / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)
    (target_meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _write_stats_json(target_meta, episode_rows)
    _write_tasks_json(target_meta, task_lookup, episode_rows)
    _write_episodes_jsonl(target_meta, episode_rows, split_by_episode)
    _write_splits_json(target_meta, episode_rows, split_by_episode)
    _write_manifest(
        target,
        source=source,
        layout=layout,
        fps=fps,
        camera_keys=camera_keys,
        cameras_norm=cameras_norm,
        include_frames=include_frames and bool(frame_rows),
        has_media=bool(media_rows),
        state_dim=state_dim,
        action_dim=action_dim,
        episodes_written=len(episode_rows),
        frames_written=len(frame_rows) if include_frames else 0,
        media_written=len(media_rows),
        output_layout=output_layout,
        dataset_id=dataset_id,
        source_repo_id=source_repo_id,
        info=info,
        indexes_built=indexes_built,
    )
    if output_layout == "hf":
        _write_hf_sessions_json(
            target,
            source=source,
            source_repo_id=source_repo_id,
            layout=layout,
            episodes_written=len(episode_rows),
            frames_written=len(frame_rows) if include_frames else 0,
            media_written=len(media_rows),
        )
        _write_hf_readme(
            target,
            dataset_id=dataset_id or target.name,
            source_repo_id=source_repo_id,
            episodes_written=len(episode_rows),
            frames_written=len(frame_rows) if include_frames else 0,
            media_written=len(media_rows),
            fps=fps,
            camera_keys=camera_keys,
        )

    return {
        "source": str(source),
        "target": str(target),
        "output_layout": output_layout,
        "dataset_id": dataset_id,
        "layout_detected": layout,
        "episodes_written": len(episode_rows),
        "frames_written": len(frame_rows) if include_frames else 0,
        "media_written": len(media_rows),
        "fps": fps,
        "cameras": cameras_norm,
    }


# ---------------------------------------------------------------- file readers


def _write_lance_table(
    lance: Any,
    table: Any,
    path: Path,
    *,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    scalar_indexes: list[tuple[str, str]] | list[str] | None = None,
) -> list[dict[str, str]]:
    """Write a Lance table with fragment caps and create typed scalar indexes.

    `scalar_indexes` is a list of `(column, index_type)` tuples. A bare list of
    column strings is also accepted for backwards compat and treated as BTREE.

    Returns a list of `{column, index_type}` records describing the indexes
    that were actually created. Columns missing from the table or index
    creation failures are skipped silently.
    """
    write_kwargs: dict[str, Any] = {"mode": "overwrite"}
    if max_rows_per_file is not None:
        write_kwargs["max_rows_per_file"] = int(max_rows_per_file)
    if max_bytes_per_file is not None:
        write_kwargs["max_bytes_per_file"] = int(max_bytes_per_file)
    write_kwargs["data_storage_version"] = LANCE_DATA_STORAGE_VERSION
    lance.write_dataset(table, str(path), **write_kwargs)
    dataset = lance.dataset(str(path))
    if str(getattr(dataset, "data_storage_version", "")) != LANCE_DATA_STORAGE_VERSION:
        raise RuntimeError(
            f"{path} was not written with Lance data_storage_version "
            f"{LANCE_DATA_STORAGE_VERSION}"
        )

    created: list[dict[str, str]] = []
    if not scalar_indexes:
        return created
    available = set(table.schema.names)
    for entry in scalar_indexes:
        if isinstance(entry, str):
            column, index_type = entry, "BTREE"
        else:
            column, index_type = entry
        if column not in available:
            continue
        try:
            dataset.create_scalar_index(column, index_type=index_type)
        except Exception:
            # Index creation failures should not abort the conversion; the
            # data is still valid, indexes are an optimization.
            continue
        created.append({"column": column, "index_type": index_type})
    return created


def _table_from_pylist_with_blob_columns(
    pa: Any,
    lance: Any,
    rows: list[dict[str, Any]],
    *,
    schema: Any,
    blob_columns: set[str],
) -> Any:
    arrays = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if field.name in blob_columns:
            _validate_inline_blob_values(values, field.name)
            arrays.append(lance.blob_array(values))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _validate_inline_blob_values(values: list[Any], column: str) -> None:
    for value in values:
        if value is None or isinstance(value, (bytes, bytearray, memoryview)):
            continue
        raise TypeError(
            f"{column} must contain inline bytes only for published v1 bundles; "
            f"got {type(value).__name__}"
        )


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


def _episode_vector_dim(episode_rows: list[dict[str, Any]], column: str) -> int:
    for episode in episode_rows:
        for row in _as_vector_rows(episode.get(column)):
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
    source_repo_id: str | None,
    blob: bytes,
    episode_index: int,
    chunk_index: int,
    fps: float,
    num_frames: int,
    feature_info: dict[str, Any],
    session_id: str | None = None,
    embodiment_id: str | None = None,
) -> dict[str, Any]:
    height, width = _video_shape(feature_info)
    rel = str(video_path.relative_to(source))
    source_uri = _source_media_uri(source_repo_id, rel, video_path)
    media_id = _media_id(episode_index, camera_norm)
    source_payload = {
        "uri": source_uri,
        "repo_id": source_repo_id,
        "dataset_url": _hf_dataset_url(source_repo_id),
        "media_id": media_id,
        "relative_path": rel,
    }
    return {
        "media_id": media_id,
        "episode_index": episode_index,
        "camera_id": camera_norm,
        "camera_name": camera_key,
        "source": source_payload,
        "source_uri": source_uri,
        "source_dataset": source_repo_id,
        "source_dataset_url": _hf_dataset_url(source_repo_id),
        "source_media_id": media_id,
        "source_relative_path": rel,
        "source_episode_index": episode_index,
        "session_id": session_id,
        "embodiment_id": embodiment_id,
        "relative_path": rel,
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


def _media_id(episode_index: int, camera_norm: str) -> str:
    return f"episode_{episode_index:08d}_{camera_norm}"


def _camera_segment(
    *,
    camera_key: str,
    camera_norm: str,
    media_id: str,
    num_frames: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "camera_key": camera_key,
        "camera_column": camera_norm,
        "media_id": media_id,
        "from_timestamp": 0.0,
        "to_timestamp": (num_frames - 1) / fps if num_frames > 0 else None,
        "frame_start": 0,
        "frame_count": int(num_frames),
    }


def _source_media_uri(source_repo_id: str | None, rel: str, video_path: Path) -> str:
    if source_repo_id and "/" in source_repo_id:
        return f"hf://datasets/{source_repo_id}/{rel}"
    return str(video_path)


def _hf_dataset_url(source_repo_id: str | None) -> str | None:
    if source_repo_id and "/" in source_repo_id:
        return f"https://huggingface.co/datasets/{source_repo_id}"
    return None


def _task_segments_from_frames(
    frames: list[dict[str, Any]],
    *,
    timestamps: list[float],
    default_task_index: int,
    default_language: str | None,
    task_lookup: dict[int, str],
) -> list[dict[str, Any]]:
    if not frames:
        return []
    task_indices = [
        int(frame.get("task_index"))
        if frame.get("task_index") is not None
        else int(default_task_index)
        for frame in frames
    ]
    segments: list[dict[str, Any]] = []
    start = 0
    current = task_indices[0]
    for index, task_index in enumerate(task_indices[1:], start=1):
        if task_index == current:
            continue
        segments.append(
            _task_segment(
                current,
                start,
                index,
                timestamps,
                task_lookup=task_lookup,
                default_language=default_language,
            )
        )
        start = index
        current = task_index
    segments.append(
        _task_segment(
            current,
            start,
            len(task_indices),
            timestamps,
            task_lookup=task_lookup,
            default_language=default_language,
        )
    )
    return segments


def _task_segment(
    task_index: int,
    start_frame: int,
    end_frame_exclusive: int,
    timestamps: list[float],
    *,
    task_lookup: dict[int, str],
    default_language: str | None,
) -> dict[str, Any]:
    """Half-open task segment: covers frames [start_frame, end_frame_exclusive).

    `end_timestamp_exclusive` is the timestamp of the frame at
    `end_frame_exclusive` when in range, or one frame interval past the last
    covered timestamp when the segment ends at the trajectory's tail.
    """
    language = task_lookup.get(int(task_index)) or default_language
    start_ts = float(timestamps[start_frame]) if timestamps else None
    if not timestamps:
        end_ts = None
    elif end_frame_exclusive < len(timestamps):
        end_ts = float(timestamps[end_frame_exclusive])
    else:
        # Tail segment: extrapolate one frame interval past the last covered ts.
        last_covered = end_frame_exclusive - 1
        if last_covered <= 0:
            end_ts = float(timestamps[0])
        else:
            interval = float(timestamps[last_covered]) - float(timestamps[last_covered - 1])
            end_ts = float(timestamps[last_covered]) + interval
    return {
        "task_index": int(task_index),
        "language_instruction": language,
        "start_frame": int(start_frame),
        "end_frame_exclusive": int(end_frame_exclusive),
        "start_timestamp": start_ts,
        "end_timestamp_exclusive": end_ts,
    }


_TRAJECTORY_HASH_MAGIC = b"RLLAB_TRAJECTORY_V1\x00"  # 20 bytes


def _trajectory_sha256(
    timestamps: list[float],
    states: list[list[float]],
    actions: list[list[float]],
) -> str:
    """Deterministic SHA-256 over a little-endian binary trajectory encoding.

    Bytes (little-endian throughout):

        magic              : 20 bytes  b"RLLAB_TRAJECTORY_V1\\0"
        length             : int64
        state_dim          : int32
        action_dim         : int32
        timestamps         : float64 × length
        observation_state  : float32 × length × state_dim   (row-major)
        actions            : float32 × length × action_dim  (row-major)

    NaN and ±Inf are forbidden in the canonical trajectory arrays. JSON-based
    hashing is not valid for v1.0; cross-implementation reproducibility was the
    motivating reason for the binary form.
    """
    length = len(timestamps)
    if len(states) != length:
        raise ValueError(f"observation_state length {len(states)} != timestamps length {length}")
    if len(actions) != length:
        raise ValueError(f"actions length {len(actions)} != timestamps length {length}")
    state_dim = len(states[0]) if states and states[0] else 0
    action_dim = len(actions[0]) if actions and actions[0] else 0

    chunks: list[bytes] = [
        _TRAJECTORY_HASH_MAGIC,
        struct.pack("<q", length),
        struct.pack("<i", state_dim),
        struct.pack("<i", action_dim),
    ]

    if length:
        chunks.append(
            struct.pack(
                f"<{length}d",
                *(_finite_float(t, f"timestamps[{index}]") for index, t in enumerate(timestamps)),
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
                _finite_float(v, f"observation_state[{row_index}]")
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
                _finite_float(v, f"actions[{row_index}]")
                for v in row
            )
        chunks.append(struct.pack(f"<{len(flat_action)}f", *flat_action))

    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} contains non-finite value {value!r}")
    return out


def _write_stats_json(meta_dir: Path, episode_rows: list[dict[str, Any]]) -> None:
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for row in episode_rows:
        states.extend(_as_vector_rows(row.get("observation_state")))
        actions.extend(_as_vector_rows(row.get("actions")))
    stats = {
        "observation.state": _vector_stats(states),
        "action": _vector_stats(actions),
    }
    stats_dir = meta_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "state_body.json").write_text(
        json.dumps(
            {
                "schema_version": RLLAB_SCHEMA_VERSION,
                "modality": "state.body",
                "feature": "observation.state",
                **stats["observation.state"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stats_dir / "action_body.json").write_text(
        json.dumps(
            {
                "schema_version": RLLAB_SCHEMA_VERSION,
                "action": "action.body",
                "feature": "action",
                **stats["action"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_tasks_json(
    meta_dir: Path,
    task_lookup: dict[int, str],
    episode_rows: list[dict[str, Any]],
) -> None:
    task_counts: dict[int, int] = {}
    task_texts: dict[int, str] = {int(k): str(v) for k, v in task_lookup.items()}
    for row in episode_rows:
        task_index = int(row.get("task_index") or 0)
        task_counts[task_index] = task_counts.get(task_index, 0) + 1
        language = row.get("language_instruction")
        if isinstance(language, str) and language:
            task_texts.setdefault(task_index, language)

    payload = {
        "schema_version": RLLAB_SCHEMA_VERSION,
        "tasks": [
            {
                "task_index": task_index,
                "language_instruction": task_texts.get(task_index),
                "episode_count": task_counts.get(task_index, 0),
            }
            for task_index in sorted(set(task_texts) | set(task_counts))
        ],
    }
    tasks_jsonl = "\n".join(
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
        for row in payload["tasks"]
    )
    (meta_dir / "tasks.jsonl").write_text(tasks_jsonl + ("\n" if tasks_jsonl else ""), encoding="utf-8")


def _write_episodes_jsonl(
    meta_dir: Path,
    episode_rows: list[dict[str, Any]],
    split_by_episode: dict[int, str],
) -> None:
    lines = []
    for row in sorted(episode_rows, key=lambda item: int(item["episode_index"])):
        episode_index = int(row["episode_index"])
        lines.append(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "task_index": int(row.get("task_index") or 0),
                    "tasks": [row.get("language_instruction")]
                    if row.get("language_instruction")
                    else [],
                    "length": int(row.get("length") or 0),
                    "split": split_by_episode.get(episode_index, "train"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    (meta_dir / "episodes.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_splits_json(
    meta_dir: Path,
    episode_rows: list[dict[str, Any]],
    split_by_episode: dict[int, str],
) -> None:
    splits: dict[str, list[int]] = {}
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        split = split_by_episode.get(episode_index, "train")
        splits.setdefault(split, []).append(episode_index)
    payload = {
        "schema_version": RLLAB_SCHEMA_VERSION,
        "strategy": "source_split_or_all_train",
        "splits": {key: sorted(value) for key, value in sorted(splits.items())},
    }
    (meta_dir / "splits.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_vector_rows(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for item in value:
        if isinstance(item, list):
            rows.append([float(v) for v in item])
    return rows


def _vector_stats(vectors: list[list[float]]) -> dict[str, list[float]]:
    if not vectors:
        return {"mean": [], "std": [], "min": [], "max": [], "count": []}
    dim = max(len(vector) for vector in vectors)
    columns = [
        [float(vector[index]) for vector in vectors if index < len(vector) and math.isfinite(float(vector[index]))]
        for index in range(dim)
    ]
    means = [_mean(column) for column in columns]
    return {
        "mean": means,
        "std": [_std(column, mean) for column, mean in zip(columns, means)],
        "min": [min(column) if column else 0.0 for column in columns],
        "max": [max(column) if column else 0.0 for column in columns],
        "count": [len(column) for column in columns],
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], mean: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _video_modality_key(camera_key: str) -> str:
    if camera_key.startswith("observation.images."):
        return f"video.{camera_key.removeprefix('observation.images.')}"
    return f"video.{_normalize_camera_key(camera_key)}"


def _build_modalities(
    *,
    camera_keys: list[str],
    cameras_norm: list[str],
    state_dim: int,
    fps: float,
    episodes_path: str,
    frames_path: str,
    media_path: str,
    state_semantics: dict[str, Any],
) -> dict[str, Any]:
    modalities: dict[str, Any] = {
        "state.body": {
            "kind": "state",
            "source_key": "observation.state",
            "table": "episodes",
            "path": episodes_path,
            "column": "observation_state",
            "frame_table": "frames",
            "frame_path": frames_path,
            "frame_column": "observation_state",
            "names_ref": "meta/info.json#/features/observation.state/names",
            "shape": [int(state_dim)],
            "shape_policy": "single",
            "rate_hz": float(fps),
            "stats": "meta/stats/state_body.json",
            "semantics": dict(state_semantics),
        }
    }
    for camera_key, camera_norm in zip(camera_keys, cameras_norm):
        modalities[_video_modality_key(camera_key)] = {
            "kind": "video",
            "source_key": camera_key,
            "camera_key": camera_key,
            "camera_column": camera_norm,
            "table": "videos",
            "path": media_path,
            "media_id_column": "media_id",
            "blob_column": "video_blob",
            "segment_column": "camera_segments",
            "encoding": "rgb8_h264",
            "names_ref": f"meta/info.json#/features/{camera_key}/names",
            "shape_ref": f"meta/info.json#/features/{camera_key}/shape",
            "rate_hz": float(fps),
        }
    return modalities


def _default_unknown_action_semantics() -> dict[str, Any]:
    return {
        "command_type": "unknown",
        "absolute_or_delta": "unknown",
        "units": "unknown",
        "control_frame": "unknown",
        "applies_to_interval": "[t_i, t_{i+1})",
        "normalized": False,
    }


def _default_unknown_state_semantics() -> dict[str, Any]:
    return {
        "observation_type": "unknown",
        "units": "unknown",
        "control_frame": "unknown",
        "normalized": False,
    }


def _feature_names(info: dict[str, Any], feature_key: str, dim: int) -> list[str]:
    feature = (info.get("features") or {}).get(feature_key) or {}
    names = feature.get("names")
    if isinstance(names, list) and len(names) == int(dim) and all(isinstance(name, str) for name in names):
        return list(names)
    return []


def _has_specific_ffw_names(info: dict[str, Any], feature_key: str, dim: int) -> bool:
    names = _feature_names(info, feature_key, dim)
    return bool(names) and all(name in FFW_BG2_REV4_JOINT_SPECS for name in names)


def _is_ffw_family(info: dict[str, Any], source_repo_id: str | None) -> bool:
    robot_type = str(info.get("robot_type") or "").lower()
    robot_name = str(info.get("robot_name") or "").lower()
    repo = str(source_repo_id or "").lower()
    return (
        robot_type.startswith("ffw_")
        or robot_name.startswith("ffw_")
        or "ffw_bg2" in repo
        or "ffw_sg2" in repo
        or "ffw_arm_only" in repo
        or repo.startswith(("robotis/", "robotissw/", "dongkkka/"))
    )


def _ffw_joint_order_for_dim(dim: int, names: list[str]) -> list[str]:
    if names and all(name in FFW_BG2_REV4_JOINT_SPECS for name in names):
        return list(names)
    if int(dim) == 19:
        return list(FFW_BG2_REV4_JOINT_ORDER)
    if int(dim) == 16:
        return list(FFW_BG2_REV4_JOINT_ORDER[:16])
    return []


def _with_inferred_ffw_feature_names(
    info: dict[str, Any],
    *,
    source_repo_id: str | None,
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    """Upgrade generic FFW feature names to the inferred joint order.

    Some public LeRobot datasets advertise 19-D BG2 arrays but keep generic
    names such as ``joint_0`` for ``action``. The v2 registry points readers to
    meta/info.json via names_ref, so when we can infer the FFW layout from
    robot metadata and dimensionality, info.json should carry those names too.
    """

    if not _is_ffw_family(info, source_repo_id):
        return info
    out = deepcopy(info)
    features = out.setdefault("features", {})
    if not isinstance(features, dict):
        return out
    for feature_key, dim in (
        ("observation.state", state_dim),
        ("action", action_dim),
    ):
        joint_order = _ffw_joint_order_for_dim(dim, _feature_names(out, feature_key, dim))
        if not joint_order or _has_specific_ffw_names(out, feature_key, dim):
            continue
        feature = features.setdefault(feature_key, {})
        if isinstance(feature, dict):
            feature["names"] = list(joint_order)
    return out


def _ffw_joint_groups(joint_order: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for component, role, side in (
        ("left_arm", "arm", "left"),
        ("left_gripper", "gripper", "left"),
        ("right_arm", "arm", "right"),
        ("right_gripper", "gripper", "right"),
        ("head", "head", "center"),
        ("lift", "lift", "center"),
    ):
        indices = [
            index
            for index, name in enumerate(joint_order)
            if FFW_BG2_REV4_JOINT_SPECS.get(name, {}).get("component") == component
        ]
        if not indices:
            continue
        groups.append(
            {
                "name": component,
                "role": role,
                "side": side,
                "indices": indices,
                "joint_names": [joint_order[index] for index in indices],
            }
        )
    return groups


def _ffw_joint_layout(joint_order: list[str]) -> dict[str, Any]:
    joints = []
    for index, name in enumerate(joint_order):
        spec = dict(FFW_BG2_REV4_JOINT_SPECS[name])
        urdf_joint_type = str(spec["urdf_joint_type"])
        unit = "m" if urdf_joint_type == "prismatic" else "rad"
        joints.append(
            {
                "index": index,
                "name": name,
                "component": spec["component"],
                "role": spec["role"],
                "side": spec["side"],
                "unit": unit,
                "urdf_joint_type": urdf_joint_type,
                "axis": spec["axis"],
                "position_limit": {
                    "lower": spec["lower"],
                    "upper": spec["upper"],
                },
                "velocity_limit": spec["velocity"],
                "effort_limit": spec["effort"],
                **({"mimic_driver": True} if spec.get("mimic_driver") else {}),
            }
        )
    return {
        "robot_type": "ffw_bg2_rev4",
        "urdf_source": "ai_worker/ffw_description/urdf/ffw_bg2_rev4_follower/ffw_bg2_follower.urdf",
        "collection_source": "rllab-data-collection/config/bg2_topics.yaml",
        "joint_order": list(joint_order),
        "groups": _ffw_joint_groups(joint_order),
        "joints": joints,
    }


def _infer_state_semantics(
    info: dict[str, Any],
    *,
    source_repo_id: str | None,
    state_dim: int,
) -> dict[str, Any]:
    if _is_ffw_family(info, source_repo_id):
        joint_order = _ffw_joint_order_for_dim(
            state_dim,
            _feature_names(info, "observation.state", state_dim),
        )
        if joint_order:
            return {
                "observation_type": "joint_position",
                "units": "mixed",
                "control_frame": "robot_base",
                "normalized": False,
                "joint_layout": _ffw_joint_layout(joint_order),
            }
    return _default_unknown_state_semantics()


def _infer_action_semantics(
    info: dict[str, Any],
    *,
    source_repo_id: str | None,
    action_dim: int,
) -> dict[str, Any]:
    """Best-effort semantics for known ROBOTIS/FFW LeRobot datasets.

    LeRobot's feature schema tells us shape/names, but not whether the command
    is position, velocity, delta pose, or normalized. Keep unknown as the
    default and only fill concrete semantics for datasets whose robot family
    and dimensionality match the FFW joint-position recordings we use for BG2.
    """

    if _is_ffw_family(info, source_repo_id) and int(action_dim) in {16, 19, 25}:
        semantics = {
            "command_type": "joint_position",
            "absolute_or_delta": "absolute",
            "units": "mixed",
            "control_frame": "robot_base",
            "applies_to_interval": "[t_i, t_{i+1})",
            "normalized": False,
        }
        joint_order = _ffw_joint_order_for_dim(
            action_dim,
            _feature_names(info, "action", action_dim),
        )
        if joint_order:
            semantics["joint_layout"] = _ffw_joint_layout(joint_order)
        return semantics
    return _default_unknown_action_semantics()


def _build_actions(
    *,
    action_dim: int,
    fps: float,
    episodes_path: str,
    frames_path: str,
    action_semantics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "action.body": {
            "kind": "action",
            "source_key": "action",
            "table": "episodes",
            "path": episodes_path,
            "column": "actions",
            "frame_table": "frames",
            "frame_path": frames_path,
            "frame_column": "action",
            "names_ref": "meta/info.json#/features/action/names",
            "shape": [int(action_dim)],
            "shape_policy": "single",
            "rate_hz": float(fps),
            "stats": "meta/stats/action_body.json",
            "alignment": "same_frame_timestamp",
            "semantics": dict(action_semantics),
        }
    }


def _format_created_indexes(
    indexes_built: dict[str, list[dict[str, str]] | list[str]],
    *,
    episodes_path: str,
    frames_path: str | None,
    media_path: str | None,
) -> list[dict[str, Any]]:
    """Return one manifest entry per (table, index_type) actually created.

    Accepts either the new typed shape (list of {column, index_type} dicts) or
    the legacy bare-column-list shape, which is treated as all-BTREE.
    """
    table_paths = {"episodes": episodes_path, "frames": frames_path, "videos": media_path}
    out: list[dict[str, Any]] = []
    for kind, entries in indexes_built.items():
        path = table_paths.get(kind)
        if not path or not entries:
            continue
        grouped: dict[str, list[str]] = {}
        for entry in entries:
            if isinstance(entry, str):
                column, index_type = entry, "BTREE"
            else:
                column = entry.get("column")
                index_type = entry.get("index_type", "BTREE")
            if not column:
                continue
            grouped.setdefault(index_type, []).append(column)
        for index_type, columns in grouped.items():
            out.append(
                {
                    "table": path,
                    "index_type": index_type,
                    "columns": columns,
                    "status": "ready",
                }
            )
    return out


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
    output_layout: str,
    dataset_id: str | None,
    source_repo_id: str | None,
    info: dict[str, Any],
    indexes_built: dict[str, list[dict[str, str]] | list[str]] | None = None,
) -> None:
    is_hf = output_layout == "hf"
    episodes_path = "data/episodes.lance" if is_hf else "episodes.lance"
    frames_path = "data/frames.lance" if is_hf else "frames.lance"
    media_path = "data/videos.lance" if is_hf else "media.lance"
    has_videos_table = bool(has_media or is_hf)
    tables = {
        "episodes": episodes_path,
        "primary_training": episodes_path,
    }
    if include_frames:
        tables["frames"] = frames_path
    if has_videos_table:
        tables["videos" if is_hf else "media"] = media_path

    action_semantics = _infer_action_semantics(
        info,
        source_repo_id=source_repo_id,
        action_dim=action_dim,
    )
    state_semantics = _infer_state_semantics(
        info,
        source_repo_id=source_repo_id,
        state_dim=state_dim,
    )

    manifest = {
        "format": RLLAB_PUBLISHED_FORMAT if is_hf else RLLAB_SESSION_FORMAT,
        "schema_version": RLLAB_SCHEMA_VERSION,
        "source_format": f"lerobot_{layout}",
        "lance": {
            "data_storage_version": LANCE_DATA_STORAGE_VERSION,
            "blob_encoding": LANCE_BLOB_ENCODING,
            "published_blob_policy": PUBLISHED_BLOB_POLICY,
            "external_blob_uris_allowed": False,
            "requires_take_blobs": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "source": str(source),
        "primary_training_table": episodes_path,
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
                "the converter does not shift actions to the next state."
            ),
        },
        "modalities": _build_modalities(
            camera_keys=camera_keys,
            cameras_norm=cameras_norm,
            state_dim=state_dim,
            fps=fps,
            episodes_path=episodes_path,
            frames_path=frames_path,
            media_path=media_path,
            state_semantics=state_semantics,
        ),
        "actions": _build_actions(
            action_dim=action_dim,
            fps=fps,
            episodes_path=episodes_path,
            frames_path=frames_path,
            action_semantics=action_semantics,
        ),
        "training_targets": ["action.body"],
        "rates": {
            "fps": float(fps),
            "modalities": {
                "state.body": float(fps),
                **{_video_modality_key(key): float(fps) for key in camera_keys},
            },
            "actions": {"action.body": float(fps)},
        },
        "capabilities": {
            "inline_video_blobs": bool(has_media),
            "lance_blob_v2": bool(has_videos_table),
            "videos_table": bool(has_videos_table),
            "frames_table": bool(include_frames),
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
            "lazy_blob_columns": {media_path: ["video_blob"]} if has_videos_table else {},
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
            "fragment_strategy": {
                **({episodes_path: {"max_rows_per_file": TRAJ_MAX_ROWS_PER_FILE}}),
                **({frames_path: {"max_rows_per_file": TRAJ_MAX_ROWS_PER_FILE}} if include_frames else {}),
                **(
                    {
                        media_path: {
                            "max_bytes_per_file": VIDEOS_MAX_BYTES_PER_FILE,
                            "max_rows_per_file": VIDEOS_MAX_ROWS_PER_FILE,
                        }
                    }
                    if has_videos_table
                    else {}
                ),
            },
        },
        "indexes": {
            "created": _format_created_indexes(
                indexes_built or {},
                episodes_path=episodes_path,
                frames_path=frames_path if include_frames else None,
                media_path=media_path if has_videos_table else None,
            ),
            "recommended": [
                {"table": episodes_path, "columns": ["episode_index"]},
                *(
                    [{"table": frames_path, "columns": ["episode_index", "frame_index"]}]
                    if include_frames
                    else []
                ),
                *(
                    [{"table": media_path, "columns": ["media_id", "episode_index", "camera_id"]}]
                    if has_videos_table
                    else []
                ),
            ],
        },
        "tables": tables,
        "meta": {
            "info": "meta/info.json",
            "stats_dir": "meta/stats",
            "state_body_stats": "meta/stats/state_body.json",
            "action_body_stats": "meta/stats/action_body.json",
            "tasks_jsonl": "meta/tasks.jsonl",
            "episodes_jsonl": "meta/episodes.jsonl",
            "splits": "meta/splits.json",
        },
        "counts": {
            "episodes": int(episodes_written),
            "frames": int(frames_written),
            "videos": int(media_written),
        },
    }
    if is_hf:
        manifest["meta"]["sessions"] = "meta/sessions.json"
    text = json.dumps(manifest, indent=2, sort_keys=True)
    (target / "manifest.json").write_text(text, encoding="utf-8")


def _source_repo_id(info: dict[str, Any], source: Path) -> str | None:
    for key in ("repo_id", "dataset_repo_id", "hf_repo_id"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    return source.name if source.name else None


def _write_hf_sessions_json(
    target: Path,
    *,
    source: Path,
    source_repo_id: str | None,
    layout: str,
    episodes_written: int,
    frames_written: int,
    media_written: int,
) -> None:
    payload = [
        {
            "source_format": f"lerobot_{layout}",
            "source": str(source),
            "source_dataset": source_repo_id,
            "episodes": int(episodes_written),
            "frames": int(frames_written),
            "videos": int(media_written),
        }
    ]
    (target / "meta" / "sessions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_hf_readme(
    target: Path,
    *,
    dataset_id: str,
    source_repo_id: str | None,
    episodes_written: int,
    frames_written: int,
    media_written: int,
    fps: float,
    camera_keys: list[str],
) -> None:
    source_label = source_repo_id or "local LeRobot dataset"
    camera_text = ", ".join(camera_keys) if camera_keys else "none"
    text = f"""# {dataset_id}

Lance-formatted LeRobot dataset converted from `{source_label}` with
`lerobot2lance`.

## Tables

| Table | Purpose |
| --- | --- |
| `data/episodes.lance` | One row per episode: timestamps, state/action arrays, task text, and camera timestamp ranges. |
| `data/frames.lance` | One row per frame for browsing, QA, and frame-level filtering. |
| `data/videos.lance` | Canonical media table: one row per episode/camera MP4 with inline video blob and media metadata. |

## Summary

- Episodes: {episodes_written}
- Frames: {frames_written}
- Videos: {media_written}
- FPS: {fps:g}
- Cameras: {camera_text}

`manifest.json` records the table paths and training contract. Version history
should be managed with Hugging Face commits, branches, and tags.
"""
    (target / "README.md").write_text(text, encoding="utf-8")


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


def _build_episodes_schema(
    pa: Any,
    cameras_norm: list[str],
    *,
    state_dim: int,
    action_dim: int,
) -> Any:
    fields = [
        pa.field("episode_index", pa.int64(), nullable=False),
        pa.field("task_index", pa.int64()),
        pa.field("fps", pa.float64()),
        pa.field("length", pa.int64()),
        pa.field("timestamps", pa.list_(pa.float64())),
        pa.field(
            "observation_state",
            pa.large_list(pa.list_(pa.float32(), int(state_dim))),
        ),
        pa.field(
            "actions",
            pa.large_list(pa.list_(pa.float32(), int(action_dim))),
        ),
        pa.field("language_instruction", pa.string()),
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
        pa.field("trajectory_sha256", pa.string()),
        pa.field("split", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string()),
        pa.field("source_episode_index", pa.int64()),
        pa.field("session_id", pa.string()),
        pa.field("embodiment_id", pa.string()),
    ]
    return pa.schema(fields)


def _build_frames_schema(pa: Any, *, state_dim: int, action_dim: int) -> Any:
    return pa.schema(
        [
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("global_frame_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
            pa.field("task_index", pa.int64()),
            pa.field("observation_state", pa.list_(pa.float32(), int(state_dim))),
            pa.field("action", pa.list_(pa.float32(), int(action_dim))),
            pa.field("state_norm", pa.float32()),
            pa.field("action_norm", pa.float32()),
            pa.field("is_bad_frame", pa.bool_(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("embodiment_id", pa.string()),
        ]
    )


def _build_media_schema(pa: Any, lance: Any) -> Any:
    return pa.schema(
        [
            pa.field("media_id", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("camera_id", pa.string()),
            pa.field("camera_name", pa.string(), nullable=False),
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
            pa.field("source_dataset", pa.string()),
            pa.field("source_dataset_url", pa.string()),
            pa.field("source_media_id", pa.string()),
            pa.field("source_relative_path", pa.string()),
            pa.field("source_episode_index", pa.int64()),
            pa.field("session_id", pa.string()),
            pa.field("embodiment_id", pa.string()),
            pa.field("relative_path", pa.string()),
            lance.blob_field("video_blob"),
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
