#!/usr/bin/env python3
"""RLLAB Lance v2.0 strict validator.

The validator is intentionally stricter than a reader.  It is meant to be run
after conversion and before upload, so it checks schema, counts, sidecars,
trajectory integrity, media blob integrity, and declared scalar indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot2lance.converter import (
    RLLAB_PUBLISHED_FORMAT,
    RLLAB_SESSION_FORMAT,
    _trajectory_sha256,
)


ALLOWED_FORMATS = {RLLAB_PUBLISHED_FORMAT, RLLAB_SESSION_FORMAT}


FORBIDDEN_MANIFEST_KEYS = {
    "published_layout",
    "published_data_dir",
    "source_dataset",
    "source_session_count",
    "training_row_unit",
    "training_index_column",
    "source_episode_column",
    "video_frame_offset_column",
    "frame_table",
    "media_table",
    "state_column",
    "action_column",
    "training_columns",
    "frame_columns",
    "state_dim",
    "action_dim",
    "camera_keys",
    "camera_columns",
    "camera_key_to_column",
    "fps",
    "media_mode",
    "camera_storage",
    "blob_storage",
    "total_episodes",
    "total_frames",
    "total_videos",
    "total_video_segments",
    "stats",
    "tasks",
    "media_path_columns",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to a converted Lance bundle")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full validation result as JSON.",
    )
    parser.add_argument(
        "--skip-blob-bytes",
        action="store_true",
        help="Skip video byte reads and sha256/byte_size checks.",
    )
    args = parser.parse_args()

    result = validate_bundle(args.bundle, check_blob_bytes=not args.skip_blob_bytes)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0 if result["ok"] else 1


def validate_bundle(bundle: Path, *, check_blob_bytes: bool = True) -> dict[str, Any]:
    try:
        import lance
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("validate_bundle requires pyarrow and lance") from exc

    bundle = bundle.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        return failure(bundle, ["manifest.json missing"], warnings)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episodes_path = resolve_table_path(bundle, manifest, "episodes")
    frames_path = resolve_table_path(bundle, manifest, "frames")
    videos_path = resolve_table_path(bundle, manifest, "videos")
    for label, path in (
        ("episodes", episodes_path),
        ("frames", frames_path),
        ("videos", videos_path),
    ):
        if path is None:
            errors.append(f"{label} table not declared in manifest.tables")
        elif not path.exists():
            errors.append(f"{label} table path does not exist: {path}")
    if errors:
        return failure(bundle, errors, warnings)

    episodes = lance.dataset(str(episodes_path))
    frames = lance.dataset(str(frames_path)) if frames_path else None
    videos = lance.dataset(str(videos_path)) if videos_path else None

    validate_manifest(manifest, errors)
    validate_sidecars(bundle, manifest, errors, warnings)

    state_dim = registry_dim(manifest, ["modalities", "state.body", "shape"])
    action_dim = registry_dim(manifest, ["actions", "action.body", "shape"])
    if state_dim is None:
        errors.append("modalities.state.body.shape missing from manifest")
    if action_dim is None:
        errors.append("actions.action.body.shape missing from manifest")
    if state_dim is not None and action_dim is not None:
        validate_schemas(pa, episodes, frames, videos, state_dim, action_dim, errors)

    episode_rows = scan_all(
        episodes,
        [
            "episode_index",
            "length",
            "timestamps",
            "observation_state",
            "actions",
            "camera_segments",
            "trajectory_sha256",
        ],
    )
    frame_rows = scan_all(
        frames,
        [
            "episode_index",
            "frame_index",
            "global_frame_index",
            "timestamp",
            "observation_state",
            "action",
        ],
    )
    video_rows = scan_all(
        videos,
        [
            "media_id",
            "episode_index",
            "camera_id",
            "camera_name",
            "source_relative_path",
            "relative_path",
            "sha256",
            "byte_size",
        ],
    )

    validate_counts(manifest, episodes, frames, videos, episode_rows, errors)
    validate_trajectories(episode_rows, errors)
    validate_frame_alignment(episode_rows, frame_rows, errors)
    validate_media(videos, video_rows, episode_rows, errors, check_blob_bytes)
    validate_indexes(manifest, {"episodes": episodes, "frames": frames, "videos": videos}, errors)

    summary = {
        "bundle": str(bundle),
        "format": manifest.get("format"),
        "schema_version": manifest.get("schema_version"),
        "counts": manifest.get("counts"),
        "tables": {
            "episodes": str(episodes_path.relative_to(bundle)) if episodes_path else None,
            "frames": str(frames_path.relative_to(bundle)) if frames_path else None,
            "videos": str(videos_path.relative_to(bundle)) if videos_path else None,
        },
        "rows": {
            "episodes": episodes.count_rows(),
            "frames": frames.count_rows() if frames else 0,
            "videos": videos.count_rows() if videos else 0,
        },
    }
    return {
        "ok": not errors,
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
    }


def failure(bundle: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "ok": False,
        "summary": {"bundle": str(bundle.resolve())},
        "errors": errors,
        "warnings": warnings,
    }


def print_human(result: dict[str, Any]) -> None:
    summary = result["summary"]
    status = "OK" if result["ok"] else "FAILED"
    print(f"{status}: {summary.get('bundle')}")
    if "rows" in summary:
        print(f"  rows: {summary['rows']}")
    if "counts" in summary:
        print(f"  manifest counts: {summary['counts']}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")
    for error in result["errors"]:
        print(f"  error: {error}")


def resolve_table_path(bundle: Path, manifest: dict[str, Any], name: str) -> Path | None:
    tables = manifest.get("tables") or {}
    value = tables.get(name)
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    return bundle / str(value)


def validate_manifest(manifest: dict[str, Any], errors: list[str]) -> None:
    required = [
        "format",
        "schema_version",
        "dataset_id",
        "source",
        "source_format",
        "lance",
        "counts",
        "tables",
        "modalities",
        "actions",
        "rates",
        "state_action_alignment",
        "indexes",
        "reader_hints",
        "capabilities",
        "meta",
    ]
    for key in required:
        if key not in manifest:
            errors.append(f"manifest missing canonical key: {key}")
    fmt = manifest.get("format")
    schema_version = str(manifest.get("schema_version") or "")
    if fmt in {"rllab_published_lance_dataset_v1", "rllab_lance_session_v1"} or schema_version.startswith("1."):
        errors.append(
            "v1 bundle detected; v2.0 requires re-conversion from the source LeRobot dataset"
        )
    if fmt not in ALLOWED_FORMATS:
        allowed = ", ".join(sorted(ALLOWED_FORMATS))
        errors.append(
            f"manifest.format must be one of {{{allowed}}}, got {fmt!r}"
        )
    if not schema_version.startswith("2."):
        errors.append("manifest.schema_version must be 2.x for v2 conformance")
    for key in sorted(FORBIDDEN_MANIFEST_KEYS & set(manifest)):
        errors.append(f"v2 manifest contains forbidden legacy alias key: {key}")
    lance_block = manifest.get("lance") or {}
    if lance_block.get("blob_encoding") != "lance.blob.v2":
        errors.append("manifest.lance.blob_encoding must be lance.blob.v2")
    if data_storage_version_tuple(lance_block.get("data_storage_version")) < (2, 2):
        errors.append("manifest.lance.data_storage_version must be >= 2.2")
    if lance_block.get("external_blob_uris_allowed") is not False:
        errors.append("manifest.lance.external_blob_uris_allowed must be false")
    capabilities = manifest.get("capabilities") or {}
    if capabilities.get("modality_registry_v2") is not True:
        errors.append("capabilities.modality_registry_v2 must be true")
    if "modality_registry_v1" in capabilities:
        errors.append("capabilities.modality_registry_v1 is forbidden in v2")
    if capabilities.get("fixed_size_state_action") is not True:
        errors.append("capabilities.fixed_size_state_action must be true")
    if capabilities.get("action_semantics") is not True:
        errors.append("capabilities.action_semantics must be true")
    if capabilities.get("lance_blob_v2") is not True and capabilities.get("inline_video_blobs"):
        errors.append("capabilities.lance_blob_v2 must be true when videos are inline")
    reader_hints = manifest.get("reader_hints") or {}
    for key in ("legacy_normalization", "legacy_aliases_available"):
        if key in reader_hints:
            errors.append(f"reader_hints.{key} is forbidden in v2")
    meta = manifest.get("meta") or {}
    for key in ("stats", "tasks"):
        if key in meta:
            errors.append(f"manifest.meta.{key} is forbidden in v2")
    for meta_key, meta_value in meta.items():
        if not isinstance(meta_value, str):
            continue
        normalized = meta_value.replace("\\", "/")
        if normalized in {"meta/stats.json", "meta/tasks.json"}:
            errors.append(
                f"manifest.meta.{meta_key} references forbidden root sidecar path: {meta_value}"
            )


def validate_sidecars(
    bundle: Path,
    manifest: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    for rel in (
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/splits.json",
        "meta/sessions.json",
        "meta/stats/state_body.json",
        "meta/stats/action_body.json",
    ):
        if not (bundle / rel).exists():
            errors.append(f"canonical sidecar missing: {rel}")

    if (bundle / "meta/stats.json").exists():
        errors.append("root compatibility sidecar meta/stats.json is forbidden in v2")
    if (bundle / "meta/tasks.json").exists():
        errors.append("root compatibility sidecar meta/tasks.json is forbidden in v2")


def registry_dim(manifest: dict[str, Any], path: list[str]) -> int | None:
    value: Any = manifest
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, list) and value:
        return int(value[0])
    return None


def validate_schemas(
    pa: Any,
    episodes: Any,
    frames: Any,
    videos: Any,
    state_dim: int,
    action_dim: int,
    errors: list[str],
) -> None:
    require_columns(
        episodes.schema.names,
        "episodes",
        [
            "episode_index",
            "task_index",
            "fps",
            "length",
            "timestamps",
            "observation_state",
            "actions",
            "language_instruction",
            "camera_segments",
            "task_segments",
            "trajectory_sha256",
            "split",
            "source_dataset",
            "source_episode_index",
            "session_id",
            "embodiment_id",
        ],
        errors,
    )
    if frames:
        require_columns(
            frames.schema.names,
            "frames",
            [
                "episode_index",
                "frame_index",
                "global_frame_index",
                "timestamp",
                "task_index",
                "observation_state",
                "action",
                "is_bad_frame",
                "split",
                "source_dataset",
                "session_id",
                "embodiment_id",
            ],
            errors,
        )
    if videos:
        require_columns(
            videos.schema.names,
            "videos",
            [
                "media_id",
                "episode_index",
                "camera_id",
                "camera_name",
                "source",
                "source_uri",
                "source_dataset",
                "source_dataset_url",
                "source_media_id",
                "source_relative_path",
                "source_episode_index",
                "session_id",
                "embodiment_id",
                "relative_path",
                "video_blob",
                "from_timestamp",
                "to_timestamp",
                "num_frames",
                "sha256",
                "byte_size",
                "width_pixels",
                "height_pixels",
                "fps",
                "codec",
            ],
            errors,
        )
    for column in episodes.schema.names:
        if column.endswith("_video_blob"):
            errors.append(f"episodes contains forbidden video blob column: {column}")
        if column.endswith("_from_timestamp") or column.endswith("_to_timestamp"):
            errors.append(f"episodes contains forbidden camera timestamp alias column: {column}")
    if videos:
        for column in ("episode_id", "media_type", "uri", "video_path"):
            if column in videos.schema.names:
                errors.append(f"videos contains forbidden legacy alias column: {column}")

    expected_episode_state = pa.large_list(pa.list_(pa.float32(), state_dim))
    expected_episode_action = pa.large_list(pa.list_(pa.float32(), action_dim))
    expected_frame_state = pa.list_(pa.float32(), state_dim)
    expected_frame_action = pa.list_(pa.float32(), action_dim)
    if episodes.schema.field("observation_state").type != expected_episode_state:
        errors.append("episodes.observation_state is not large_list<fixed_size_list<float32,state_dim>>")
    if episodes.schema.field("actions").type != expected_episode_action:
        errors.append("episodes.actions is not large_list<fixed_size_list<float32,action_dim>>")
    if frames and frames.schema.field("observation_state").type != expected_frame_state:
        errors.append("frames.observation_state is not fixed_size_list<float32,state_dim>")
    if frames and frames.schema.field("action").type != expected_frame_action:
        errors.append("frames.action is not fixed_size_list<float32,action_dim>")
    if videos and getattr(videos.schema.field("video_blob").type, "extension_name", None) != "lance.blob.v2":
        errors.append("videos.video_blob is not Lance Blob v2")
    if videos and data_storage_version_tuple(getattr(videos, "data_storage_version", "")) < (2, 2):
        errors.append("videos.lance data_storage_version is < 2.2")


def require_columns(
    actual: list[str],
    table: str,
    required: list[str],
    errors: list[str],
) -> None:
    names = set(actual)
    for column in required:
        if column not in names:
            errors.append(f"{table} missing required v2 column: {column}")


def scan_all(ds: Any, columns: list[str]) -> list[dict[str, Any]]:
    if ds is None:
        return []
    available = [column for column in columns if column in ds.schema.names]
    return ds.scanner(columns=available).to_table().to_pylist()


def validate_counts(
    manifest: dict[str, Any],
    episodes: Any,
    frames: Any,
    videos: Any,
    episode_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    counts = manifest.get("counts") or {}
    expected_frames = sum(int(row.get("length") or 0) for row in episode_rows)
    checks = {
        "episodes": episodes.count_rows(),
        "frames": frames.count_rows() if frames else 0,
        "videos": videos.count_rows() if videos else 0,
    }
    for key, observed in checks.items():
        if counts.get(key) is not None and int(counts[key]) != int(observed):
            errors.append(f"manifest counts.{key}={counts[key]} != table rows {observed}")
    if frames and frames.count_rows() != expected_frames:
        errors.append(f"frames rows {frames.count_rows()} != sum(episodes.length) {expected_frames}")


def validate_trajectories(rows: list[dict[str, Any]], errors: list[str]) -> None:
    for row in rows:
        ep = row.get("episode_index")
        timestamps = row.get("timestamps") or []
        states = row.get("observation_state") or []
        actions = row.get("actions") or []
        if len(timestamps) != int(row.get("length") or len(timestamps)):
            errors.append(f"episode {ep}: timestamps length does not match length")
        try:
            digest = _trajectory_sha256(timestamps, states, actions)
        except ValueError as exc:
            errors.append(f"episode {ep}: {exc}")
            continue
        if digest != row.get("trajectory_sha256"):
            errors.append(f"episode {ep}: trajectory_sha256 mismatch")


def validate_frame_alignment(
    episode_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    episodes = {int(row["episode_index"]): row for row in episode_rows}
    global_indices = set()
    for frame in frame_rows:
        ep = int(frame["episode_index"])
        fi = int(frame["frame_index"])
        global_indices.add(int(frame["global_frame_index"]))
        episode = episodes.get(ep)
        if episode is None:
            errors.append(f"frame references missing episode_index {ep}")
            continue
        if fi >= len(episode["timestamps"]):
            errors.append(f"frame {ep}/{fi}: frame_index out of range")
            continue
        if not close_float(frame["timestamp"], episode["timestamps"][fi]):
            errors.append(f"frame {ep}/{fi}: timestamp differs from episode trajectory")
        if not close_vector(frame["observation_state"], episode["observation_state"][fi]):
            errors.append(f"frame {ep}/{fi}: observation_state differs from episode trajectory")
        if not close_vector(frame["action"], episode["actions"][fi]):
            errors.append(f"frame {ep}/{fi}: action differs from episode trajectory")
    if len(global_indices) != len(frame_rows):
        errors.append("frames.global_frame_index is not unique")


def validate_media(
    videos: Any,
    video_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    errors: list[str],
    check_blob_bytes: bool,
) -> None:
    if videos is None:
        return
    media_ids = [row.get("media_id") for row in video_rows]
    if len(set(media_ids)) != len(media_ids):
        errors.append("videos.media_id contains duplicates")
    segment_ids = {
        segment.get("media_id")
        for episode in episode_rows
        for segment in (episode.get("camera_segments") or [])
    }
    missing = sorted(segment_ids - set(media_ids))
    if missing:
        errors.append(f"camera_segments reference missing media_id values: {missing[:5]}")

    blob_meta_rows = videos.scanner(columns=["video_blob"]).to_table().to_pylist()
    for index, row in enumerate(blob_meta_rows):
        metadata = row.get("video_blob")
        if isinstance(metadata, dict):
            uri = metadata.get("blob_uri") or metadata.get("uri")
            if uri:
                errors.append(f"video row {index}: external blob URI is not allowed")

    if not check_blob_bytes:
        return
    for index, row in enumerate(video_rows):
        handle = videos.take_blobs("video_blob", indices=[index])[0]
        try:
            data = handle.readall() if hasattr(handle, "readall") else handle.read()
        finally:
            handle.close()
        digest = hashlib.sha256(data).hexdigest()
        if row.get("sha256") and row["sha256"] != digest:
            errors.append(f"video row {index}: sha256 mismatch")
        if row.get("byte_size") is not None and int(row["byte_size"]) != len(data):
            errors.append(f"video row {index}: byte_size mismatch")


def validate_indexes(manifest: dict[str, Any], datasets: dict[str, Any], errors: list[str]) -> None:
    created = (manifest.get("indexes") or {}).get("created") or []
    path_to_name = {}
    for name in ("episodes", "frames", "videos"):
        path = resolve_table_path(Path("."), manifest, name)
        if path is not None:
            path_to_name[str(path)] = name
    for entry in created:
        table = entry.get("table")
        name = path_to_name.get(str(Path(table))) or table_name_from_path(str(table))
        ds = datasets.get(name)
        if ds is None:
            continue
        actual = actual_indexes(ds)
        expected_type = normalize_index_type(entry.get("index_type"))
        for column in entry.get("columns") or []:
            if actual.get(column) != expected_type:
                errors.append(
                    f"index missing/mismatched for {table}.{column}: "
                    f"expected {expected_type}, got {actual.get(column)}"
                )


def actual_indexes(ds: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for index in ds.list_indices():
        fields = index.get("fields") or []
        if len(fields) == 1:
            out[str(fields[0])] = normalize_index_type(index.get("type"))
    return out


def table_name_from_path(path: str) -> str:
    name = Path(path).name
    if name == "media.lance":
        return "videos"
    return name.removesuffix(".lance")


def normalize_index_type(value: Any) -> str:
    text = str(value or "").upper()
    if text == "BTREE":
        return "BTREE"
    if text == "BITMAP":
        return "BITMAP"
    if text == "LABEL_LIST":
        return "LABEL_LIST"
    return text


def close_float(a: Any, b: Any) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(af) and math.isfinite(bf) and abs(af - bf) <= 1e-6


def close_vector(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if not close_float(left, right):
            return False
    return True


def data_storage_version_tuple(value: Any) -> tuple[int, int]:
    parts = str(value or "0.0").split(".", 1)
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return (0, 0)
    return (major, minor)


if __name__ == "__main__":
    sys.exit(main())
