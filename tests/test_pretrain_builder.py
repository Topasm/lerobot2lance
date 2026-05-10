from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from lerobot2lance.converter import _trajectory_sha256


try:
    import lance
    import pyarrow as pa

    HAS_LANCE_DEPS = True
except ImportError:  # pragma: no cover - skipped when extras unavailable
    HAS_LANCE_DEPS = False


def _write_converted_19d_bundle(
    root: Path,
    *,
    name: str,
    repo_id: str,
    camera_key: str,
    camera_column: str,
) -> None:
    bundle = root / name
    (bundle / "data").mkdir(parents=True)
    (bundle / "meta").mkdir(parents=True)

    states = [[float(frame)] * 19 for frame in range(2)]
    actions = [[float(frame + 1)] * 19 for frame in range(2)]
    episode_rows = [
        {
            "episode_index": 0,
            "task_index": 0,
            "fps": 10.0,
            "length": 2,
            "timestamps": [0.0, 0.1],
            "observation_state": states,
            "actions": actions,
            "language_instruction": "pick",
            "camera_segments": [
                {
                    "camera_key": camera_key,
                    "camera_column": camera_column,
                    "media_id": f"episode_00000000_{camera_column}",
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.1,
                    "frame_start": 0,
                    "frame_count": 2,
                }
            ],
            "task_segments": [
                {
                    "task_index": 0,
                    "language_instruction": "pick",
                    "start_frame": 0,
                    "end_frame_exclusive": 2,
                    "start_timestamp": 0.0,
                    "end_timestamp_exclusive": 0.2,
                }
            ],
            "trajectory_sha256": _trajectory_sha256([0.0, 0.1], states, actions),
            "split": "train",
            "source_dataset": repo_id,
            "source_episode_index": 0,
            "session_id": repo_id,
            "embodiment_id": "ffw_bg2_rev4",
        }
    ]
    frame_rows = [
        {
            "episode_index": 0,
            "frame_index": frame,
            "global_frame_index": frame,
            "timestamp": frame / 10.0,
            "task_index": 0,
            "observation_state": states[frame],
            "action": actions[frame],
            "state_norm": 1.0,
            "action_norm": 1.0,
            "is_bad_frame": False,
            "split": "train",
            "source_dataset": repo_id,
            "session_id": repo_id,
            "embodiment_id": "ffw_bg2_rev4",
        }
        for frame in range(2)
    ]
    video_rows = [
        {
            "media_id": f"episode_00000000_{camera_column}",
            "episode_index": 0,
            "camera_id": camera_column,
            "camera_name": camera_key,
            "source": {
                "uri": "videos/episode_000000.mp4",
                "repo_id": repo_id,
                "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
                "media_id": f"episode_00000000_{camera_column}",
                "relative_path": "videos/episode_000000.mp4",
            },
            "source_uri": "videos/episode_000000.mp4",
            "source_dataset": repo_id,
            "source_dataset_url": f"https://huggingface.co/datasets/{repo_id}",
            "source_media_id": f"episode_00000000_{camera_column}",
            "source_relative_path": "videos/episode_000000.mp4",
            "source_episode_index": 0,
            "session_id": repo_id,
            "embodiment_id": "ffw_bg2_rev4",
            "relative_path": "videos/episode_000000.mp4",
            "video_blob": b"abc",
            "from_timestamp": 0.0,
            "to_timestamp": 0.1,
            "num_frames": 2,
            "chunk_index": 0,
            "file_index": 0,
            "sha256": "abc",
            "byte_size": 3,
            "width_pixels": 320,
            "height_pixels": 240,
            "fps": 10.0,
            "codec": "h264",
        }
    ]

    episode_schema = pa.schema(
        [
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64()),
            pa.field("fps", pa.float64()),
            pa.field("length", pa.int64()),
            pa.field("timestamps", pa.list_(pa.float64())),
            pa.field("observation_state", pa.large_list(pa.list_(pa.float32(), 19))),
            pa.field("actions", pa.large_list(pa.list_(pa.float32(), 19))),
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
    )
    frame_schema = pa.schema(
        [
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("global_frame_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
            pa.field("task_index", pa.int64()),
            pa.field("observation_state", pa.list_(pa.float32(), 19)),
            pa.field("action", pa.list_(pa.float32(), 19)),
            pa.field("state_norm", pa.float32()),
            pa.field("action_norm", pa.float32()),
            pa.field("is_bad_frame", pa.bool_(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("embodiment_id", pa.string()),
        ]
    )
    video_schema = pa.schema(
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
    lance.write_dataset(
        pa.Table.from_pylist(episode_rows, schema=episode_schema),
        str(bundle / "data" / "episodes.lance"),
        mode="overwrite",
        data_storage_version="2.2",
    )
    lance.write_dataset(
        pa.Table.from_pylist(frame_rows, schema=frame_schema),
        str(bundle / "data" / "frames.lance"),
        mode="overwrite",
        data_storage_version="2.2",
    )
    lance.write_dataset(
        _table_with_blob(pa, video_rows, video_schema, "video_blob"),
        str(bundle / "data" / "videos.lance"),
        mode="overwrite",
        data_storage_version="2.2",
    )

    manifest = {
        "format": "rllab_published_lance_dataset_v2",
        "schema_version": "2.0",
        "dataset_id": name,
        "source_repo_id": repo_id,
        "source_robot_type": "ffw_bg2_rev4",
        "pretrain_tier": "A_bg2_full_19d",
        "modalities": {
            "state.body": {"kind": "state", "shape": [19], "shape_policy": "single"},
            f"video.{camera_column}": {
                "kind": "video",
                "camera_key": camera_key,
                "camera_column": camera_column,
            },
        },
        "actions": {"action.body": {"kind": "action", "shape": [19], "shape_policy": "single"}},
        "rates": {"fps": 10.0},
        "counts": {"episodes": 1, "frames": 2, "videos": 1},
        "tables": {
            "episodes": "data/episodes.lance",
            "frames": "data/frames.lance",
            "videos": "data/videos.lance",
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _table_with_blob(pa: object, rows: list[dict], schema: object, blob_column: str):
    arrays = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if field.name == blob_column:
            arrays.append(lance.blob_array(values))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


@unittest.skipUnless(HAS_LANCE_DEPS, "requires pyarrow + lance")
class PretrainBuilderTest(unittest.TestCase):
    def test_merge_writes_single_videos_table_media_contract(self) -> None:
        with self.subTest("build"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                converted = tmp / "converted_19d"
                _write_converted_19d_bundle(
                    converted,
                    name="a",
                    repo_id="Org/A",
                    camera_key="observation.images.cam_head",
                    camera_column="observation_images_cam_head",
                )
                _write_converted_19d_bundle(
                    converted,
                    name="b",
                    repo_id="Org/B",
                    camera_key="observation.images.cam_wrist_left",
                    camera_column="observation_images_cam_wrist_left",
                )
                out = tmp / "pretrain"
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/build_pretrain_19d_lance.py",
                        "--converted-root",
                        str(converted),
                        "--output",
                        str(out),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).resolve().parents[1],
                )

                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["format"], "rllab_published_lance_dataset_v2")
                self.assertEqual(manifest["schema_version"], "2.0")
                self.assertEqual(manifest["primary_training_table"], "data/train_episodes.lance")
                self.assertEqual(manifest["state_action_alignment"]["type"], "same_frame_timestamp")
                self.assertIn("state.body", manifest["modalities"])
                self.assertIn("action.body", manifest["actions"])
                self.assertIn("video.cam_head", manifest["modalities"])
                self.assertIn("video.cam_wrist_left", manifest["modalities"])
                self.assertTrue(manifest["capabilities"]["camera_segments"])
                self.assertTrue(manifest["capabilities"]["modality_registry_v2"])
                self.assertTrue(manifest["capabilities"]["fixed_size_state_action"])
                self.assertTrue(manifest["capabilities"]["action_semantics"])
                self.assertEqual(manifest["counts"]["episodes"], 2)
                self.assertEqual(manifest["counts"]["frames"], 4)
                self.assertEqual(manifest["counts"]["videos"], 2)
                self.assertNotIn("media", manifest["counts"])
                self.assertEqual(manifest["meta"]["tasks_jsonl"], "meta/tasks.jsonl")
                self.assertEqual(manifest["meta"]["episodes_jsonl"], "meta/episodes.jsonl")
                self.assertEqual(manifest["meta"]["splits"], "meta/splits.json")
                self.assertNotIn("stats", manifest)
                self.assertNotIn("tasks", manifest)
                for forbidden in (
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
                ):
                    self.assertNotIn(forbidden, manifest)

                episodes = lance.dataset(str(out / "data" / "episodes.lance"))
                self.assertEqual(episodes.count_rows(), 2)
                self.assertFalse(
                    any(name.endswith("_video_blob") for name in episodes.schema.names)
                )
                episode = episodes.scanner(
                    columns=["camera_segments", "task_segments", "trajectory_sha256"],
                    limit=1,
                ).to_table().to_pylist()[0]
                self.assertEqual(episode["task_segments"][0]["task_index"], 0)
                self.assertEqual(len(episode["trajectory_sha256"]), 64)

                train_episodes = lance.dataset(str(out / "data" / "train_episodes.lance"))
                self.assertEqual(train_episodes.count_rows(), 2)
                self.assertFalse(
                    any(name.endswith("_video_blob") for name in train_episodes.schema.names)
                )

                videos = lance.dataset(str(out / "data" / "videos.lance"))
                row = videos.scanner(
                    columns=[
                        "source_uri",
                        "source_dataset_url",
                        "source_video_table",
                        "source_media_id",
                        "source_relative_path",
                    ],
                    limit=1,
                ).to_table().to_pylist()[0]
                video_blob = _read_blob(videos, "video_blob")
                self.assertEqual(video_blob, b"abc")
                self.assertEqual(row["source_uri"], "videos/episode_000000.mp4")
                self.assertEqual(row["source_dataset_url"], "https://huggingface.co/datasets/Org/A")
                self.assertTrue(row["source_video_table"].endswith("/data/videos.lance"))
                self.assertEqual(row["source_relative_path"], "videos/episode_000000.mp4")

                state_stats = json.loads((out / "meta" / "stats" / "state_body.json").read_text())
                action_stats = json.loads((out / "meta" / "stats" / "action_body.json").read_text())
                self.assertEqual(state_stats["count"], [4] * 19)
                self.assertEqual(action_stats["count"], [4] * 19)
                self.assertFalse((out / "meta" / "stats.json").exists())
                self.assertFalse((out / "meta" / "tasks.json").exists())
                self.assertTrue((out / "meta" / "tasks.jsonl").exists())
                self.assertTrue((out / "meta" / "episodes.jsonl").exists())
                self.assertTrue((out / "meta" / "splits.json").exists())
                task = json.loads((out / "meta" / "tasks.jsonl").read_text().splitlines()[0])
                self.assertEqual(task["language_instruction"], "pick")


def _read_blob(ds: object, column: str, index: int = 0) -> bytes:
    handle = ds.take_blobs(column, indices=[index])[0]
    try:
        data = handle.readall() if hasattr(handle, "readall") else handle.read()
    finally:
        if hasattr(handle, "close"):
            handle.close()
    return bytes(data)


if __name__ == "__main__":
    unittest.main()
