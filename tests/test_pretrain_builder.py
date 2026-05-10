from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


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
            f"{camera_column}_video_blob": b"abc",
            f"{camera_column}_from_timestamp": 0.0,
            f"{camera_column}_to_timestamp": 0.1,
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
        }
        for frame in range(2)
    ]
    video_rows = [
        {
            "media_id": f"episode_000000_{camera_column}",
            "episode_id": "episode_000000",
            "episode_index": 0,
            "camera_id": camera_column,
            "camera_name": camera_key,
            "media_type": "video",
            "uri": "videos/episode_000000.mp4",
            "relative_path": "videos/episode_000000.mp4",
            "video_blob": b"abc",
            "video_path": None,
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
            pa.field("observation_state", pa.list_(pa.list_(pa.float32()))),
            pa.field("actions", pa.list_(pa.list_(pa.float32()))),
            pa.field("language_instruction", pa.string()),
            pa.field(
                f"{camera_column}_video_blob",
                pa.large_binary(),
                metadata={b"lance-encoding:blob": b"true"},
            ),
            pa.field(f"{camera_column}_from_timestamp", pa.float64()),
            pa.field(f"{camera_column}_to_timestamp", pa.float64()),
        ]
    )
    frame_schema = pa.schema(
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
    video_schema = pa.schema(
        [
            pa.field("media_id", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("camera_id", pa.string()),
            pa.field("camera_name", pa.string()),
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
    lance.write_dataset(
        pa.Table.from_pylist(episode_rows, schema=episode_schema),
        str(bundle / "data" / "episodes.lance"),
        mode="overwrite",
    )
    lance.write_dataset(
        pa.Table.from_pylist(frame_rows, schema=frame_schema),
        str(bundle / "data" / "frames.lance"),
        mode="overwrite",
    )
    lance.write_dataset(
        pa.Table.from_pylist(video_rows, schema=video_schema),
        str(bundle / "data" / "videos.lance"),
        mode="overwrite",
    )

    manifest = {
        "format": "rllab_published_lance_dataset_v1",
        "dataset_id": name,
        "source_repo_id": repo_id,
        "source_dataset": repo_id,
        "source_robot_type": "ffw_bg2_rev4",
        "pretrain_tier": "A_bg2_full_19d",
        "fps": 10.0,
        "state_dim": 19,
        "action_dim": 19,
        "camera_keys": [camera_key],
        "camera_columns": [camera_column],
        "total_episodes": 1,
        "total_frames": 2,
        "total_videos": 1,
        "total_video_segments": 1,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@unittest.skipUnless(HAS_LANCE_DEPS, "requires pyarrow + lance")
class PretrainBuilderTest(unittest.TestCase):
    def test_default_merge_keeps_source_media_references_without_blobs(self) -> None:
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
                    cwd=Path(__file__).resolve().parents[1],
                )

                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["media_mode"], "source_reference")
                self.assertFalse(manifest["training_ready"])
                self.assertEqual(manifest["primary_training_table"], "data/train_episodes.lance")
                self.assertEqual(manifest["blob_storage"]["episodes"], "metadata_only")
                self.assertEqual(
                    manifest["blob_storage"]["train_episodes"],
                    "metadata_only_source_reference",
                )
                self.assertEqual(manifest["camera_keys"], [])
                self.assertEqual(
                    manifest["source_camera_keys"],
                    [
                        "observation.images.cam_head",
                        "observation.images.cam_wrist_left",
                    ],
                )

                episodes = lance.dataset(str(out / "data" / "episodes.lance"))
                self.assertEqual(episodes.count_rows(), 2)
                self.assertFalse(
                    any(name.endswith("_video_blob") for name in episodes.schema.names)
                )

                train_episodes = lance.dataset(str(out / "data" / "train_episodes.lance"))
                self.assertEqual(train_episodes.count_rows(), 2)
                self.assertFalse(
                    any(name.endswith("_video_blob") for name in train_episodes.schema.names)
                )

                videos = lance.dataset(str(out / "data" / "videos.lance"))
                row = videos.scanner(
                    columns=[
                        "video_blob",
                        "source_dataset_url",
                        "source_video_table",
                        "source_media_id",
                        "source_relative_path",
                    ],
                    limit=1,
                ).to_table().to_pylist()[0]
                self.assertIsNone(row["video_blob"])
                self.assertEqual(row["source_dataset_url"], "https://huggingface.co/datasets/Org/A")
                self.assertTrue(row["source_video_table"].endswith("/data/videos.lance"))
                self.assertEqual(row["source_relative_path"], "videos/episode_000000.mp4")

    def test_copy_video_blobs_writes_training_and_viewer_media_blobs(self) -> None:
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
                out = tmp / "pretrain"
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/build_pretrain_19d_lance.py",
                        "--converted-root",
                        str(converted),
                        "--output",
                        str(out),
                        "--copy-video-blobs",
                    ],
                    check=True,
                    cwd=Path(__file__).resolve().parents[1],
                )

                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(manifest["training_ready"])
                self.assertEqual(manifest["media_mode"], "videos_table")
                self.assertEqual(manifest["camera_keys"], ["observation.images.cam_head"])
                self.assertEqual(manifest["blob_storage"]["episodes"], "metadata_only")
                self.assertEqual(manifest["blob_storage"]["train_episodes"], "video_blob_columns")
                self.assertEqual(manifest["blob_storage"]["videos"], "video_blob_column")

                episodes = lance.dataset(str(out / "data" / "episodes.lance"))
                self.assertFalse(
                    any(name.endswith("_video_blob") for name in episodes.schema.names)
                )

                train_episodes = lance.dataset(str(out / "data" / "train_episodes.lance"))
                self.assertIn(
                    "observation_images_cam_head_video_blob",
                    train_episodes.schema.names,
                )
                train_blob = _read_blob(
                    train_episodes,
                    "observation_images_cam_head_video_blob",
                )
                self.assertEqual(train_blob, b"abc")

                videos = lance.dataset(str(out / "data" / "videos.lance"))
                video_blob = _read_blob(videos, "video_blob")
                self.assertEqual(video_blob, b"abc")


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
