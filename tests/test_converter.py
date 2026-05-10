from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lerobot2lance import convert_lerobot_to_lance


try:
    import lance  # noqa: F401
    import pyarrow  # noqa: F401

    HAS_CONVERSION_DEPS = True
except ImportError:  # pragma: no cover - skipped when extras unavailable
    HAS_CONVERSION_DEPS = False


def _atom(atom_type: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + atom_type + payload


# Small but valid-ish MP4 prefix: ftyp + moov->trak->tkhd
_FAKE_MP4 = (
    _atom(b"ftyp", b"mp42\x00\x00\x00\x00mp42isom")
    + _atom(
        b"moov",
        _atom(
            b"trak",
            _atom(
                b"tkhd",
                b"\x00\x00\x00\x07"
                + b"\x00" * 72
                + (320 << 16).to_bytes(4, "big")
                + (240 << 16).to_bytes(4, "big"),
            ),
        ),
    )
)


def _write_v2_1_dataset(root: Path, *, episodes: int = 2, length: int = 4) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": episodes,
        "total_frames": episodes * length,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.cam_head": {
                "dtype": "video",
                "shape": [240, 320, 3],
                "info": {
                    "video.fps": 30,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                },
            },
            "observation.state": {"dtype": "float32", "shape": [3]},
            "action": {"dtype": "float32", "shape": [3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick-and-place"}) + "\n",
        encoding="utf-8",
    )
    episode_lines = "\n".join(
        json.dumps(
            {"episode_index": i, "tasks": ["pick-and-place"], "length": length}
        )
        for i in range(episodes)
    )
    (root / "meta" / "episodes.jsonl").write_text(episode_lines + "\n", encoding="utf-8")

    import pyarrow as pa
    import pyarrow.parquet as pq

    for ep in range(episodes):
        rows = [
            {
                "timestamp": float(f) / 30.0,
                "frame_index": f,
                "episode_index": ep,
                "index": ep * length + f,
                "task_index": 0,
                "observation.state": [float(ep), float(f), 0.0],
                "action": [float(ep), float(f), 1.0],
            }
            for f in range(length)
        ]
        schema = pa.schema(
            [
                ("timestamp", pa.float32()),
                ("frame_index", pa.int64()),
                ("episode_index", pa.int64()),
                ("index", pa.int64()),
                ("task_index", pa.int64()),
                ("observation.state", pa.list_(pa.float32(), 3)),
                ("action", pa.list_(pa.float32(), 3)),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            root / f"data/chunk-000/episode_{ep:06d}.parquet",
        )

        cam_dir = root / "videos" / "chunk-000" / "observation.images.cam_head"
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / f"episode_{ep:06d}.mp4").write_bytes(_FAKE_MP4)


@unittest.skipUnless(HAS_CONVERSION_DEPS, "requires pyarrow + lance")
class LerobotToLanceConversionTest(unittest.TestCase):
    def test_v2_1_writes_lance_session_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "lance"
            _write_v2_1_dataset(source, episodes=2, length=4)

            events: list[tuple[str, dict]] = []
            report = convert_lerobot_to_lance(
                source,
                target,
                output_layout="session",
                progress_callback=lambda kind, payload: events.append((kind, payload)),
            )

            self.assertEqual(report["layout_detected"], "v2_1")
            self.assertEqual(report["episodes_written"], 2)
            self.assertEqual(report["frames_written"], 8)
            self.assertEqual(report["media_written"], 2)
            self.assertNotIn("videos_written", report)
            self.assertEqual(report["fps"], 30.0)
            self.assertEqual(report["cameras"], ["observation_images_cam_head"])
            self.assertEqual([e[0] for e in events], ["episode_converted"] * 2)
            for name in ("episodes.lance", "frames.lance", "media.lance"):
                self.assertTrue((target / name).exists(), f"{name} missing")
            self.assertFalse((target / "videos.lance").exists())
            self.assertTrue((target / "manifest.json").exists())
            # Source info.json copied — downstream camera_info discovery relies on this.
            self.assertTrue((target / "meta" / "info.json").exists())
            self.assertTrue((target / "meta" / "stats.json").exists())
            self.assertTrue((target / "meta" / "stats" / "state_body.json").exists())
            self.assertTrue((target / "meta" / "stats" / "action_body.json").exists())
            self.assertTrue((target / "meta" / "tasks.json").exists())
            self.assertTrue((target / "meta" / "tasks.jsonl").exists())
            self.assertTrue((target / "meta" / "episodes.jsonl").exists())
            self.assertTrue((target / "meta" / "splits.json").exists())

            # Episode rows include the language/task caption from tasks.jsonl
            import lance

            ds = lance.dataset(str(target / "episodes.lance"))
            row = ds.scanner(
                columns=[
                    "episode_index",
                    "language_instruction",
                    "fps",
                    "length",
                    "camera_segments",
                    "task_segments",
                    "trajectory_sha256",
                ],
                limit=2,
            ).to_table().to_pylist()
            self.assertEqual(row[0]["language_instruction"], "pick-and-place")
            self.assertEqual(row[0]["fps"], 30.0)
            self.assertEqual(row[0]["length"], 4)
            self.assertEqual(row[0]["camera_segments"][0]["media_id"], "episode_000000_observation_images_cam_head")
            self.assertEqual(row[0]["task_segments"][0]["task_index"], 0)
            self.assertEqual(len(row[0]["trajectory_sha256"]), 64)

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "rllab_lance_session_v1")
            self.assertEqual(manifest["primary_training_table"], "episodes.lance")
            self.assertEqual(manifest["media_mode"], "videos_table")
            self.assertEqual(manifest["blob_storage"]["episodes"], "absent")
            self.assertEqual(manifest["blob_storage"]["media"], "video_blob_column")
            self.assertEqual(manifest["training_row_unit"], "episode")
            self.assertEqual(manifest["training_index_column"], "episode_index")
            self.assertIsNone(manifest["source_episode_column"])
            self.assertIsNone(manifest["video_frame_offset_column"])
            self.assertEqual(manifest["training_columns"]["state"], "observation_state")
            self.assertEqual(manifest["training_columns"]["action"], "actions")
            self.assertEqual(manifest["frame_columns"]["state"], "observation_state")
            self.assertEqual(manifest["frame_columns"]["action"], "action")
            self.assertEqual(manifest["state_action_alignment"]["type"], "same_frame_timestamp")
            self.assertEqual(manifest["camera_keys"], ["observation.images.cam_head"])
            self.assertEqual(manifest["camera_columns"], ["observation_images_cam_head"])
            self.assertEqual(
                manifest["camera_key_to_column"],
                {"observation.images.cam_head": "observation_images_cam_head"},
            )
            self.assertIn("state.body", manifest["modalities"])
            self.assertIn("video.cam_head", manifest["modalities"])
            self.assertIn("action.body", manifest["actions"])
            self.assertTrue(manifest["capabilities"]["camera_segments"])
            self.assertEqual(manifest["meta"]["tasks_jsonl"], "meta/tasks.jsonl")
            self.assertEqual(manifest["meta"]["state_body_stats"], "meta/stats/state_body.json")
            self.assertEqual(manifest["counts"]["episodes"], 2)
            self.assertEqual(manifest["counts"]["frames"], 8)
            self.assertEqual(manifest["counts"]["videos"], 2)
            self.assertNotIn("media", manifest["counts"])
            self.assertEqual(manifest["state_dim"], 3)
            self.assertEqual(manifest["action_dim"], 3)
            self.assertNotIn(
                "observation_images_cam_head_video_blob",
                ds.schema.names,
            )

            frames = lance.dataset(str(target / "frames.lance"))
            frame = frames.scanner(
                columns=[
                    "global_frame_index",
                    "state_norm",
                    "action_norm",
                    "is_bad_frame",
                ],
                limit=1,
            ).to_table().to_pylist()[0]
            self.assertEqual(frame["global_frame_index"], 0)
            self.assertFalse(frame["is_bad_frame"])

            media = lance.dataset(str(target / "media.lance"))
            media_row = media.scanner(
                columns=[
                    "camera_name",
                    "media_type",
                    "source",
                    "source_uri",
                    "source_relative_path",
                    "relative_path",
                    "video_path",
                    "from_timestamp",
                    "to_timestamp",
                    "num_frames",
                    "width_pixels",
                    "height_pixels",
                ],
                limit=1,
            ).to_table().to_pylist()[0]
            self.assertEqual(media_row["camera_name"], "observation.images.cam_head")
            self.assertEqual(media_row["media_type"], "video")
            self.assertTrue(media_row["source"]["uri"].endswith("episode_000000.mp4"))
            self.assertTrue(media_row["source_uri"].endswith("episode_000000.mp4"))
            self.assertEqual(
                media_row["source_relative_path"],
                "videos/chunk-000/observation.images.cam_head/episode_000000.mp4",
            )
            self.assertIsNone(media_row["video_path"])
            self.assertEqual(media_row["from_timestamp"], 0.0)
            self.assertAlmostEqual(media_row["to_timestamp"], 0.1)
            self.assertEqual(media_row["num_frames"], 4)
            self.assertEqual(media_row["width_pixels"], 320)
            self.assertEqual(media_row["height_pixels"], 240)

            stats = json.loads((target / "meta" / "stats.json").read_text())
            self.assertEqual(stats["observation.state"]["count"], [8, 8, 8])
            self.assertEqual(stats["action"]["count"], [8, 8, 8])
            tasks = json.loads((target / "meta" / "tasks.json").read_text())
            self.assertEqual(tasks["tasks"][0]["language_instruction"], "pick-and-place")
            splits = json.loads((target / "meta" / "splits.json").read_text())
            self.assertEqual(splits["splits"]["train"], [0, 1])

    def test_refuses_existing_target_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "lance"
            _write_v2_1_dataset(source)
            convert_lerobot_to_lance(source, target)

            with self.assertRaises(FileExistsError):
                convert_lerobot_to_lance(source, target)

            convert_lerobot_to_lance(source, target, overwrite=True)

    def test_limit_only_processes_first_n_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "lance"
            _write_v2_1_dataset(source, episodes=3, length=2)

            report = convert_lerobot_to_lance(source, target, limit=1)

            self.assertEqual(report["episodes_written"], 1)
            self.assertEqual(report["frames_written"], 2)
            self.assertEqual(report["media_written"], 1)

    def test_episode_table_never_stores_video_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "lance"
            _write_v2_1_dataset(source)

            convert_lerobot_to_lance(source, target)

            import lance

            ds = lance.dataset(str(target / "data" / "episodes.lance"))
            self.assertNotIn("observation_images_cam_head_video_blob", ds.schema.names)
            media = lance.dataset(str(target / "data" / "videos.lance"))
            self.assertEqual(media.count_rows(), 2)
            media_row = (
                media.scanner(columns=["video_blob"], limit=1)
                .to_table()
                .to_pylist()[0]
            )
            self.assertTrue(media_row["video_blob"])

    def test_hf_layout_writes_data_tables_and_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "published"
            _write_v2_1_dataset(source, episodes=2, length=3)

            report = convert_lerobot_to_lance(
                source,
                target,
                output_layout="hf",
                dataset_id="rllab-postech/bg2-test",
            )

            self.assertEqual(report["output_layout"], "hf")
            self.assertEqual(report["dataset_id"], "rllab-postech/bg2-test")
            self.assertTrue((target / "manifest.json").exists())
            self.assertTrue((target / "README.md").exists())
            self.assertTrue((target / "meta" / "info.json").exists())
            self.assertTrue((target / "meta" / "sessions.json").exists())
            self.assertTrue((target / "meta" / "stats.json").exists())
            self.assertTrue((target / "meta" / "stats" / "state_body.json").exists())
            self.assertTrue((target / "meta" / "tasks.json").exists())
            self.assertTrue((target / "meta" / "tasks.jsonl").exists())
            self.assertTrue((target / "meta" / "episodes.jsonl").exists())
            self.assertTrue((target / "meta" / "splits.json").exists())
            for name in ("episodes.lance", "frames.lance", "videos.lance"):
                self.assertTrue((target / "data" / name).exists(), f"{name} missing")
            self.assertFalse((target / "media.lance").exists())

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "rllab_published_lance_dataset_v1")
            self.assertEqual(manifest["published_layout"], "rllab_published_dataset_v1")
            self.assertEqual(manifest["published_data_dir"], "data")
            self.assertEqual(manifest["dataset_id"], "rllab-postech/bg2-test")
            self.assertEqual(manifest["primary_training_table"], "data/episodes.lance")
            self.assertEqual(manifest["media_mode"], "videos_table")
            self.assertEqual(manifest["camera_storage"], "videos_table")
            self.assertEqual(manifest["blob_storage"]["episodes"], "absent")
            self.assertEqual(manifest["blob_storage"]["videos"], "video_blob_column")
            self.assertEqual(manifest["tables"]["episodes"], "data/episodes.lance")
            self.assertEqual(manifest["tables"]["frames"], "data/frames.lance")
            self.assertEqual(manifest["tables"]["videos"], "data/videos.lance")
            self.assertEqual(manifest["counts"]["episodes"], 2)
            self.assertEqual(manifest["counts"]["frames"], 6)
            self.assertEqual(manifest["counts"]["videos"], 2)
            self.assertEqual(manifest["meta"]["stats"], "meta/stats.json")
            self.assertEqual(manifest["meta"]["splits"], "meta/splits.json")
            self.assertEqual(manifest["modalities"]["state.body"]["column"], "observation_state")
            self.assertEqual(manifest["actions"]["action.body"]["column"], "actions")
            self.assertEqual(manifest["stats"]["source_table"], "data/episodes.lance")
            self.assertEqual(manifest["tasks"]["path"], "meta/tasks.json")
            self.assertEqual(manifest["total_episodes"], 2)
            self.assertEqual(manifest["total_frames"], 6)
            self.assertEqual(manifest["total_videos"], 2)
            self.assertEqual(manifest["total_video_segments"], 2)

            import lance

            episodes = lance.dataset(str(target / "data" / "episodes.lance"))
            self.assertEqual(episodes.count_rows(), 2)
            self.assertNotIn(
                "observation_images_cam_head_video_blob",
                episodes.schema.names,
            )
            videos = lance.dataset(str(target / "data" / "videos.lance"))
            self.assertEqual(videos.count_rows(), 2)
            sessions = json.loads((target / "meta" / "sessions.json").read_text())
            self.assertEqual(sessions[0]["episodes"], 2)

            blob_field = videos.schema.field("video_blob")
            self.assertEqual(
                blob_field.metadata.get(b"lance-encoding:blob"), b"true"
            )

            video_indices = {idx["name"] for idx in videos.list_indices()}
            self.assertIn("episode_index_idx", video_indices)
            self.assertIn("camera_id_idx", video_indices)
            self.assertIn("media_id_idx", video_indices)
            episode_indices = {idx["name"] for idx in episodes.list_indices()}
            self.assertIn("episode_index_idx", episode_indices)

            created = manifest["indexes"]["created"]
            tables_indexed = {entry["table"] for entry in created}
            self.assertIn("data/videos.lance", tables_indexed)
            self.assertIn("data/episodes.lance", tables_indexed)
            self.assertEqual(
                manifest["reader_hints"]["lazy_blob_columns"],
                {"data/videos.lance": ["video_blob"]},
            )
            frag_strategy = manifest["reader_hints"]["fragment_strategy"]
            self.assertEqual(
                frag_strategy["data/videos.lance"]["max_bytes_per_file"],
                2 * 1024 * 1024 * 1024,
            )


class ErrorPathTest(unittest.TestCase):
    def test_missing_source_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            convert_lerobot_to_lance(Path("/nonexistent/path"), Path("/tmp/out"))


if __name__ == "__main__":
    unittest.main()
