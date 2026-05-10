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

            # Episode rows include the language/task caption from tasks.jsonl
            import lance

            ds = lance.dataset(str(target / "episodes.lance"))
            row = ds.scanner(
                columns=["episode_index", "language_instruction", "fps", "length"], limit=2
            ).to_table().to_pylist()
            self.assertEqual(row[0]["language_instruction"], "pick-and-place")
            self.assertEqual(row[0]["fps"], 30.0)
            self.assertEqual(row[0]["length"], 4)

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "rllab_lance_session_v1")
            self.assertEqual(manifest["primary_training_table"], "episodes.lance")
            self.assertEqual(manifest["training_row_unit"], "episode")
            self.assertEqual(manifest["training_index_column"], "episode_index")
            self.assertIsNone(manifest["source_episode_column"])
            self.assertIsNone(manifest["video_frame_offset_column"])
            self.assertEqual(manifest["training_columns"]["state"], "observation_state")
            self.assertEqual(manifest["training_columns"]["action"], "actions")
            self.assertEqual(manifest["camera_keys"], ["observation.images.cam_head"])
            self.assertEqual(manifest["camera_columns"], ["observation_images_cam_head"])
            self.assertEqual(manifest["state_dim"], 3)
            self.assertEqual(manifest["action_dim"], 3)

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
            self.assertIsNone(media_row["video_path"])
            self.assertEqual(media_row["from_timestamp"], 0.0)
            self.assertAlmostEqual(media_row["to_timestamp"], 0.1)
            self.assertEqual(media_row["num_frames"], 4)
            self.assertEqual(media_row["width_pixels"], 320)
            self.assertEqual(media_row["height_pixels"], 240)

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

    def test_no_video_blobs_omits_blob_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "lance"
            _write_v2_1_dataset(source)

            convert_lerobot_to_lance(source, target, include_video_blobs=False)

            import lance

            ds = lance.dataset(str(target / "episodes.lance"))
            row = ds.scanner(
                columns=["observation_images_cam_head_video_blob"], limit=1
            ).to_table().to_pylist()[0]
            handle = row["observation_images_cam_head_video_blob"]
            # Lance blob columns surface as a {position, size} handle even when
            # the underlying payload is omitted; size==0 confirms no blob bytes
            # were written.
            if handle is not None:
                self.assertEqual(handle.get("size", 0), 0)
            media = lance.dataset(str(target / "media.lance"))
            self.assertEqual(media.count_rows(), 2)

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
            for name in ("episodes.lance", "frames.lance", "videos.lance"):
                self.assertTrue((target / "data" / name).exists(), f"{name} missing")
            self.assertFalse((target / "media.lance").exists())

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "rllab_published_lance_dataset_v1")
            self.assertEqual(manifest["published_layout"], "rllab_published_dataset_v1")
            self.assertEqual(manifest["published_data_dir"], "data")
            self.assertEqual(manifest["dataset_id"], "rllab-postech/bg2-test")
            self.assertEqual(manifest["primary_training_table"], "data/episodes.lance")
            self.assertEqual(manifest["tables"]["episodes"], "data/episodes.lance")
            self.assertEqual(manifest["tables"]["frames"], "data/frames.lance")
            self.assertEqual(manifest["tables"]["videos"], "data/videos.lance")
            self.assertEqual(manifest["total_episodes"], 2)
            self.assertEqual(manifest["total_frames"], 6)
            self.assertEqual(manifest["total_videos"], 2)
            self.assertEqual(manifest["total_video_segments"], 2)

            import lance

            episodes = lance.dataset(str(target / "data" / "episodes.lance"))
            self.assertEqual(episodes.count_rows(), 2)
            videos = lance.dataset(str(target / "data" / "videos.lance"))
            self.assertEqual(videos.count_rows(), 2)
            sessions = json.loads((target / "meta" / "sessions.json").read_text())
            self.assertEqual(sessions[0]["episodes"], 2)


class ErrorPathTest(unittest.TestCase):
    def test_missing_source_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            convert_lerobot_to_lance(Path("/nonexistent/path"), Path("/tmp/out"))


if __name__ == "__main__":
    unittest.main()
