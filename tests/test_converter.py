from __future__ import annotations

import json
import subprocess
import sys
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


def _write_v2_1_dataset(
    root: Path,
    *,
    episodes: int = 2,
    length: int = 4,
    nonfinite_state: bool = False,
    vector_dim: int = 3,
    robot_type: str | None = None,
) -> None:
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
            "observation.state": {"dtype": "float32", "shape": [vector_dim]},
            "action": {"dtype": "float32", "shape": [vector_dim]},
        },
    }
    if robot_type is not None:
        info["robot_type"] = robot_type
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
                "observation.state": [
                    (
                        float("nan")
                        if nonfinite_state and ep == 0 and f == 0 and i == 0
                        else float(ep + f + i)
                    )
                    for i in range(vector_dim)
                ],
                "action": [float(ep + f + i + 1) for i in range(vector_dim)],
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
                ("observation.state", pa.list_(pa.float32(), vector_dim)),
                ("action", pa.list_(pa.float32(), vector_dim)),
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
            self.assertTrue((target / "meta" / "stats" / "state_body.json").exists())
            self.assertTrue((target / "meta" / "stats" / "action_body.json").exists())
            self.assertTrue((target / "meta" / "tasks.jsonl").exists())
            # v2: no aggregate meta/stats.json or meta/tasks.json
            self.assertFalse((target / "meta" / "stats.json").exists())
            self.assertFalse((target / "meta" / "tasks.json").exists())
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
            self.assertEqual(row[0]["camera_segments"][0]["media_id"], "episode_00000000_observation_images_cam_head")
            seg0 = row[0]["task_segments"][0]
            self.assertEqual(seg0["task_index"], 0)
            self.assertEqual(seg0["start_frame"], 0)
            self.assertEqual(seg0["end_frame_exclusive"], 4)
            self.assertNotIn("end_frame", seg0)
            self.assertEqual(len(row[0]["trajectory_sha256"]), 64)
            # trajectory_sha256 is binary-deterministic: same input must hash
            # to the same value across runs.
            from lerobot2lance.converter import _trajectory_sha256
            full = ds.scanner(
                columns=["timestamps", "observation_state", "actions"],
                limit=1,
            ).to_table().to_pylist()[0]
            self.assertEqual(
                row[0]["trajectory_sha256"],
                _trajectory_sha256(
                    full["timestamps"], full["observation_state"], full["actions"]
                ),
            )

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "rllab_lance_session_v2")
            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(manifest["lance"]["data_storage_version"], "2.2")
            self.assertEqual(manifest["lance"]["blob_encoding"], "lance.blob.v2")
            self.assertFalse(manifest["lance"]["external_blob_uris_allowed"])
            self.assertEqual(manifest["primary_training_table"], "episodes.lance")
            self.assertEqual(manifest["state_action_alignment"]["type"], "same_frame_timestamp")
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
                self.assertNotIn(forbidden, manifest, f"flat alias leaked: {forbidden}")
            self.assertIn("state.body", manifest["modalities"])
            self.assertIn("video.cam_head", manifest["modalities"])
            self.assertIn("action.body", manifest["actions"])
            self.assertEqual(manifest["modalities"]["state.body"]["shape_policy"], "single")
            self.assertEqual(manifest["modalities"]["video.cam_head"]["encoding"], "rgb8_h264")
            self.assertEqual(manifest["actions"]["action.body"]["shape_policy"], "single")
            self.assertTrue(manifest["capabilities"]["lance_blob_v2"])
            self.assertTrue(manifest["capabilities"]["camera_segments"])
            self.assertEqual(manifest["meta"]["tasks_jsonl"], "meta/tasks.jsonl")
            self.assertEqual(manifest["meta"]["state_body_stats"], "meta/stats/state_body.json")
            self.assertEqual(manifest["counts"]["episodes"], 2)
            self.assertEqual(manifest["counts"]["frames"], 8)
            self.assertEqual(manifest["counts"]["videos"], 2)
            self.assertNotIn("media", manifest["counts"])
            self.assertEqual(manifest["tables"]["media"], "media.lance")
            self.assertEqual(manifest["modalities"]["state.body"]["shape"], [3])
            self.assertEqual(manifest["actions"]["action.body"]["shape"], [3])
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
                    "source",
                    "source_uri",
                    "source_relative_path",
                    "relative_path",
                    "from_timestamp",
                    "to_timestamp",
                    "num_frames",
                    "width_pixels",
                    "height_pixels",
                ],
                limit=1,
            ).to_table().to_pylist()[0]
            self.assertEqual(media_row["camera_name"], "observation.images.cam_head")
            self.assertNotIn("media_type", media_row)
            self.assertNotIn("episode_id", media_row)
            self.assertTrue(media_row["source"]["uri"].endswith("episode_000000.mp4"))
            self.assertTrue(media_row["source_uri"].endswith("episode_000000.mp4"))
            self.assertEqual(
                media_row["source_relative_path"],
                "videos/chunk-000/observation.images.cam_head/episode_000000.mp4",
            )
            self.assertNotIn("video_path", media_row)
            self.assertNotIn("uri", media_row)
            self.assertEqual(media_row["from_timestamp"], 0.0)
            self.assertAlmostEqual(media_row["to_timestamp"], 0.1)
            self.assertEqual(media_row["num_frames"], 4)
            self.assertEqual(media_row["width_pixels"], 320)
            self.assertEqual(media_row["height_pixels"], 240)

            state_stats = json.loads(
                (target / "meta" / "stats" / "state_body.json").read_text()
            )
            self.assertEqual(state_stats["count"], [8, 8, 8])
            tasks_jsonl = (target / "meta" / "tasks.jsonl").read_text().splitlines()
            tasks = [json.loads(line) for line in tasks_jsonl if line.strip()]
            self.assertEqual(tasks[0]["language_instruction"], "pick-and-place")
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

    def test_hf_layout_video_blob_invariants(self) -> None:
        """C3/C5 invariants on videos.lance: media_id uniqueness,
        camera_segments resolution, sha256 round-trip, byte_size consistency.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "published"
            _write_v2_1_dataset(source, episodes=2, length=3)

            convert_lerobot_to_lance(
                source,
                target,
                output_layout="hf",
                dataset_id="rllab-postech/bg2-test",
            )

            import hashlib

            import lance

            videos = lance.dataset(str(target / "data" / "videos.lance"))
            rows = videos.scanner(
                columns=["media_id", "sha256", "byte_size"]
            ).to_table().to_pylist()

            # C3: media_id uniqueness within videos.lance
            media_ids = [r["media_id"] for r in rows]
            self.assertEqual(
                len(media_ids), len(set(media_ids)), "media_id duplicated"
            )

            # C5: video_blob round-trips through sha256 / byte_size
            blob_files = videos.take_blobs(
                "video_blob", indices=list(range(len(rows)))
            )
            try:
                for idx, row in enumerate(rows):
                    blob_bytes = blob_files[idx].read()
                    self.assertEqual(
                        row["byte_size"],
                        len(blob_bytes),
                        f"byte_size mismatch for {row['media_id']}",
                    )
                    self.assertEqual(
                        row["sha256"],
                        hashlib.sha256(blob_bytes).hexdigest(),
                        f"sha256 mismatch for {row['media_id']}",
                    )
            finally:
                for bf in blob_files:
                    bf.close()

            # C3: every camera_segments[*].media_id resolves to a videos row
            episodes = lance.dataset(str(target / "data" / "episodes.lance"))
            ep_rows = episodes.scanner(
                columns=["camera_segments"]
            ).to_table().to_pylist()
            referenced = {
                seg["media_id"]
                for ep in ep_rows
                for seg in ep["camera_segments"]
            }
            self.assertTrue(referenced)
            missing = referenced - set(media_ids)
            self.assertFalse(
                missing,
                f"camera_segments reference unknown media_id: {missing}",
            )

    def test_hf_layout_a8_denormalized_columns(self) -> None:
        """A8: episodes/frames/videos carry split, source_dataset,
        session_id, embodiment_id; values align across tables.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "published"
            _write_v2_1_dataset(source, episodes=2, length=3)

            convert_lerobot_to_lance(
                source,
                target,
                output_layout="hf",
                dataset_id="rllab-postech/bg2-test",
                embodiment_id="bg2_dual_arm_v1",
            )

            import lance

            episodes = lance.dataset(str(target / "data" / "episodes.lance"))
            frames = lance.dataset(str(target / "data" / "frames.lance"))
            videos = lance.dataset(str(target / "data" / "videos.lance"))

            for ds, table in (
                (episodes, "episodes"),
                (frames, "frames"),
                (videos, "videos"),
            ):
                names = ds.schema.names
                self.assertIn("source_dataset", names, table)
                self.assertIn("session_id", names, table)
                self.assertIn("embodiment_id", names, table)
            self.assertIn("split", episodes.schema.names)
            self.assertIn("source_episode_index", episodes.schema.names)
            self.assertIn("split", frames.schema.names)

            ep_rows = episodes.scanner(
                columns=[
                    "episode_index",
                    "split",
                    "source_dataset",
                    "source_episode_index",
                    "session_id",
                    "embodiment_id",
                ]
            ).to_table().to_pylist()
            for row in ep_rows:
                self.assertEqual(row["split"], "train")
                self.assertEqual(row["embodiment_id"], "bg2_dual_arm_v1")
                self.assertEqual(row["source_episode_index"], row["episode_index"])
                self.assertIsNotNone(row["session_id"])

            def _index_types(ds) -> dict[str, str]:
                return {idx["name"]: idx["type"] for idx in ds.list_indices()}

            for ds, label in (
                (episodes, "episodes"),
                (frames, "frames"),
                (videos, "videos"),
            ):
                types = _index_types(ds)
                for col in ("source_dataset_idx", "session_id_idx", "embodiment_id_idx"):
                    self.assertEqual(types.get(col), "Bitmap", f"{label}.{col}")
            for ds, label in ((episodes, "episodes"), (frames, "frames")):
                self.assertEqual(
                    _index_types(ds).get("split_idx"), "Bitmap", f"{label}.split"
                )

    def test_hf_layout_episode_frame_alignment(self) -> None:
        """C4 invariants: counts.frames matches sum of episode lengths,
        and frames.lance rows align with episodes.lance per-frame arrays.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "published"
            _write_v2_1_dataset(source, episodes=2, length=3)

            convert_lerobot_to_lance(
                source,
                target,
                output_layout="hf",
                dataset_id="rllab-postech/bg2-test",
            )

            import lance

            manifest = json.loads(
                (target / "manifest.json").read_text(encoding="utf-8")
            )
            episodes = lance.dataset(str(target / "data" / "episodes.lance"))
            frames = lance.dataset(str(target / "data" / "frames.lance"))

            ep_rows = episodes.scanner(
                columns=[
                    "episode_index",
                    "length",
                    "timestamps",
                    "observation_state",
                    "actions",
                ]
            ).to_table().to_pylist()
            ep_rows.sort(key=lambda r: r["episode_index"])

            # C4: counts.frames == sum(episodes.length)
            self.assertEqual(
                manifest["counts"]["frames"],
                sum(r["length"] for r in ep_rows),
            )
            self.assertEqual(
                manifest["counts"]["frames"],
                frames.count_rows(),
            )

            frame_rows = frames.scanner(
                columns=[
                    "episode_index",
                    "frame_index",
                    "timestamp",
                    "observation_state",
                    "action",
                ]
            ).to_table().to_pylist()

            # C4: each frame row matches the per-episode arrays at frame_index
            ep_by_index = {r["episode_index"]: r for r in ep_rows}
            for frame in frame_rows:
                ep = ep_by_index[frame["episode_index"]]
                fi = frame["frame_index"]
                self.assertAlmostEqual(
                    frame["timestamp"], ep["timestamps"][fi], places=6
                )
                self.assertEqual(
                    list(frame["observation_state"]),
                    list(ep["observation_state"][fi]),
                )
                self.assertEqual(
                    list(frame["action"]), list(ep["actions"][fi])
                )

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
            self.assertTrue((target / "meta" / "stats" / "state_body.json").exists())
            self.assertTrue((target / "meta" / "stats" / "action_body.json").exists())
            self.assertTrue((target / "meta" / "tasks.jsonl").exists())
            self.assertTrue((target / "meta" / "episodes.jsonl").exists())
            self.assertTrue((target / "meta" / "splits.json").exists())
            # v2: no aggregate compatibility sidecars
            self.assertFalse((target / "meta" / "stats.json").exists())
            self.assertFalse((target / "meta" / "tasks.json").exists())
            for name in ("episodes.lance", "frames.lance", "videos.lance"):
                self.assertTrue((target / "data" / name).exists(), f"{name} missing")
            self.assertFalse((target / "media.lance").exists())

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "rllab_published_lance_dataset_v2")
            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(manifest["dataset_id"], "rllab-postech/bg2-test")
            self.assertEqual(manifest["primary_training_table"], "data/episodes.lance")
            self.assertEqual(manifest["tables"]["episodes"], "data/episodes.lance")
            self.assertEqual(manifest["tables"]["frames"], "data/frames.lance")
            self.assertEqual(manifest["tables"]["videos"], "data/videos.lance")
            self.assertEqual(manifest["counts"]["episodes"], 2)
            self.assertEqual(manifest["counts"]["frames"], 6)
            self.assertEqual(manifest["counts"]["videos"], 2)
            self.assertEqual(manifest["modalities"]["state.body"]["column"], "observation_state")
            self.assertEqual(manifest["actions"]["action.body"]["column"], "actions")
            self.assertEqual(manifest["training_targets"], ["action.body"])
            semantics = manifest["actions"]["action.body"]["semantics"]
            for key in (
                "command_type",
                "absolute_or_delta",
                "units",
                "control_frame",
                "applies_to_interval",
                "normalized",
            ):
                self.assertIn(key, semantics)
            # v2: flat aliases and aggregate sidecar paths must be absent
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
                self.assertNotIn(forbidden, manifest, f"flat alias leaked: {forbidden}")

    def test_ffw_robot_type_gets_concrete_joint_action_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "published"
            _write_v2_1_dataset(
                source,
                episodes=2,
                length=3,
                vector_dim=19,
                robot_type="ffw_bg2_rev4",
            )

            convert_lerobot_to_lance(
                source,
                target,
                output_layout="hf",
                dataset_id="rllab-postech/bg2-test",
            )

            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            semantics = manifest["actions"]["action.body"]["semantics"]
            self.assertEqual(semantics["command_type"], "joint_position")
            self.assertEqual(semantics["absolute_or_delta"], "absolute")
            self.assertEqual(semantics["units"], "mixed")
            self.assertEqual(semantics["control_frame"], "robot_base")
            self.assertIs(semantics["normalized"], False)
            self.assertNotIn("stats", manifest.get("meta", {}))
            self.assertNotIn("tasks", manifest.get("meta", {}))

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

            import pyarrow as pa

            frames_ds = lance.dataset(str(target / "data" / "frames.lance"))
            blob_field = videos.schema.field("video_blob")
            self.assertEqual(
                getattr(blob_field.type, "extension_name", None), "lance.blob.v2"
            )
            self.assertEqual(
                frames_ds.schema.field("observation_state").type,
                pa.list_(pa.float32(), 19),
            )
            self.assertEqual(
                episodes.schema.field("observation_state").type,
                pa.large_list(pa.list_(pa.float32(), 19)),
            )

            def _index_types(ds) -> dict[str, str]:
                return {idx["name"]: idx["type"] for idx in ds.list_indices()}

            video_idx_types = _index_types(videos)
            self.assertEqual(video_idx_types.get("media_id_idx"), "BTree")
            self.assertEqual(video_idx_types.get("episode_index_idx"), "BTree")
            self.assertEqual(video_idx_types.get("camera_id_idx"), "Bitmap")

            episode_idx_types = _index_types(episodes)
            self.assertEqual(episode_idx_types.get("episode_index_idx"), "BTree")
            self.assertEqual(episode_idx_types.get("task_index_idx"), "Bitmap")

            frame_idx_types = _index_types(frames_ds)
            self.assertEqual(frame_idx_types.get("global_frame_index_idx"), "BTree")
            self.assertEqual(frame_idx_types.get("episode_index_idx"), "BTree")
            self.assertEqual(frame_idx_types.get("frame_index_idx"), "BTree")
            self.assertEqual(frame_idx_types.get("task_index_idx"), "Bitmap")
            self.assertEqual(frame_idx_types.get("is_bad_frame_idx"), "Bitmap")

            created = manifest["indexes"]["created"]
            tables_indexed = {entry["table"] for entry in created}
            self.assertIn("data/videos.lance", tables_indexed)
            self.assertIn("data/episodes.lance", tables_indexed)
            # Manifest must record both BTREE and BITMAP entries for frames.
            frame_index_types = {
                entry["index_type"]
                for entry in created
                if entry["table"] == "data/frames.lance"
            }
            self.assertEqual(frame_index_types, {"BTREE", "BITMAP"})
            self.assertEqual(
                manifest["reader_hints"]["lazy_blob_columns"],
                {"data/videos.lance": ["video_blob"]},
            )
            frag_strategy = manifest["reader_hints"]["fragment_strategy"]
            self.assertEqual(
                frag_strategy["data/videos.lance"]["max_bytes_per_file"],
                2 * 1024 * 1024 * 1024,
            )
            validation = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_bundle.py",
                    str(target),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertTrue(json.loads(validation.stdout)["ok"])

    def test_rejects_nonfinite_trajectory_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lerobot"
            target = Path(tmpdir) / "published"
            _write_v2_1_dataset(source, episodes=1, length=2, nonfinite_state=True)

            with self.assertRaisesRegex(ValueError, "non-finite"):
                convert_lerobot_to_lance(
                    source,
                    target,
                    output_layout="hf",
                    dataset_id="rllab-postech/bad-nan",
                )


class ErrorPathTest(unittest.TestCase):
    def test_missing_source_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            convert_lerobot_to_lance(Path("/nonexistent/path"), Path("/tmp/out"))


if __name__ == "__main__":
    unittest.main()
