from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_converter import _write_v2_1_dataset  # type: ignore  # noqa: E402

from lerobot2lance import convert_lerobot_to_lance  # noqa: E402


try:
    import lance  # noqa: F401
    import pyarrow  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "requires pyarrow + lance")
class ValidateBundleTest(unittest.TestCase):
    def _convert(self, target: Path) -> None:
        source = target.parent / f"lerobot_{target.name}"
        _write_v2_1_dataset(source, episodes=2, length=3)
        convert_lerobot_to_lance(
            source,
            target,
            output_layout="hf",
            dataset_id="rllab-postech/bg2-test",
        )

    def test_clean_bundle_validates(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)

            result = validate_bundle(target)
            self.assertTrue(
                result["ok"],
                f"clean bundle failed: {result['errors']}",
            )
            self.assertEqual(result["errors"], [])

    def test_missing_canonical_sidecar_fails(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)
            (target / "meta" / "tasks.jsonl").unlink()

            result = validate_bundle(target)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("tasks.jsonl" in e for e in result["errors"]),
                result["errors"],
            )

    def test_wrong_blob_encoding_fails(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)

            mfst_path = target / "manifest.json"
            mfst = json.loads(mfst_path.read_text())
            mfst["lance"]["blob_encoding"] = "lance.blob.v1"
            mfst_path.write_text(json.dumps(mfst))

            result = validate_bundle(target, check_blob_bytes=False)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("blob_encoding" in e for e in result["errors"]),
                result["errors"],
            )

    def test_v1_marker_gets_labeled_error(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)

            mfst_path = target / "manifest.json"
            mfst = json.loads(mfst_path.read_text())
            mfst["format"] = "rllab_published_lance_dataset_v1"
            mfst["schema_version"] = "1.0"
            mfst_path.write_text(json.dumps(mfst))

            result = validate_bundle(target, check_blob_bytes=False)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("v1 bundle detected" in e for e in result["errors"]),
                result["errors"],
            )

    def test_count_mismatch_fails(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)

            mfst_path = target / "manifest.json"
            mfst = json.loads(mfst_path.read_text())
            mfst["counts"]["frames"] = mfst["counts"]["frames"] + 99
            mfst_path.write_text(json.dumps(mfst))

            result = validate_bundle(target, check_blob_bytes=False)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("counts.frames" in e for e in result["errors"]),
                result["errors"],
            )

    def test_root_tasks_json_forbidden_in_v2(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)

            (target / "meta" / "tasks.json").write_text(json.dumps({"tasks": []}))

            result = validate_bundle(target, check_blob_bytes=False)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("tasks.json" in e for e in result["errors"]),
                result["errors"],
            )

    def test_root_stats_json_forbidden_in_v2(self) -> None:
        from scripts.validate_bundle import validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "published"
            self._convert(target)

            (target / "meta" / "stats.json").write_text(json.dumps({}))

            result = validate_bundle(target, check_blob_bytes=False)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("stats.json" in e for e in result["errors"]),
                result["errors"],
            )

    def test_validate_converted_root_reports_each_bundle(self) -> None:
        from scripts.validate_converted_root import validate_converted_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "converted"
            valid = root / "valid"
            invalid = root / "invalid"
            self._convert(valid)
            self._convert(invalid)
            (invalid / "meta" / "tasks.jsonl").unlink()

            status_path = Path(tmp) / "status.jsonl"
            summary_path = Path(tmp) / "summary.json"
            summary = validate_converted_root(
                root,
                status_path=status_path,
                summary_path=summary_path,
                check_blob_bytes=False,
            )

            self.assertEqual(summary["total_discovered"], 2)
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertTrue(status_path.exists())
            self.assertTrue(summary_path.exists())
            lines = [json.loads(line) for line in status_path.read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertTrue(any(row["relative_bundle"] == "valid" and row["ok"] for row in lines))
            self.assertTrue(any(row["relative_bundle"] == "invalid" and not row["ok"] for row in lines))


if __name__ == "__main__":
    unittest.main()
