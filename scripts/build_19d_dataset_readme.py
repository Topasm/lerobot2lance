#!/usr/bin/env python3
"""Build provenance docs for converted 19D BG2 datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted", default="data/converted_19d")
    parser.add_argument("--status", default="data/index/convert_19d_status.jsonl")
    parser.add_argument("--index", default="data/index/hf_robotis_19d.json")
    parser.add_argument("--json-out", default="data/index/converted_19d_sources.json")
    parser.add_argument("--readme-out", default="data/converted_19d/README.md")
    args = parser.parse_args()

    converted_root = Path(args.converted)
    status_path = Path(args.status)
    index_path = Path(args.index)
    json_out = Path(args.json_out)
    readme_out = Path(args.readme_out)

    index_rows = load_index(index_path)
    status_rows = load_jsonl(status_path)
    status_by_repo = latest_status_by_repo(status_rows)
    converted_rows = load_converted_manifests(converted_root, status_by_repo, index_rows)
    failed_rows = load_failed_rows(status_by_repo, converted_rows)

    summary = build_summary(converted_rows, failed_rows)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "converted_root": str(converted_root),
        "summary": summary,
        "converted": converted_rows,
        "not_converted": failed_rows,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme_out.parent.mkdir(parents=True, exist_ok=True)
    readme_out.write_text(render_readme(payload), encoding="utf-8")
    print(f"wrote {json_out}")
    print(f"wrote {readme_out}")
    return 0


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("repo_id")): row for row in rows if row.get("repo_id")}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_status_by_repo(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_repo: dict[str, dict[str, Any]] = {}
    for row in rows:
        repo_id = row.get("repo_id")
        if repo_id:
            by_repo[str(repo_id)] = row
    return by_repo


def load_converted_manifests(
    converted_root: Path,
    status_by_repo: dict[str, dict[str, Any]],
    index_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(converted_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        session = first_session(manifest_path.parent)
        info = read_json_if_exists(manifest_path.parent / "meta" / "info.json")
        repo_id = (
            session.get("source_dataset")
            or session.get("source_repo_id")
            or info.get("repo_id")
            or manifest.get("source_repo_id")
            or manifest.get("source_dataset")
            or manifest.get("dataset_id")
        )
        status = status_by_repo.get(str(repo_id), {})
        index_row = index_rows.get(str(repo_id), {})
        report = status.get("report", {})
        cameras = registry_cameras(manifest) or manifest.get("camera_keys") or report.get("cameras") or []
        counts = manifest.get("counts") or {}
        frames = counts.get("frames") or manifest.get("total_frames") or report.get("frames_written")
        episodes = counts.get("episodes") or manifest.get("total_episodes") or report.get("episodes_written")
        media = (
            counts.get("videos")
            or counts.get("media")
            or manifest.get("total_video_segments")
            or manifest.get("total_videos")
            or report.get("media_written")
        )
        fps = (manifest.get("rates") or {}).get("fps") or manifest.get("fps") or report.get("fps") or index_row.get("fps")
        row = {
            "dataset_id": manifest.get("dataset_id") or manifest_path.parent.name,
            "local_path": str(manifest_path.parent),
            "source_repo_id": repo_id,
            "source_url": hf_dataset_url(repo_id),
            "status": "converted",
            "robot_type": info.get("robot_type") or manifest.get("source_robot_type") or index_row.get("robot_type") or status.get("robot_type"),
            "robot_name": info.get("robot_name") or manifest.get("source_robot_name") or index_row.get("robot_name"),
            "pretrain_tier": manifest.get("pretrain_tier"),
            "episodes": as_int(episodes),
            "frames": as_int(frames),
            "media": as_int(media),
            "fps": fps,
            "action_dim": registry_dim(manifest, "actions", "action.body") or manifest.get("action_dim") or index_row.get("action_dim"),
            "state_dim": registry_dim(manifest, "modalities", "state.body") or manifest.get("state_dim") or index_row.get("state_dim"),
            "cameras": cameras,
            "quality_flag": quality_flag(repo_id),
        }
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("source_repo_id")), str(row.get("dataset_id"))))
    return rows


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def first_session(bundle: Path) -> dict[str, Any]:
    path = bundle / "meta" / "sessions.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list) and sessions and isinstance(sessions[0], dict):
            return sessions[0]
    return {}


def registry_cameras(manifest: dict[str, Any]) -> list[str]:
    cameras: list[str] = []
    for entry in (manifest.get("modalities") or {}).values():
        if isinstance(entry, dict) and entry.get("kind") == "video":
            key = entry.get("camera_key") or entry.get("source_key")
            if key:
                cameras.append(str(key))
    return sorted(cameras)


def registry_dim(manifest: dict[str, Any], section: str, name: str) -> int | None:
    entry = ((manifest.get(section) or {}).get(name) or {})
    shape = entry.get("shape") or []
    if isinstance(shape, list) and shape:
        return as_int(shape[0])
    return None


def load_failed_rows(
    status_by_repo: dict[str, dict[str, Any]],
    converted_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted_repos = {str(row.get("source_repo_id")) for row in converted_rows}
    rows = []
    for repo_id, status in sorted(status_by_repo.items()):
        if repo_id in converted_repos:
            continue
        if status.get("status") not in {"error", "locked"}:
            continue
        rows.append(
            {
                "source_repo_id": repo_id,
                "source_url": hf_dataset_url(repo_id),
                "status": status.get("status"),
                "robot_type": status.get("robot_type"),
                "frames": status.get("total_frames"),
                "error": status.get("error"),
                "quality_flag": quality_flag(repo_id),
            }
        )
    return rows


def build_summary(converted_rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    namespace_counts = Counter(str(row.get("source_repo_id", "")).split("/", 1)[0] for row in converted_rows)
    tier_counts = Counter(str(row.get("pretrain_tier")) for row in converted_rows)
    quality_counts = Counter(str(row.get("quality_flag")) for row in converted_rows)
    return {
        "converted_datasets": len(converted_rows),
        "not_converted": len(failed_rows),
        "converted_frames": sum(row.get("frames") or 0 for row in converted_rows),
        "converted_episodes": sum(row.get("episodes") or 0 for row in converted_rows),
        "converted_media": sum(row.get("media") or 0 for row in converted_rows),
        "namespaces": dict(sorted(namespace_counts.items())),
        "pretrain_tiers": dict(sorted(tier_counts.items())),
        "quality_flags": dict(sorted(quality_counts.items())),
    }


def render_readme(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    converted_rows = payload["converted"]
    failed_rows = payload["not_converted"]
    lines = [
        "# BG2 19D Converted Dataset Sources",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "This folder contains LeRobot datasets converted to the RLLAB published Lance layout.",
        "Each converted bundle keeps its original Hugging Face dataset id in `manifest.json` as `source_repo_id`.",
        "",
        "## Summary",
        "",
        f"- Converted datasets: `{summary['converted_datasets']}`",
        f"- Not converted / needs review: `{summary['not_converted']}`",
        f"- Converted episodes: `{summary['converted_episodes']}`",
        f"- Converted frames: `{summary['converted_frames']}`",
        f"- Converted media segments: `{summary['converted_media']}`",
        f"- Local root: `{payload['converted_root']}`",
        "",
        "Namespace counts:",
        "",
    ]
    lines.extend(f"- `{name}`: `{count}`" for name, count in summary["namespaces"].items())
    lines.extend(
        [
            "",
            "Quality flags are name-based hints only. `review_name` usually means the source repo name contains",
            "`test`, `upload`, `your-new-dataset`, or similar strings and should be reviewed before pretraining.",
            "",
            "## Converted Sources",
            "",
            "| Source dataset | Local folder | Tier | Robot type | Episodes | Frames | Media | FPS | Cameras | Quality |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in converted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_link(row.get("source_repo_id"), row.get("source_url")),
                    f"`{Path(str(row.get('local_path'))).name}`",
                    f"`{row.get('pretrain_tier') or ''}`",
                    f"`{row.get('robot_type') or ''}`",
                    str(row.get("episodes") or ""),
                    str(row.get("frames") or ""),
                    str(row.get("media") or ""),
                    str(row.get("fps") or ""),
                    br_join(row.get("cameras") or []),
                    f"`{row.get('quality_flag') or ''}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Not Converted / Review",
            "",
            "| Source dataset | Status | Robot type | Indexed frames | Reason | Quality |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    if failed_rows:
        for row in failed_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_link(row.get("source_repo_id"), row.get("source_url")),
                        f"`{row.get('status') or ''}`",
                        f"`{row.get('robot_type') or ''}`",
                        str(row.get("frames") or ""),
                        md_escape(short_reason(row.get("error"))),
                        f"`{row.get('quality_flag') or ''}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("|  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "```bash",
            "PYTHONPATH=. ./.venv/bin/python scripts/build_19d_dataset_readme.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def hf_dataset_url(repo_id: Any) -> str:
    if not repo_id:
        return ""
    return f"https://huggingface.co/datasets/{repo_id}"


def md_link(label: Any, url: Any) -> str:
    label_s = md_escape(str(label or ""))
    url_s = str(url or "")
    if not url_s:
        return label_s
    return f"[{label_s}]({url_s})"


def br_join(items: list[Any]) -> str:
    return "<br>".join(f"`{md_escape(str(item))}`" for item in items)


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def short_reason(error: Any) -> str:
    if not error:
        return ""
    text = str(error)
    if len(text) > 180:
        return text[:177] + "..."
    return text


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def quality_flag(repo_id: Any) -> str:
    name = str(repo_id or "").lower()
    review_tokens = (
        "test",
        "upload",
        "your-new-dataset",
        "with-token",
        "latched",
        "latch",
    )
    if any(token in name for token in review_tokens):
        return "review_name"
    return "ok"


if __name__ == "__main__":
    raise SystemExit(main())
