"""Command-line entry point for ``lerobot2lance``.

Usage:
    lerobot2lance --source /path/to/lerobot/dataset \\
                  --target /path/to/output/lance_bundle \\
                  [--overwrite] [--limit N] [--no-frames]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lerobot2lance import convert_lerobot_to_lance


def _emit(kind: str, payload: dict) -> None:
    if kind == "episode_converted":
        print(
            f"  episode {payload['episode_index']:>5}  "
            f"({payload['completed']}/{payload['total']})",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot v2.1 or v3 dataset into a Lance bundle."
    )
    parser.add_argument("--source", required=True, help="Path to the LeRobot dataset root")
    parser.add_argument("--target", required=True, help="Output directory for the Lance bundle")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Lance tables in --target")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N episodes")
    parser.add_argument(
        "--layout",
        choices=("session", "hf"),
        default="hf",
        help=(
            "Output layout. hf is the standard published layout with "
            "manifest/README/meta plus data/{episodes,frames,videos}.lance; "
            "session is a legacy flat local layout."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Stable dataset id written to manifest.json. Defaults to target dir name for --layout hf.",
    )
    parser.add_argument("--no-frames", action="store_true", help="Skip writing frames.lance")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After conversion, upload the HF-layout bundle to Hugging Face.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HF dataset repo id. Defaults to <RLLAB_HF_NAMESPACE>/<dataset_id> when possible.",
    )
    parser.add_argument("--private", action="store_true", help="Create/upload the HF repo as private")
    parser.add_argument("--public", action="store_true", help="Create/upload the HF repo as public")
    parser.add_argument("--revision", default=None, help="HF branch/revision target")
    parser.add_argument("--commit-message", default=None, help="HF upload commit message")
    parser.add_argument("--tag", default=None, help="Create an HF git tag after upload")
    parser.add_argument("--tag-message", default=None, help="HF git tag message")
    args = parser.parse_args()

    output_layout = "hf" if args.upload else args.layout
    dataset_id = args.dataset_id or (Path(args.target).name if output_layout == "hf" else None)
    print(f"Source: {args.source}", flush=True)
    print(f"Target: {args.target}", flush=True)
    print(f"Layout: {output_layout}", flush=True)
    report = convert_lerobot_to_lance(
        Path(args.source),
        Path(args.target),
        overwrite=args.overwrite,
        limit=args.limit,
        include_frames=not args.no_frames,
        output_layout=output_layout,
        dataset_id=dataset_id,
        progress_callback=_emit,
    )
    print("\nReport:", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.upload:
        from lerobot2lance.hub import upload_lance_bundle_to_hub

        repo_id = _repo_id(args.repo_id, dataset_id)
        private = True
        if args.public:
            private = False
        if args.private:
            private = True
        print(f"\nUploading to https://huggingface.co/datasets/{repo_id}", flush=True)
        commit_url = upload_lance_bundle_to_hub(
            Path(args.target),
            repo_id,
            private=private,
            revision=args.revision,
            commit_message=args.commit_message,
            tag=args.tag,
            tag_message=args.tag_message,
        )
        if commit_url:
            print(f"Commit: {commit_url}", flush=True)
        if args.tag:
            print(f"Tag: {args.tag}", flush=True)
    return 0


def _repo_id(explicit: str | None, dataset_id: str | None) -> str:
    if explicit:
        return explicit
    if dataset_id and "/" in dataset_id:
        return dataset_id
    namespace = (
        os.environ.get("RLLAB_HF_NAMESPACE")
        or os.environ.get("LEROBOT2LANCE_HF_NAMESPACE")
        or os.environ.get("HF_NAMESPACE")
    )
    if namespace and dataset_id:
        return f"{namespace}/{dataset_id}"
    raise SystemExit(
        "HF repo id is required for --upload. Pass --repo-id, set "
        "RLLAB_HF_NAMESPACE, or use --dataset-id namespace/name."
    )


if __name__ == "__main__":
    raise SystemExit(main())
