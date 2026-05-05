"""Command-line entry point for ``lerobot2lance``.

Usage:
    lerobot2lance --source /path/to/lerobot/dataset \\
                  --target /path/to/output/lance_bundle \\
                  [--overwrite] [--limit N] [--no-frames] [--no-video-blobs]
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--no-frames", action="store_true", help="Skip writing frames.lance")
    parser.add_argument(
        "--no-video-blobs",
        action="store_true",
        help="Omit video blob columns from episodes.lance (videos.lance still written)",
    )
    args = parser.parse_args()

    print(f"Source: {args.source}", flush=True)
    print(f"Target: {args.target}", flush=True)
    report = convert_lerobot_to_lance(
        Path(args.source),
        Path(args.target),
        overwrite=args.overwrite,
        limit=args.limit,
        include_frames=not args.no_frames,
        include_video_blobs=not args.no_video_blobs,
        progress_callback=_emit,
    )
    print("\nReport:", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
