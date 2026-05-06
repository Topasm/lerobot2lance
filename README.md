# lerobot2lance

Convert a local **[LeRobot](https://github.com/huggingface/lerobot)** v2.1 or v3 dataset into a **[Lance](https://lancedb.com/)** session bundle that downstream Lance-native tools can open directly.

The two LeRobot disk layouts are auto-detected:

- **v2.1** — single Parquet per episode under `data/chunk-*/episode_*.parquet`, single `meta/episodes.jsonl`
- **v3** — sharded `data/chunk-*/file-*.parquet` plus sharded `meta/episodes/chunk-*/file-*.parquet|jsonl`

Output is identical for both, so downstream code only ever has to deal with one shape.

## Install

```bash
pip install git+https://github.com/Topasm/lerobot2lance.git
# or, for local development:
git clone https://github.com/Topasm/lerobot2lance.git
cd lerobot2lance
pip install -e ".[dev]"
```

Hard dependencies: `pyarrow>=16.0`, `pylance>=0.18`. Nothing else.

## Quick Start

### CLI

```bash
# 1. Download a LeRobot dataset
hf download ubless607/ffw_bg2_rev4_pick-place \
  --repo-type dataset \
  --local-dir data/lerobot/ffw_bg2_rev4_pick-place

# 2. Convert to a Lance bundle
lerobot2lance \
  --source data/lerobot/ffw_bg2_rev4_pick-place \
  --target data/lance/ffw_bg2_rev4_pick-place \
  --overwrite
```

Useful flags:

| Flag | Effect |
|---|---|
| `--overwrite` | Replace any existing `*.lance` directories under `--target` |
| `--limit N` | Convert only the first N episodes (smoke testing) |
| `--no-frames` | Skip writing `frames.lance` (saves disk if your trainer only reads `episodes.lance`) |
| `--no-video-blobs` | Omit per-camera video blobs from `episodes.lance` (`media.lance` still has the raw MP4 rows) |

### Python API

```python
from pathlib import Path
from lerobot2lance import convert_lerobot_to_lance

report = convert_lerobot_to_lance(
    Path("data/lerobot/ffw_bg2_rev4_pick-place"),
    Path("data/lance/ffw_bg2_rev4_pick-place"),
    overwrite=True,
    progress_callback=lambda kind, payload: print(kind, payload),
)
print(report)
```

A `progress_callback(kind, payload)` is invoked once per episode with `kind="episode_converted"` and `{"episode_index", "completed", "total"}`.

## Output bundle

```text
target/
  manifest.json           # rllab_lance_session_v1 contract for viewers/trainers
  episodes.lance/        # one row per episode, includes per-camera *_video_blob columns
  frames.lance/          # one row per frame: episode_index, frame_index, timestamp,
                         # task_index, observation_state, action, QA norms/flags
  media.lance/           # canonical media index with sha256 + raw bytes blob
  meta/
    info.json            # copy of the source LeRobot info.json (used by viewers for codec metadata)
```

`episodes.lance` columns:

| Column | Type | Notes |
|---|---|---|
| `episode_index` | int64 | unique key |
| `task_index` | int64 | from per-frame `task_index` or `meta/tasks.jsonl` lookup |
| `fps` | float64 | from `meta/info.json` |
| `length` | int64 | frame count |
| `timestamps` | list&lt;float64&gt; | length T |
| `observation_state` | list&lt;list&lt;float32&gt;&gt; | shape (T, state_dim) |
| `actions` | list&lt;list&lt;float32&gt;&gt; | shape (T, action_dim) |
| `language_instruction` | string | from `meta/tasks.jsonl` lookup |
| `{camera_norm}_video_blob` | large_binary, blob-encoded | per-camera MP4 segment |
| `{camera_norm}_from_timestamp` | float64 | always `0.0` for whole-episode segments |
| `{camera_norm}_to_timestamp` | float64 | `(length - 1) / fps` |

Camera-name normalization: `observation.images.cam_head` → `observation_images_cam_head` (Lance column-name rules require underscore-only). The original dotted name stays in `meta/info.json` for viewer reference.

`frames.lance` also includes `global_frame_index`, `state_norm`, `action_norm`, and `is_bad_frame=false` so Robot Data Studio can run frame-level QA without recomputing basic statistics.

`media.lance` is the canonical media table. It includes `episode_index`, `camera_name` (original dotted LeRobot feature key), `camera_key`, `media_type`, `relative_path`, `sha256`, `byte_size`, `num_frames`, `fps`, `width_pixels`, `height_pixels`, `codec`, and `video_blob`.

`manifest.json` marks `episodes.lance` as the `primary_training_table`, records `training_columns`, `camera_keys`, `camera_columns`, `fps`, `state_dim`, `action_dim`, and lists the available Lance tables. This lets `rllab-training`, `robo_dataview`, and the stack scripts share one contract without extra CLI flags.

## Examples

### `ubless607/ffw_bg2_rev4_pick-place` (LeRobot v2.1)

```bash
hf download ubless607/ffw_bg2_rev4_pick-place --repo-type dataset --local-dir /tmp/ffw_src
lerobot2lance --source /tmp/ffw_src --target /tmp/ffw_lance --overwrite
```

Result (10 episodes, 3 cameras at 376×672 / 240×424):

```json
{
  "layout_detected": "v2_1",
  "episodes_written": 10,
  "frames_written": 3150,
  "media_written": 30,
  "fps": 30.0,
  "cameras": [
    "observation_images_cam_head",
    "observation_images_cam_wrist_left",
    "observation_images_cam_wrist_right"
  ]
}
```

### `lance-format/lerobot-xvla-soft-fold` (already in Lance)

This dataset ships pre-converted at `hf://datasets/lance-format/lerobot-xvla-soft-fold/data` — you can open it directly in any Lance-native viewer without `lerobot2lance`.

## Integrations

### [robo_dataview](https://github.com/Topasm/robo_dataview) (web viewer + REST API)

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/convert-lerobot \
  -H 'Content-Type: application/json' \
  -d '{"source":"/tmp/ffw_src","target":"/tmp/ffw_lance","overwrite":true}'
```

The endpoint imports `convert_lerobot_to_lance` from this package; it returns the same report and optionally opens the result as a registered dataset (`open_after: true`).

### [rllab-training](https://github.com/Topasm/rllab-training) (Diffusion Policy training)

After conversion, point any training config at the bundle root:

```yaml
dataset:
  root: /tmp/ffw_lance
```

`rllab_training.data.EpisodeDataset` reads the generated `manifest.json` and then uses `episodes.lance` as the primary training table — no further conversion step.

## Troubleshooting

- **`FileNotFoundError: ... meta/info.json`** — `--source` doesn't look like a LeRobot dataset root. Check the directory contains `meta/info.json`, `data/`, and `videos/`.
- **`FileNotFoundError: ... episodes`** — neither v3 sharded `meta/episodes/` nor v2.1 `meta/episodes.jsonl` was found. The dataset may use an unsupported layout; file an issue with the `info.json` snippet.
- **`FileExistsError`** — pass `--overwrite` to replace existing `*.lance` tables in the target directory.
- **Missing per-frame video metadata in `episodes.lance`** — the converter sets `from_timestamp=0.0` and `to_timestamp=(length-1)/fps` since the whole episode segment is embedded. Per-frame timestamps are in `frames.lance` (and the `timestamps` array on each episode row).
- **HF auth on `hf://` source paths** — this tool reads from local paths only. Use `huggingface-cli download` (or `hf download`) to materialize the dataset locally first.

## Development

```bash
git clone https://github.com/Topasm/lerobot2lance.git
cd lerobot2lance
pip install -e ".[dev]"
pytest -q
```

Tests are skipped when `pyarrow`/`lance` are unavailable so they degrade gracefully in light environments.

## License

MIT — see [LICENSE](LICENSE).
