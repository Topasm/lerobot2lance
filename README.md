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

Hard dependencies: `pyarrow>=16.0`, `pylance>=0.18`. Upload support is optional:
install `.[hub]` when you want `--upload`.

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

To create the Hugging Face / RLLAB published layout directly:

```bash
lerobot2lance \
  --source data/lerobot/ffw_bg2_rev4_pick-place \
  --target data/published/bg2-grasp-v1 \
  --layout hf \
  --dataset-id bg2-grasp-v1 \
  --overwrite
```

To convert and upload in one command:

```bash
RLLAB_HF_NAMESPACE=rllab-postech \
lerobot2lance \
  --source data/lerobot/ffw_bg2_rev4_pick-place \
  --target data/published/bg2-grasp-v1 \
  --dataset-id bg2-grasp-v1 \
  --upload \
  --tag v0.1.0 \
  --overwrite
```

Useful flags:

| Flag | Effect |
|---|---|
| `--overwrite` | Replace any existing `*.lance` directories under `--target` |
| `--limit N` | Convert only the first N episodes (smoke testing) |
| `--layout session\|hf` | `session` keeps the flat local bundle; `hf` writes `data/*.lance` under an HF repo root |
| `--dataset-id ID` | Stable dataset id recorded in `manifest.json`; defaults to target dir name for HF layout |
| `--no-frames` | Skip writing `frames.lance` (saves disk if your trainer only reads `episodes.lance`) |
| `--no-video-blobs` | Omit per-camera video blobs from `episodes.lance`; `media.lance` / `data/videos.lance` still has the raw MP4 rows |
| `--upload --repo-id org/name` | Upload the HF-layout bundle to Hugging Face |
| `--tag v0.1.0` | Create an HF git tag after upload |

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

### Local session layout (default)

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

### Hugging Face published layout (`--layout hf` or `--upload`)

```text
target/
  manifest.json           # rllab_published_lance_dataset_v1 contract
  README.md               # dataset card
  meta/
    info.json             # copy of source LeRobot info.json
    sessions.json         # provenance for this converted source dataset
  data/
    episodes.lance        # primary training table
    frames.lance          # frame-level browsing/QA table
    videos.lance          # canonical source MP4 table with inline blobs
```

This is the layout expected by the newer RLLAB stack and Hugging Face dataset
repos. Version history should live in HF commits/branches/tags, not local
`v1/v2` folders.

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

The media table (`media.lance` in session layout, `data/videos.lance` in HF/published layout) includes `episode_index`, `camera_name` (original dotted LeRobot feature key), `media_type`, `uri`, `relative_path`, `video_blob`, `video_path`, `from_timestamp`, `to_timestamp`, `sha256`, `byte_size`, `num_frames`, `fps`, `width_pixels`, `height_pixels`, and `codec`.

`manifest.json` marks `episodes.lance` as the `primary_training_table`, records `training_row_unit="episode"`, `training_columns`, `camera_keys`, `camera_columns`, `fps`, `state_dim`, `action_dim`, and lists the available Lance tables. This lets `rllab-training`, `robo_dataview`, and the stack scripts share one contract without extra CLI flags.

## 한국어 사용법

### 설치

```bash
cd /home/shkim_rllab/Desktop/lerobot2lance
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 변환

```bash
lerobot2lance \
  --source /path/to/lerobot_dataset \
  --target /path/to/output_lance_session \
  --overwrite
```

변환 결과는 현재 RLLAB/Dataview Lance 구조에 맞는 raw training session입니다.

```text
output_lance_session/
  manifest.json
  episodes.lance/   # raw episode training table, row 하나 = episode 하나
  frames.lance/     # frame-level QA/search table
  media.lance/      # canonical media table
  meta/info.json
```

`episodes.lance`가 기본 학습 테이블입니다. Dataview에서 skill clip을 자르고 export하면, 그 curated export 쪽에서 `train_skill_clips.lance`가 생성됩니다. `lerobot2lance`는 LeRobot raw dataset을 raw Lance session으로 바꾸는 도구라서 `skills.lance`나 `train_skill_clips.lance`를 만들지 않습니다.

### HF 공개용 포맷과 업로드

RLLAB stack의 `data/published/<dataset_id>`와 같은 구조로 바로 만들려면:

```bash
lerobot2lance \
  --source /path/to/lerobot_dataset \
  --target /path/to/data/published/bg2-grasp-v1 \
  --layout hf \
  --dataset-id bg2-grasp-v1 \
  --overwrite
```

업로드까지 한 번에:

```bash
RLLAB_HF_NAMESPACE=rllab-postech \
lerobot2lance \
  --source /path/to/lerobot_dataset \
  --target /path/to/data/published/bg2-grasp-v1 \
  --dataset-id bg2-grasp-v1 \
  --upload \
  --tag v0.1.0 \
  --overwrite
```

`--upload`은 `huggingface_hub`가 필요합니다. 개발 환경에서는
`pip install -e ".[dev,hub]"`로 설치하면 됩니다.

### 확인

```bash
pytest -q
```

스택에서 사용할 때는 변환된 target 디렉터리를 그대로 Dataview나 training에 넘기면 됩니다.

```bash
./scripts/view.sh /path/to/output_lance_session
./scripts/train_policy.sh /path/to/output_lance_session
```

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

## AI Worker / BG2 19D Pretraining Bundle

After converting individual 19D LeRobot datasets into published Lance bundles
under `data/converted_19d`, merge them into one training bundle:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/build_pretrain_19d_lance.py \
  --converted-root data/converted_19d \
  --output data/pretrain_aiworker_19d \
  --dataset-id rllab-postech/pretraining-aiworker-bg2-19d \
  --overwrite
```

The merged bundle keeps the published layout:

```text
data/pretrain_aiworker_19d/
  manifest.json
  README.md
  data/episodes.lance
  data/train_episodes.lance
  data/frames.lance
  data/videos.lance
  meta/sessions.json
  meta/sources.json
```

`data/episodes.lance` is the published episode table with state/action arrays
and no video blob columns. `data/train_episodes.lance` is the rllab-training
table named by `manifest.json.primary_training_table`.

By default this is a light numeric/text pretraining bundle. It does not
duplicate MP4 bytes; `data/train_episodes.lance` has no `*_video_blob` columns
and `data/videos.lance` is a source media index with `video_blob = null` plus
`source_local_path`, `source_video_table`, `source_media_id`, and
`source_relative_path`. Rows are re-indexed into a single episode/frame space
and keep provenance columns such as `source_dataset`, `source_repo_id`,
`source_dataset_url`, `source_episode_index`, `source_robot_type`, and
`pretrain_tier`.

Source repos whose names look like scratch/test uploads are excluded by default;
pass `--include-review-names` to include them. To build only strict BG2
full-body data, add `--strict-bg2-only`.

Current `rllab-training` image policies require camera blobs in the selected
training table, and `robo_dataview` plays merged videos from `data/videos.lance`.
For a self-contained publish/training bundle, build the heavier variant:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/build_pretrain_19d_lance.py \
  --converted-root data/converted_19d \
  --output data/pretrain_aiworker_19d_train \
  --dataset-id rllab-postech/pretraining-aiworker-bg2-19d \
  --copy-video-blobs \
  --overwrite
```

`--copy-video-blobs` re-materializes MP4 bytes through Lance `take_blobs()`
instead of copying the `{position, size}` blob handles returned by a normal
Lance scan.

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
