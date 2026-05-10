# lerobot2lance

Convert a local **[LeRobot](https://github.com/huggingface/lerobot)** v2.1 or v3 dataset into a **[Lance](https://lancedb.com/)** session bundle that downstream Lance-native tools can open directly.

The two LeRobot disk layouts are auto-detected:

- **v2.1** — single Parquet per episode under `data/chunk-*/episode_*.parquet`, single `meta/episodes.jsonl`
- **v3** — sharded `data/chunk-*/file-*.parquet` plus sharded `meta/episodes/chunk-*/file-*.parquet|jsonl`

Output is identical for both, so downstream code only ever has to deal with one shape.

The published bundle format is fixed by [`docs/STANDARD.md`](docs/STANDARD.md) —
that document is the contract; the README below is a usage walkthrough.

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
| `--layout hf\|session` | `hf` is the standard published layout; `session` keeps a legacy flat local bundle |
| `--dataset-id ID` | Stable dataset id recorded in `manifest.json`; defaults to target dir name for HF layout |
| `--no-frames` | Skip writing `frames.lance` |
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

### Standard published layout (default, `--layout hf` or `--upload`)

```text
target/
  manifest.json           # rllab_published_lance_dataset_v1 contract
  README.md               # dataset card
  meta/
    info.json             # copy of the source LeRobot info.json (used by viewers for codec metadata)
    stats.json            # LeRobot-compatible observation.state/action normalization stats
    stats/
      state_body.json     # per-modality stats for manifest.modalities["state.body"]
      action_body.json    # per-action stats for manifest.actions["action.body"]
    tasks.json            # task_index -> language_instruction summary
    tasks.jsonl           # LeRobot-style canonical task sidecar
    episodes.jsonl        # lightweight episode metadata sidecar
    splits.json           # train/val/test episode lists
    sessions.json         # provenance for this converted source dataset
  data/
    episodes.lance        # primary trajectory table, no *_video_blob columns
    frames.lance          # frame-level browsing/QA table
    videos.lance          # canonical media table with inline MP4 blobs
```

This standard layout is expected by the newer RLLAB stack and Hugging Face
dataset repos. Version history should live in HF commits/branches/tags, not
local `v1/v2` folders.

### Legacy flat session layout (`--layout session`)

```text
target/
  manifest.json           # table paths, state/action alignment, media contract
  episodes.lance/         # primary trajectory table, no *_video_blob columns
  frames.lance/           # frame-level browsing/QA table
  media.lance/            # canonical media table with inline MP4 blobs
  meta/
    info.json             # copy of source LeRobot info.json
    stats.json            # LeRobot-compatible state/action normalization stats
    stats/
      state_body.json
      action_body.json
    tasks.json            # task_index -> language_instruction summary
    tasks.jsonl
    episodes.jsonl
    splits.json
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
| `camera_segments` | list&lt;struct&gt; | camera key/column, `media_id`, timestamp range, frame range |
| `task_segments` | list&lt;struct&gt; | task index/instruction and frame/timestamp span |
| `trajectory_sha256` | string | hash of timestamps + state/action arrays |
| `{camera_norm}_from_timestamp` | float64 | always `0.0` for whole-episode segments |
| `{camera_norm}_to_timestamp` | float64 | `(length - 1) / fps` |

Camera-name normalization: `observation.images.cam_head` → `observation_images_cam_head` (Lance column-name rules require underscore-only). The original dotted name stays in `meta/info.json` for viewer reference.

`observation_state` is the robot observation at timestamp `timestamps[i]`.
`actions` is the action/command vector aligned to the same timestamp index.
The converter does not shift actions to the next state; this is recorded in
`manifest.json.state_action_alignment.type = "same_frame_timestamp"`.

`frames.lance` also includes `global_frame_index`, `state_norm`, `action_norm`, and `is_bad_frame=false` so Robot Data Studio can run frame-level QA without recomputing basic statistics. Its frame-level columns are `observation_state` and singular `action`; these are the frame-row expansion of episode-level `observation_state` and `actions`.

The media table (`media.lance` in session layout, `data/videos.lance` in HF/published layout) is the only place video bytes are stored. It includes `episode_index`, `camera_name` (original dotted LeRobot feature key), `media_type`, `relative_path`, `source_uri`, `source_dataset_url`, `video_blob`, `from_timestamp`, `to_timestamp`, `sha256`, `byte_size`, `num_frames`, `fps`, `width_pixels`, `height_pixels`, and `codec`. `uri` and `video_path` are kept only as compatibility aliases; new readers should use `relative_path` and `source_uri`.

`manifest.json` names the `primary_training_table` (`episodes.lance` for a direct conversion, `train_episodes.lance` for a merged pretrain bundle), records `media_mode="videos_table"`, `training_columns`, `frame_columns`, `state_action_alignment`, `modalities`, `actions`, `camera_keys`, `camera_columns`, `camera_key_to_column`, `rates`, `capabilities`, `reader_hints`, `indexes`, `fps`, `state_dim`, `action_dim`, and lists the available Lance tables. The canonical row counts live under `counts.{episodes,frames,videos}`; older `total_*` fields remain as compatibility aliases. `rllab-training`, `robo_dataview`, and the stack scripts resolve camera MP4s through the media table, so there is no second training-only video format.

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

`rllab_training.data.EpisodeDataset` reads the generated `manifest.json`, uses `episodes.lance` for state/action, and resolves MP4s through `media.lance` or `data/videos.lance` — no second conversion step.

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
  meta/stats.json
  meta/stats/state_body.json
  meta/stats/action_body.json
  meta/tasks.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/splits.json
  meta/sessions.json
  meta/sources.json
```

`data/episodes.lance` is the published episode table with state/action arrays
and no video blob columns. `data/train_episodes.lance` is the rllab-training
table named by `manifest.json.primary_training_table`.

The pretrain bundle uses the same single media contract as every other new
RLLAB Lance dataset: `episodes.lance` and `train_episodes.lance` contain no
`*_video_blob` columns, while `data/videos.lance.video_blob` stores the MP4
bytes. Rows are re-indexed into a single episode/frame space and keep
provenance columns such as `source_dataset`, `source_repo_id`,
`source_dataset_url`, `source_episode_index`, `source_robot_type`, and
`pretrain_tier`. Media rows additionally keep `source_uri`, `source_local_path`,
`source_video_table`, `source_media_id`, and `source_relative_path`.

Source repos whose names look like scratch/test uploads are excluded by default;
pass `--include-review-names` to include them. To build only strict BG2
full-body data, add `--strict-bg2-only`.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/build_pretrain_19d_lance.py \
  --converted-root data/converted_19d \
  --output data/pretrain_aiworker_19d \
  --dataset-id rllab-postech/pretraining-aiworker-bg2-19d \
  --overwrite
```

## Troubleshooting

- **`FileNotFoundError: ... meta/info.json`** — `--source` doesn't look like a LeRobot dataset root. Check the directory contains `meta/info.json`, `data/`, and `videos/`.
- **`FileNotFoundError: ... episodes`** — neither v3 sharded `meta/episodes/` nor v2.1 `meta/episodes.jsonl` was found. The dataset may use an unsupported layout; file an issue with the `info.json` snippet.
- **`FileExistsError`** — pass `--overwrite` to replace existing `*.lance` tables in the target directory.
- **No `*_video_blob` columns in `episodes.lance`** — this is expected. Video bytes belong to `media.lance` / `data/videos.lance`; episode rows only keep the trajectory arrays and timestamp ranges.
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
