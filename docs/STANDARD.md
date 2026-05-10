# RLLAB Published Lance Dataset — Standard

Format identifier: `rllab_published_lance_dataset_v1`
Current schema version: `1.0`
Status: stable contract; additive changes allowed within `v1` major.

This document fixes the layout, schemas, and design choices of the published
Lance dataset bundle produced by `lerobot2lance`, consumed by `rllab-training`
and `robo_dataview`, and re-emitted by `rllab-data-collection`. Anything not
listed here is implementation detail and may change. Anything listed here is
contract.

## 1. Charter (one line)

> **RLLAB Lance = LeRobot semantics + Lance storage/access upgrade.**

We do not redesign LeRobot's data model. We re-host it in Lance so we get
single-bundle distribution, lazy blob access, schema evolution, indexes, and
multi-modal headroom — without losing semantic compatibility with the LeRobot
ecosystem.

## 2. Directory layout

```text
dataset_root/
  manifest.json                # primary contract document
  README.md                    # dataset card
  meta/
    info.json                  # LeRobot info.json copy (features dict)
    sessions.json              # provenance per source dataset (HF layout)
    stats.json                 # state/action stats (LeRobot-compatible)
    stats/
      state_body.json          # per-modality stats
      action_body.json         # per-action stats
    tasks.json                 # task_index → instruction
    tasks.jsonl                # LeRobot-style task sidecar
    episodes.jsonl             # one episode per line (LeRobot-compatible)
    splits.json                # train/val/test episode lists
  data/
    episodes.lance             # primary trajectory table
    frames.lance               # per-frame index/QA
    videos.lance               # canonical media table with inline MP4 blobs
```

The `meta/` files marked LeRobot-compatible are intentionally identical-shape
to LeRobot's so that LeRobot tooling can read them. Bundles produced by
non-LeRobot sources (e.g. our own collection) emit the same shape regardless.

## 3. Manifest contract

Required top-level keys (`v1.0`):

| key | type | meaning |
|---|---|---|
| `format` | string | always `rllab_published_lance_dataset_v1` for HF layout |
| `schema_version` | string | `MAJOR.MINOR`. MAJOR-strict, MINOR-additive |
| `dataset_id` | string | stable identifier; HF repo path when published |
| `created_at` | ISO-8601 string | UTC conversion timestamp |
| `source` | string / object | path or repo of the LeRobot source |
| `source_dataset` | string | source repo id (`org/name`) |
| `source_format` | string | `lerobot_v2_1` or `lerobot_v3` |
| `primary_training_table` | string | manifest-selected training table. Default conversion uses `data/episodes.lance`; merged/pretrain bundles may use `data/train_episodes.lance` |
| `state_column` | string | episodes column for state (always `observation_state`) |
| `action_column` | string | episodes column for action (always `actions`) |
| `state_action_alignment` | object | see §6 |
| `camera_keys` | list[string] | LeRobot camera keys (e.g. `observation.images.cam_head`) |
| `camera_columns` | list[string] | normalized Lance column names (e.g. `observation_images_cam_head`) |
| `camera_key_to_column` | object | mapping between the two |
| `modalities` | object | see §7 (modality registry) |
| `actions` | object | see §7 (action registry) |
| `fps` | float | primary recording rate |
| `rates` | object | per-modality rates (`fps`, `modalities.{name}`, `actions.{name}`) |
| `tables` | object | logical name → relative path |
| `indexes` | object | `created` (actual) and `recommended` |
| `reader_hints` | object | see §8 |
| `capabilities` | object | feature flags readers can match against |
| `media_mode` | string | always `videos_table` |
| `camera_storage` | string | always `videos_table` |
| `blob_storage` | object | `episodes: "absent"`, `videos: "video_blob_column"` |
| `counts` | object | canonical counts: `episodes`, `frames`, `videos` |

Legacy/redundant keys (`total_episodes`, `total_frames`, `total_videos`,
`total_video_segments`) are emitted for one minor cycle and **deprecated** —
readers should use `counts` only.

## 4. Lance tables

### 4.1 `data/episodes.lance` (episode trajectory table)

Row unit: **one episode**.

| column | type | notes |
|---|---|---|
| `episode_index` | int64 (non-null) | join key everywhere |
| `task_index` | int64 | representative task |
| `fps` | float64 | per-episode rate (almost always equal to manifest `fps`) |
| `length` | int64 | number of frames in episode |
| `timestamps` | list&lt;float64&gt; | frame timestamps, length == `length` |
| `observation_state` | list&lt;list&lt;float32&gt;&gt; | trajectory; `[i]` aligns with `timestamps[i]` |
| `actions` | list&lt;list&lt;float32&gt;&gt; | trajectory; `[i]` aligns with `timestamps[i]` (no shift) |
| `language_instruction` | string | episode-level instruction text |
| `camera_segments` | list&lt;struct&gt; | camera key/column, `media_id`, timestamp range, frame range |
| `task_segments` | list&lt;struct&gt; | task index/instruction and frame/timestamp span |
| `trajectory_sha256` | string | hash of timestamps + state/action arrays |
| `{camera}_from_timestamp` | float64 | per-camera segment start (compat) |
| `{camera}_to_timestamp` | float64 | per-camera segment end (compat) |

**Forbidden:** `*_video_blob` columns. Episodes never carry inline MP4 bytes.

### 4.2 `data/frames.lance` (per-frame index / QA)

Row unit: **one frame**.

| column | type | notes |
|---|---|---|
| `episode_index` | int64 (non-null) | |
| `frame_index` | int64 (non-null) | within-episode index |
| `global_frame_index` | int64 | unique across dataset |
| `timestamp` | float64 | aligns with `episodes.timestamps[frame_index]` |
| `task_index` | int64 | |
| `observation_state` | list&lt;float32&gt; | duplicated from episodes for fast random access |
| `action` | list&lt;float32&gt; | duplicated from episodes for fast random access |
| `state_norm`, `action_norm` | float32 | optional pre-computed norms |
| `is_bad_frame` | bool (non-null) | quality flag |

State/action duplication is **intentional**. It mirrors the official Lance
LeRobot reference and lets training loaders do random-frame sampling without
joining episodes.lance. The cost is bounded.

### 4.3 `data/videos.lance` (canonical media table)

Row unit: **one (episode, camera) MP4**.

| column | type | notes |
|---|---|---|
| `media_id` | string | stable canonical id for video lookup; default converter uses `episode_{N:08d}_{camera_norm}` |
| `episode_id` | string | textual alias for `episode_index` |
| `episode_index` | int64 | |
| `camera_id` | string | normalized Lance camera id/column (e.g. `observation_images_cam_head`) |
| `camera_name` | string | full LeRobot key (`observation.images.cam_head`) |
| `media_type` | string | always `video` for now |
| `relative_path` | string | **canonical** path inside source bundle when available; nullable for media materialized from session blobs |
| `source_uri` | string | **canonical** provenance URI |
| `uri`, `video_path` | string | **deprecated aliases** (will be removed in v2.0) |
| `source_dataset_url`, `source_media_id`, `source_relative_path` | string | provenance |
| `video_blob` | large_binary + `lance-encoding:blob=true` | inline MP4 bytes |
| `from_timestamp`, `to_timestamp` | float64 | media segment range |
| `num_frames` | int64 | |
| `chunk_index`, `file_index` | int64 | LeRobot source sharding traceability |
| `sha256`, `byte_size` | string / int64 | media integrity |
| `width_pixels`, `height_pixels`, `fps`, `codec` | various | decoder hints |

The `video_blob` field carries the explicit `lance-encoding:blob = true`
metadata so Lance treats it as a lazy blob column. This is required for
metadata-only scans to be cheap on large bundles.

## 5. Sidecar files (`meta/`)

| file | purpose | source-of-truth | LeRobot compat |
|---|---|---|---|
| `info.json` | features dict, codec metadata, joint names | yes | identical to LeRobot |
| `sessions.json` | provenance per source dataset (HF only) | yes | RLLAB-specific |
| `stats.json` | normalization stats for `observation.state` and `action` | yes | LeRobot-shaped |
| `tasks.json` | `task_index → language_instruction` map | yes | RLLAB-specific (LeRobot uses `tasks.jsonl`) |
| `tasks.jsonl` | one task per line | yes | LeRobot-shaped |
| `episodes.jsonl` | one episode per line: `{episode_index, task_index, tasks, length, split}` | yes | LeRobot-shaped |
| `splits.json` | train/val/test episode lists | yes | RLLAB-specific |

**Joint names are not duplicated.** The standard does not maintain a separate
`robot.json`; readers that need joint names follow the `names_ref` JSON
pointer in the modality entry into `meta/info.json#/features/.../names`.

## 6. State/action alignment

```json
"state_action_alignment": {
  "type": "same_frame_timestamp",
  "episode_timestamp_column": "timestamps",
  "frame_timestamp_column":   "timestamp",
  "state_episode_column":     "observation_state",
  "action_episode_column":    "actions",
  "state_frame_column":       "observation_state",
  "action_frame_column":      "action",
  "index_rule": "observation_state[i] and actions[i] are aligned to timestamps[i]; the converter does not shift actions to the next state."
}
```

**Rule:** `observation_state[i]` and `actions[i]` are observed/commanded at
the same timestamp `timestamps[i]`. The converter does **not** shift actions
to the next state. Downstream policies that want the (s_t, a_t, s_{t+1})
shape must perform the shift themselves; this keeps the published data
unambiguous.

## 7. Modality registry (forward-compat)

`modalities` and `actions` keys describe the dataset as a registry so that
non-19-D / hand / tactile / depth datasets can be expressed without breaking
the schema. For the BG2 baseline these registries each contain one vector
entry plus N video entries.

```json
"modalities": {
  "state.body": {
    "kind": "state",
    "table": "episodes",
    "path": "data/episodes.lance",
    "column": "observation_state",
    "frame_table": "frames",
    "frame_path": "data/frames.lance",
    "frame_column": "observation_state",
    "names_ref": "meta/info.json#/features/observation.state/names",
    "shape": [19],
    "rate_hz": 30.0,
    "stats": "meta/stats/state_body.json"
  },
  "video.cam_head": {
    "kind": "video",
    "table": "videos",
    "path": "data/videos.lance",
    "camera_key": "observation.images.cam_head",
    "camera_column": "observation_images_cam_head",
    "media_id_column": "media_id",
    "blob_column": "video_blob",
    "segment_column": "camera_segments",
    "names_ref": "meta/info.json#/features/observation.images.cam_head/names",
    "shape_ref": "meta/info.json#/features/observation.images.cam_head/shape",
    "rate_hz": 30.0
  }
},
"actions": {
  "action.body": {
    "kind": "action",
    "table": "episodes",
    "column": "actions",
    "frame_table": "frames",
    "frame_column": "action",
    "names_ref": "meta/info.json#/features/action/names",
    "shape": [19],
    "rate_hz": 30.0,
    "stats": "meta/stats/action_body.json",
    "alignment": "same_frame_timestamp"
  }
}
```

The registry is a **shadow** in v1.x — readers may use it for forward compat
but must still tolerate the flat fields (`state_column`, `action_column`,
`camera_keys`). The flat fields will be removed in v2.0.

## 8. Reader hints

```json
"reader_hints": {
  "prefer_registry": true,
  "video_lookup": "videos.media_id",
  "normalization": "meta/stats.json",
  "per_modality_stats_dir": "meta/stats",
  "legacy_aliases_available": true,
  "lazy_blob_columns": { "data/videos.lance": ["video_blob"] },
  "fragment_strategy": {
    "data/episodes.lance": { "max_rows_per_file": 100000 },
    "data/frames.lance":   { "max_rows_per_file": 100000 },
    "data/videos.lance":   { "max_bytes_per_file": 2147483648, "max_rows_per_file": 4096 }
  }
}
```

Readers must:

- Prefer `media_id` for video lookup (joins are cheap with the index).
- Use the lazy blob path for `video_blob` (don't request the column unless
  fetching bytes).

## 9. Storage and indexing decisions (intent)

| decision | rationale |
|---|---|
| **Inline MP4 blobs** in `data/videos.lance` (not external files) | Single-bundle distribution; preserves Lance schema evolution; enables byte-range reads from `hf://`. External-file layouts forfeit all three. Matches the official `lance-format/lerobot-pusht-lance` reference. |
| **`lance-encoding:blob = true`** field metadata on `video_blob` | Makes Lance treat the column as a lazy blob: metadata scans skip bytes; `take_blobs()` fetches on demand. Without this, every projection touches video bytes. |
| **Episodes carry no `*_video_blob` columns** | The 1.0 dedup decision: video bytes live in exactly one canonical place (`videos.lance`). The official LeRobot Lance reference duplicates blobs into `episodes.lance` for convenience; we do not, to keep dataset size predictable when N cameras grow. |
| **Three-table layout (episodes / frames / videos)** | Matches the official Lance reference. `episodes` for trajectory loaders, `frames` for QA / random-frame sampling, `videos` for media. Each is independently scannable. |
| **state/action duplicated in frames.lance** | Frame-level random access without joining episodes. The official reference does the same. The cost is bounded; the alternative requires every random-access loader to do a hash join. |
| **`max_bytes_per_file = 2 GB` on `videos.lance`** | HF LFS object cap is ~50 GB; 2 GB gives parallel-download granularity and stays well under the cap for any single-shard pretrain merge. |
| **`max_rows_per_file = 100 000` on episodes/frames** | Bounded fragment growth on long pretrain merges; avoids regenerating one huge file when adding a single dataset. |
| **Scalar BTREE index on `episode_index`, `frame_index`, `task_index`, `camera_id`, `media_id`** | These are the join/filter keys that any reader (training loader, viewer, eval) hits. Index is created at write time so first-time readers don't pay for it. |
| **`task_segments` in episodes** | Preserves LeRobot v3-style intra-episode task transitions when present. `task_index` remains a representative compatibility alias. |
| **`camera_segments` list&lt;struct&gt; in episodes** | Avoids schema churn as camera count grows. Per-camera `*_from_timestamp`/`*_to_timestamp` columns remain compatibility aliases when present. |
| **No `meta/robot.json`** | LeRobot's `info.json.features.{...}.names` already encodes joint names. We point at it via `names_ref` instead of duplicating. `robot.json` is reserved for non-LeRobot sources. |
| **Single canonical `relative_path` + `source_uri`; `uri` and `video_path` are aliases** | Three URI-shaped columns confused readers. v1.0 fixes the canonical pair and keeps the aliases for one minor cycle to avoid breaking publish/training/viewer simultaneously. |

## 10. Versioning and migration

- `format` is the **major** contract. Reader strict-matches.
- `schema_version` is `MAJOR.MINOR`.
  - **MINOR** changes are **additive only**: new columns, new manifest keys,
    new sidecars. Older readers must continue to function.
  - **MAJOR** changes can remove or rename. They go through a deprecation
    cycle (one minor) before the alias is dropped.

Planned **v2.0** (major, future):

- Remove flat fields (`state_column`, `action_column`, `camera_keys`,
  `camera_columns`); modality registry becomes mandatory.
- Drop deprecated alias columns (`uri`, `video_path`).

A reverse converter (Lance → LeRobot v2.1/v3 directory layout) is on the
roadmap but is not part of the v1 contract.

## 11. Compatibility constants (for implementers)

```python
# lerobot2lance/converter.py
RLLAB_PUBLISHED_FORMAT  = "rllab_published_lance_dataset_v1"
RLLAB_PUBLISHED_LAYOUT  = "rllab_published_dataset_v1"
VIDEOS_MAX_BYTES_PER_FILE = 2 * 1024 * 1024 * 1024   # 2 GiB
VIDEOS_MAX_ROWS_PER_FILE  = 4096
TRAJ_MAX_ROWS_PER_FILE    = 100_000
SCALAR_INDEX_COLUMNS = {
    "episodes": ["episode_index", "task_index"],
    "frames":   ["episode_index", "frame_index", "task_index"],
    "videos":   ["episode_index", "camera_id", "media_id"],
}
```

These constants are part of the contract for **producers**. Readers must
not assume specific values; they read them from `manifest.reader_hints`.

## 12. Out of scope (deliberately)

The following are **not** part of this standard. They may be addressed in
separate documents or future major versions, but readers must not depend on
them:

- External MP4 file trees (Lance native ⇒ MP4 stays inline).
- Per-frame keyframe materialization (`keyframes.lance`) for ultra-high-throughput
  training. Optional, opt-in, not in the core layout.
- Encrypted blobs / access control.
- Streaming append semantics. Bundles are immutable per-version; new data
  becomes new HF revisions or new datasets.
