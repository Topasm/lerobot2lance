# RLLAB Published Lance Dataset — Standard

Format identifier: `rllab_published_lance_dataset_v2`
Current schema version: `2.0`
Status: **v2.0 stable candidate**. This document is the source of truth for
the initial robust v2.0 implementation. It becomes stable after the converter,
pretrain builder, training loader, viewer, and publish tooling pass conformance
tests against it.

Implementation progress (what is shipped vs pending) is tracked in
[`V2_IMPLEMENTATION_CHECKLIST.md`](V2_IMPLEMENTATION_CHECKLIST.md). Promotion
from candidate to stable is gated on the conformance section of that file.

This document fixes the layout, schemas, and design choices of the published
Lance dataset bundle produced by `lerobot2lance`, consumed by `rllab-training`
and `robo_dataview`, and re-emitted by `rllab-data-collection`. Anything not
listed here is implementation detail and may change. Anything listed here is
contract.

## 1. Charter

> **RLLAB Lance = LeRobot semantics + Lance storage/access upgrade.**

We do not redesign LeRobot's data model. We re-host it in Lance so we get
single-bundle distribution, lazy blob access, schema evolution, indexes, and
multi-modal headroom — without losing semantic compatibility with the LeRobot
ecosystem.

The standard is a strict semantic superset of LeRobot where possible:

- LeRobot concepts (`info.json.features`, `tasks.jsonl`, `episodes.jsonl`,
  feature names, task transitions) remain the semantic source of truth.
- Lance provides the storage container, inline video blobs, indexing,
  byte-range access, and schema evolution path.
- Compatibility aliases are **not** kept in v2.0. New readers MUST consume the
  canonical registry (`modalities`, `actions`) and the canonical JSONL
  sidecars. Producers MUST NOT emit flat manifest aliases.

**Compatibility scope (important).** RLLAB Lance v2 is **LeRobot-semantic
compatible**, not byte-layout compatible with LeRobot v2.1 or v3. Conversion
preserves feature semantics, task metadata, episode boundaries, intra-episode
task transitions, and stats shape, but re-hosts storage into Lance tables.
A `source_format` of `lerobot_v3` indicates the *source* the bundle was
converted from, not that the published bundle is a drop-in replacement for
LeRobot v3's chunked Parquet/MP4 directory tree. Reverse export to native
LeRobot v2.1/v3 directory layout is outside the v2 contract.

RLLAB-collected bundles emit `source_format = "rllab_raw_rosbag_session_v1"`
when produced directly from a raw rosbag+MCAP session via
`rllab-data-collection/tools/publish_raw_dataset.py`. The published v2
bundle is the same shape regardless of which source pipeline produced it —
the discriminator is `source_format`, not the storage layout.

## 2. Directory layout

```text
dataset_root/
  manifest.json                # primary contract document
  README.md                    # dataset card
  meta/
    info.json                  # canonical LeRobot-style features dict
    sessions.json              # canonical provenance per source dataset/session
    tasks.jsonl                # canonical task_index -> instruction sidecar
    episodes.jsonl             # canonical one-episode-per-line sidecar
    splits.json                # canonical train/val/test episode lists
    stats/
      state_body.json          # canonical state.body normalization stats
      action_body.json         # canonical action.body normalization stats
  data/
    episodes.lance             # episode trajectory table
    frames.lance               # per-frame index/QA table
    videos.lance               # canonical media table with inline MP4 blobs
```

Canonical producers MUST emit only the canonical files listed above. v2.0
does not define compatibility sidecars. Root-level `meta/tasks.json` and
`meta/stats.json` are forbidden, and there is no `meta/compat/` escape hatch
in the canonical published layout.

`meta/robot.json` is not part of the LeRobot-source standard. It is reserved
for non-LeRobot sources that cannot provide `meta/info.json.features.*.names`.

## 3. Manifest contract

### 3.1 Canonical top-level keys

| key | type | meaning |
|---|---|---|
| `format` | string | `rllab_published_lance_dataset_v2` for HF/published layout |
| `schema_version` | string | `MAJOR.MINOR`; major is strict, minor is additive |
| `dataset_id` | string | stable identifier; HF repo path when published |
| `created_at` | ISO-8601 string | UTC conversion timestamp |
| `source` | string | source path/repo/provenance descriptor; object form is reserved and not valid in v2.0 |
| `source_format` | string | source layout. Known values: `lerobot_v2_1`, `lerobot_v3` (HF LeRobot tree input to `lerobot2lance`); `rllab_raw_rosbag_session_v1` (raw rosbag+MCAP session published directly by `rllab-data-collection/tools/publish_raw_dataset.py`); other RLLAB source formats may be added with the same string-tagged convention |
| `lance` | object | Lance file/storage contract; see §3.4 |
| `primary_training_table` | string | manifest-selected training table; default conversion uses `data/episodes.lance`, merged/pretrain bundles may use `data/train_episodes.lance` |
| `counts` | object | canonical counts: `episodes`, `frames`, `videos`, plus optional task/source counts |
| `tables` | object | logical table name -> relative path or table descriptor |
| `modalities` | object | canonical observation/input registry; see §7 |
| `actions` | object | canonical action/target registry; see §7 |
| `training_targets` | list[string] | optional ordered action registry keys that a default training reader should predict |
| `rates` | object | primary and per-modality/action rates |
| `state_action_alignment` | object | timestamp/index alignment contract; see §6 |
| `indexes` | object | `created` and `recommended` indexes |
| `reader_hints` | object | non-semantic read/write hints; see §8 |
| `capabilities` | object | feature flags readers can match against |
| `meta` | object | canonical sidecar paths |

v2.0 emits no flat manifest aliases beyond the keys above. Producers MUST NOT
emit `state_column`, `action_column`, `camera_keys`, `fps`, `state_dim`,
`action_dim`, `total_episodes`, `source_dataset`, or any other top-level alias
for data exposed by the registry / `tables` / `counts` keys.

### 3.2 Flat manifest aliases (removed in v2.0)

v2.0 emits **no** flat manifest aliases. Producers MUST emit only the
canonical top-level keys defined in §3.1, and MUST NOT emit any of the
following legacy v1 aliases:

`source_dataset`, `state_column`, `action_column`, `training_columns`,
`frame_columns`, `state_dim`, `action_dim`, `camera_keys`, `camera_columns`,
`camera_key_to_column`, `fps` (top-level), `media_mode`, `camera_storage`,
`blob_storage`, `total_episodes`, `total_frames`, `total_videos`,
`total_video_segments`, `meta.stats`, `meta.tasks`.

State/action column locations come from `manifest.modalities` and
`manifest.actions`; dimensions come from registry `shape`; cameras come from
`modalities.video.*`; rates come from `manifest.rates`; media storage comes
from `manifest.tables` / `manifest.lance` / `manifest.capabilities`; counts
come from `manifest.counts`.

Rules:

- A v2 reader MUST NOT consume flat aliases. If a manifest contains them, it
  is not v2.0-conformant.
- A v2 producer MUST NOT emit flat aliases.
- Readers that still need to load v1 bundles MUST use a separate v1 reader
  path (e.g. a `read_v1_bundle()` helper). Cross-format synthesis from
  aliases is not part of the v2 contract.

### 3.3 Capability flags

`capabilities` is an object of boolean feature flags. Missing means unknown or
unsupported. Unknown flags must be ignored by readers unless the reader was
explicitly configured to require that capability.

| flag | meaning |
|---|---|
| `modality_registry_v2` | manifest includes the §7 `modalities`/`actions` registry |
| `frames_table` | `data/frames.lance` exists |
| `videos_table` | `data/videos.lance` exists |
| `inline_video_blobs` | videos are stored inline in Lance, not as external files |
| `lance_blob_v2` | blob fields use Lance Blob v2 (`lance.blob_field` / `lance.blob_array`) |
| `camera_segments` | `episodes.lance.camera_segments` is present |
| `task_segments` | `episodes.lance.task_segments` is present |
| `trajectory_sha256` | `episodes.lance.trajectory_sha256` is present |
| `per_modality_stats` | `meta/stats/{modality}.json` files exist |
| `fixed_size_state_action` | state/action columns are FixedSizeList-based (mandatory for `shape_policy="single"`) |
| `action_semantics` | every `actions.*` entry carries the §7.1 `semantics` block |

### 3.4 Lance storage contract

Published v2.0 bundles use Lance Blob v2, not the legacy
`large_binary + lance-encoding:blob=true` metadata path.

```json
"lance": {
  "data_storage_version": "2.2",
  "blob_encoding": "lance.blob.v2",
  "published_blob_policy": "inline_bytes_only",
  "external_blob_uris_allowed": false,
  "requires_take_blobs": true
}
```

Writer requirements:

- Lance datasets containing blob columns MUST be written with
  `data_storage_version >= 2.2`.
- `video_blob` MUST be declared with `lance.blob_field("video_blob")`.
- `video_blob` values MUST be created with `lance.blob_array(...)`.
- Published bundles MUST use inline byte blobs only. External URI blob values
  and external URI slices are forbidden in published v2.0 bundles.
- Legacy `large_binary + lance-encoding:blob=true` blob columns are forbidden
  in v2.0 stable bundles.
- Published bundles SHOULD be compacted after table writes and before upload.
  The standard writer default rewrites small batches into larger Lance
  fragments, targets 4 GiB blob batches, recreates scalar indexes, and cleans
  pre-compaction table versions. If compaction is used, producers SHOULD record
  `reader_hints.fragment_strategy` with the target `max_bytes_per_file` for
  each Lance table.

## 4. Lance tables

Each table schema below is split into three classes. Producers must satisfy the
required classes; readers must tolerate optionals being absent.

| class | meaning |
|---|---|
| **Required non-null** | Producer MUST emit the column and value MUST NOT be null. Readers may assume presence and non-null. |
| **Required nullable** | Producer MUST emit the column. Value MAY be null when the source is missing the data. |
| **Optional** | Producer MAY emit. Reader MUST tolerate absence and treat null/missing identically. |

Within each table the column tables below mark this class in the "notes"
column when it is not obvious from context. Columns without a class marker
are required non-null.

### 4.1 `data/episodes.lance` (episode trajectory table)

Row unit: **one episode**.

| column | type | class | notes |
|---|---|---|---|
| `episode_index` | int64 | required non-null | join key everywhere |
| `task_index` | int64 | required nullable | representative task; null when LeRobot v3 source carries multi-task per episode |
| `fps` | float64 | required non-null | per-episode rate; usually equals manifest primary rate |
| `length` | int64 | required non-null | number of frames in episode |
| `timestamps` | list&lt;float64&gt; | required non-null | frame timestamps, length equals `length` |
| `observation_state` | large_list&lt;fixed_size_list&lt;float32, state_dim&gt;&gt; | required non-null | trajectory; `[i]` aligns with `timestamps[i]` |
| `actions` | large_list&lt;fixed_size_list&lt;float32, action_dim&gt;&gt; | required non-null | trajectory; `[i]` aligns with `timestamps[i]` |
| `language_instruction` | string | required nullable | representative episode-level instruction |
| `camera_segments` | list&lt;struct&gt; | required non-null | canonical media segment references; list may be empty, must not be null |
| `task_segments` | list&lt;struct&gt; | required non-null | canonical task spans, preserving LeRobot v3 transitions; half-open `[start, end_exclusive)` |
| `trajectory_sha256` | string | required non-null | SHA-256 of deterministic little-endian binary trajectory encoding (see §4.1.4) |
| `split` | string | required non-null | denormalized split label, one of `train`/`val`/`test`; mirrors `meta/splits.json` |
| `source_dataset` | string | required nullable | denormalized source dataset id; null when source is unknown |
| `source_episode_index` | int64 | required nullable | original episode index in the source dataset; equals `episode_index` for single-source converted bundles |
| `session_id` | string | required nullable | logical session/run id; defaults to `source_dataset` when not supplied |
| `embodiment_id` | string | required nullable | robot/embodiment identifier; null when not declared by the producer (see §1) |

`camera_segments` is the only canonical source for per-camera time ranges in
v2.0. The legacy `{camera}_from_timestamp` and `{camera}_to_timestamp` alias
columns are no longer emitted; if present in a bundle, the bundle is not
v2.0-conformant.

`camera_segments` struct fields:

```text
camera_key: string          # full LeRobot camera key
camera_column: string       # normalized Lance camera column/id
media_id: string            # stable logical media key referencing
                            # data/videos.lance.media_id;
                            # NOT a Lance row id, row index, or row address.
from_timestamp: float64
to_timestamp: float64
frame_start: int64
frame_count: int64
```

`task_segments` struct fields (half-open ranges):

```text
task_index: int64
language_instruction: string
start_frame: int64                  # inclusive
end_frame_exclusive: int64          # one past the last frame in the segment
start_timestamp: float64            # equals timestamps[start_frame]
end_timestamp_exclusive: float64    # equals timestamps[end_frame_exclusive] when in range
```

The half-open `[start, end_exclusive)` convention matches Python/Arrow/numpy
slicing and `frame_count = end_frame_exclusive - start_frame`. It is the same
convention used by `camera_segments` (`frame_start + frame_count`).

#### 4.1.4 `trajectory_sha256` deterministic recipe

`trajectory_sha256` is the lowercase hex digest of SHA-256 over the following
deterministic byte stream. Producers in any language must produce identical
bytes for identical input.

```text
magic                : 20 bytes, ASCII, b"RLLAB_TRAJECTORY_V1\0"
length               : int64 little-endian            # number of frames
state_dim            : int32 little-endian
action_dim           : int32 little-endian
timestamps           : float64 little-endian × length
observation_state    : float32 little-endian × length × state_dim   (row-major)
actions              : float32 little-endian × length × action_dim  (row-major)
```

`NaN` and `±Inf` are forbidden in the canonical trajectory arrays. JSON-based
hashing is **not** valid for v2.0; binary encoding is the only conformant form.

**Forbidden:** `*_video_blob` columns. Episodes never carry inline MP4 bytes.

### 4.2 `data/frames.lance` (per-frame index / QA)

Row unit: **one frame**.

| column | type | class | notes |
|---|---|---|---|
| `episode_index` | int64 | required non-null | |
| `frame_index` | int64 | required non-null | within-episode index |
| `global_frame_index` | int64 | required non-null | unique across dataset |
| `timestamp` | float64 | required non-null | aligns with `episodes.timestamps[frame_index]` |
| `task_index` | int64 | required nullable | representative task for the frame |
| `observation_state` | fixed_size_list&lt;float32, state_dim&gt; | required non-null | duplicated from episodes for fast random access |
| `action` | fixed_size_list&lt;float32, action_dim&gt; | required non-null | duplicated from episodes for fast random access |
| `state_norm` | float32 | optional | pre-computed norm; null means not computed |
| `action_norm` | float32 | optional | pre-computed norm; null means not computed |
| `is_bad_frame` | bool | required non-null | quality flag |
| `split` | string | required non-null | denormalized from parent episode; mirrors `episodes.split` |
| `source_dataset` | string | required nullable | denormalized from parent episode |
| `session_id` | string | required nullable | denormalized from parent episode |
| `embodiment_id` | string | required nullable | denormalized from parent episode |

State/action duplication is **intentional**. It lets training loaders,
viewers, and QA tools perform random-frame sampling without joining
`episodes.lance`. The storage cost is bounded relative to video.

`state_norm` and `action_norm` are optional QA/cache fields. Producers may
write nulls when norms were not computed; readers must not treat null as an
invalid frame.

### 4.3 `data/videos.lance` (canonical media table)

Row unit: **one (episode, camera) MP4**.

Core media fields:

| column | type | class | notes |
|---|---|---|---|
| `media_id` | string | required non-null | stable opaque id; new producers should use `episode_{N:08d}_{camera_id}` |
| `episode_index` | int64 | required non-null | join key to episodes/frames |
| `camera_id` | string | required non-null | normalized Lance camera id/column, e.g. `observation_images_cam_head` |
| `camera_name` | string | required non-null | full LeRobot key, e.g. `observation.images.cam_head` |
| `video_blob` | lance.blob.v2 | required non-null | inline MP4 bytes; metadata scans must not project this |
| `from_timestamp` | float64 | required non-null | segment start |
| `to_timestamp` | float64 | required nullable | segment end; null when `num_frames == 0` |
| `num_frames` | int64 | required non-null | decoded/video frame count |
| `sha256` | string | required non-null | media-byte integrity hash, equals `sha256(video_blob bytes)` |
| `byte_size` | int64 | required non-null | MP4 byte size, equals `len(video_blob)` |
| `width_pixels` | int64 | required non-null | video width |
| `height_pixels` | int64 | required non-null | video height |
| `fps` | float64 | required non-null | video rate |
| `codec` | string | required nullable | decoder hint; null when not advertised by source |

`media_id` is an opaque stable string. Readers must not parse the zero-padding
width or infer fields from it. Existing datasets that used
`episode_{N:06d}_{camera_id}` remain valid as long as the id is unique within
the dataset.

Provenance fields must be present as columns, though values may be null when a
source does not provide them:

| column | type | class | notes |
|---|---|---|---|
| `relative_path` | string | required nullable | canonical path inside source bundle when available |
| `source_uri` | string | required nullable | canonical source URI or original media URI |
| `source` | struct | required nullable | grouped provenance: `uri`, `repo_id`, `dataset_url`, `media_id`, `relative_path` |
| `source_dataset` | string | required nullable | denormalized source dataset id; mirrors parent episode |
| `source_episode_index` | int64 | required nullable | original episode index in the source dataset; mirrors parent episode |
| `source_dataset_url` | string | required nullable | source dataset URL, usually Hugging Face |
| `source_media_id` | string | required nullable | media id in the source bundle |
| `source_relative_path` | string | required nullable | relative path in the source bundle |
| `session_id` | string | required nullable | logical session/run id; mirrors parent episode |
| `embodiment_id` | string | required nullable | robot/embodiment identifier; mirrors parent episode |
| `chunk_index` | int64 | required nullable | LeRobot source chunk index |
| `file_index` | int64 | required nullable | LeRobot source file index |

`uri`, `video_path`, `episode_id`, and `media_type` were all removed before
v2.0 stable. They duplicated `relative_path` / `source_uri` / `episode_index`
or were implicit for this table, and confused readers. Existing v1 bundles
that emitted these columns must be re-converted from source; there is no
in-place migration tool.

The `video_blob` field must be a Lance Blob v2 column created with
`lance.blob_field` / `lance.blob_array` and written with
`data_storage_version >= 2.2`. Metadata scans must not project `video_blob`;
byte readers must use Lance blob access such as `take_blobs()`.

**Inline-only policy.** Published v2 bundles MUST contain inline byte blobs.
External-URI blob values and external URI slices, which Lance Blob v2 can also
represent, are **not** valid `video_blob` content in this contract; bundles are
expected to be self-contained.

### 4.4 Optional `data/train_episodes.lance`

Merged/pretrain bundles may include `data/train_episodes.lance` and point
`manifest.primary_training_table` at it. When present:

- `data/episodes.lance` still exists as the published episode table.
- `data/train_episodes.lance` must include the same trajectory contract as
  §4.1 (`timestamps`, `observation_state`, `actions`, `camera_segments`,
  `task_segments`, `trajectory_sha256`) and may add provenance/source columns.
- It must not contain `*_video_blob` columns.
- Training readers use `primary_training_table`; viewers may continue to use
  `data/episodes.lance` for published episode browsing.

## 5. Sidecar files (`meta/`)

| file | purpose | source-of-truth | LeRobot compat |
|---|---|---|---|
| `info.json` | features dict, codec metadata, joint/action names | yes | identical shape to LeRobot |
| `tasks.jsonl` | one task per line, `task_index -> language_instruction` | yes | LeRobot-shaped |
| `episodes.jsonl` | one episode per line with task/split/length metadata | yes | LeRobot-shaped |
| `splits.json` | train/val/test episode lists | yes | RLLAB-specific |
| `stats/state_body.json` | state.body normalization stats | yes | RLLAB per-modality |
| `stats/action_body.json` | action.body normalization stats | yes | RLLAB per-modality |
| `sessions.json` | provenance per source dataset/session | yes | RLLAB-specific |

Root-level `meta/tasks.json` and `meta/stats.json` are removed in v2.0 and
are not part of the canonical sidecar set. Canonical producers MUST NOT emit
compatibility copies elsewhere in the published bundle.

`tasks.jsonl` and `episodes.jsonl` are the canonical task/episode metadata.
Do not introduce a second disk source of truth for the same task data. If a
reader wants an aggregate task map, it should build it in memory.

**Joint names are not duplicated.** Readers that need joint/action names follow
the `names_ref` JSON pointer in the registry into
`meta/info.json#/features/.../names`. `meta/robot.json` is reserved for
non-LeRobot sources that do not have a usable `info.json.features` map.

## 6. State/action alignment

```json
"state_action_alignment": {
  "type": "same_frame_timestamp",
  "episode_timestamp_column": "timestamps",
  "frame_timestamp_column": "timestamp",
  "state_episode_column": "observation_state",
  "action_episode_column": "actions",
  "state_frame_column": "observation_state",
  "action_frame_column": "action",
  "index_rule": "observation_state[i] and actions[i] are aligned to timestamps[i]; the converter does not shift actions to the next state."
}
```

**Rule:** `observation_state[i]` and `actions[i]` are observed/commanded at
the same timestamp `timestamps[i]`. The converter does **not** shift actions
to the next state. Downstream policies that want `(s_t, a_t, s_{t+1})` must
perform the shift themselves.

## 7. Modality registry

`modalities` and `actions` are the canonical registry. They let non-19-D,
hand, tactile, depth, and future datasets add entries without changing the
core table contract. For the BG2 baseline the registry contains `state.body`,
`action.body`, and one `video.<camera>` entry per camera.

`state.body` and `action.body` are conventional names for the BG2/body-vector
baseline, not universal requirements. Datasets may publish action entries such
as `action.arm`, `action.ee_delta`, `action.gripper`, or task-specific targets
when those names better describe the command. Consumers MUST select the action
target from their configuration or `manifest.training_targets` when present;
they MUST NOT hard-code `action.body` unless they intentionally only support
BG2 body-vector bundles.

Valid `kind` values:

| kind | meaning |
|---|---|
| `state` | observation/proprioceptive vector or tensor |
| `action` | action/target vector or tensor |
| `video` | time-aligned encoded video stream |
| `depth` | depth image/sequence modality |
| `audio` | audio stream |
| `pointcloud` | point cloud modality |
| `tensor` | generic numeric tensor when no more specific kind applies |

Readers must gracefully skip unknown `kind` values unless the user requested
that modality as an input or target. If a requested modality has an unknown
kind, the reader must fail with a clear unsupported-modality error.

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
    "shape_policy": "single",
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
    "encoding": "rgb8_h264",
    "names_ref": "meta/info.json#/features/observation.images.cam_head/names",
    "shape_ref": "meta/info.json#/features/observation.images.cam_head/shape",
    "rate_hz": 30.0
  }
},
"actions": {
  "action.body": {
    "kind": "action",
    "table": "episodes",
    "path": "data/episodes.lance",
    "column": "actions",
    "frame_table": "frames",
    "frame_path": "data/frames.lance",
    "frame_column": "action",
    "names_ref": "meta/info.json#/features/action/names",
    "shape": [19],
    "shape_policy": "single",
    "rate_hz": 30.0,
    "stats": "meta/stats/action_body.json",
    "alignment": "same_frame_timestamp",
    "semantics": {
      "command_type": "joint_position",
      "absolute_or_delta": "absolute",
      "units": "rad",
      "control_frame": "robot_base",
      "applies_to_interval": "[t_i, t_{i+1})",
      "normalized": false
    }
  }
}
```

The `modalities` / `actions` registry is the only canonical source for this
data in v2.0. Legacy flat fields (`state_column`, `action_column`,
`camera_keys`, and friends) are not emitted; see §3.2.

`shape_policy` defines how vector/tensor dimensions are represented:

| value | meaning |
|---|---|
| `single` | one fixed shape for the whole dataset; use FixedSizeList-based columns |
| `fixed_padded_with_mask` | values are padded to a fixed shape and a mask modality/column declares valid dimensions |
| `variable_shape_with_explicit_shape_policy` | variable shape is intentional and the modality must document how to decode each row |

Producers MUST NOT silently mix 16-D arm-only and 19-D BG2 vectors into one
`single` fixed-shape column. Use separate bundles, an explicit padded+mask
policy, or a documented multi-embodiment policy.

#### 7.1 `action.*.semantics`

Each action entry MUST include a `semantics` object describing what the action
vector represents. Without this, downstream policy code cannot decide what
output head to attach. Conservative defaults (`"unknown"`) are valid when the
source dataset does not declare semantics; producers SHOULD upgrade to
specific values as soon as the source provides them.

| field | values | meaning |
|---|---|---|
| `command_type` | `joint_position`, `joint_velocity`, `ee_pose`, `ee_delta`, `gripper`, `mixed`, `unknown` | what the vector commands |
| `absolute_or_delta` | `absolute`, `delta`, `unknown` | whether the value is an absolute target or an increment |
| `units` | `rad`, `m`, `normalized`, `mixed`, `unknown` | physical unit of the values |
| `control_frame` | `robot_base`, `end_effector`, `world`, `unknown` | reference frame for spatial commands |
| `applies_to_interval` | string | interval the command is applied over; default `"[t_i, t_{i+1})"` |
| `normalized` | bool | whether values were renormalized in the conversion (separately from stats) |

Readers MUST treat unknown enum values as `"unknown"` and refuse to assume
semantics. Producers MUST NOT silently change a previously-published
`semantics` object without bumping the dataset version.

Video `encoding` is the logical decoder discriminator. If omitted in v1 data,
readers assume `rgb8_h264`. Producers introducing depth, IR, event, or
non-RGB camera data must set a specific encoding before relying on generic
video readers. Suggested values include:

| encoding | meaning |
|---|---|
| `rgb8_h264` | RGB 8-bit frames encoded as H.264 MP4 |
| `rgb8_hevc` | RGB 8-bit frames encoded as HEVC MP4 |
| `mono8_h264` | single-channel 8-bit video encoded as H.264 |
| `depth_uint16_mm_ffv1` | uint16 depth in millimeters encoded losslessly |
| `event_stream` | event-camera stream requiring a specialized decoder |

## 8. Reader hints

```json
"reader_hints": {
  "prefer_registry": true,
  "video_lookup": "videos.media_id",
  "normalization": "meta/stats",
  "lazy_blob_columns": { "data/videos.lance": ["video_blob"] },
  "blob_read_api": "take_blobs",
  "default_projections": {
    "frames_training": [
      "episode_index",
      "frame_index",
      "global_frame_index",
      "timestamp",
      "observation_state",
      "action",
      "task_index",
      "is_bad_frame"
    ],
    "videos_metadata": [
      "media_id",
      "episode_index",
      "camera_id",
      "camera_name",
      "from_timestamp",
      "to_timestamp",
      "num_frames",
      "sha256",
      "byte_size"
    ]
  },
  "fragment_strategy": {
    "data/episodes.lance": { "max_rows_per_file": 100000 },
    "data/frames.lance": { "max_rows_per_file": 100000 },
    "data/videos.lance": {
      "max_bytes_per_file": 2147483648,
      "max_rows_per_file": 4096
    }
  }
}
```

Reader requirements:

- Use `modalities` and `actions` exclusively; flat aliases are not part of v2.
- Use `meta/stats/{name}.json` for normalization. Root `meta/stats.json` is
  not emitted in v2.
- Use `tasks.jsonl` for task metadata. Root `meta/tasks.json` is not emitted
  in v2.
- Use `media_id` for video lookup.
- Do not project `video_blob` unless fetching bytes.
- Gracefully skip unknown capabilities, unknown registry kinds, and unsupported
  optional modalities unless explicitly requested.

## 9. Storage and indexing decisions

| decision | rationale |
|---|---|
| **Inline MP4 blobs** in `data/videos.lance` | Single-bundle distribution, Lance schema evolution, lazy access, and byte-range reads from `hf://`. |
| **Lance Blob v2 on `video_blob`** | Uses the current Lance blob API (`blob_field` / `blob_array`) with `data_storage_version >= 2.2`; metadata scans stay cheap and blob bytes are fetched on demand. |
| **Episodes carry no `*_video_blob` columns** | Video bytes live in exactly one canonical table, keeping size predictable as cameras grow. |
| **Three-table layout** | `episodes` for trajectories, `frames` for QA/random-frame access, `videos` for media. |
| **State/action duplicated in `frames.lance`** | Avoids joins for random-frame loaders and viewers. |
| **`camera_segments` in episodes** | Canonical media references survive camera/media schema growth; per-camera timestamp alias columns are removed in v2.0. |
| **`task_segments` in episodes** | Preserves LeRobot v3-style intra-episode task transitions; `task_index` remains a representative alias. |
| **`trajectory_sha256` in episodes** | Cheap trajectory integrity check across converter/merge/publish stages. |
| **Per-modality stats** | Lets training load only the stats it needs and avoids startup scans. |
| **No duplicate robot names file for LeRobot sources** | `info.json.features.*.names` already carries joint/action semantics; registry uses `names_ref`. |
| **2 GiB video fragments** | Chosen for parallel-download granularity and to bound the cost of regenerating a single fragment when a pretrain merge is updated. HF Hub permits much larger files (recommended chunk sizes up to ~200 GB, hard limit ~500 GB), so the 2 GiB cap is intentionally conservative, not an HF-imposed ceiling. |
| **Scalar indexes typed by cardinality** | High-cardinality join keys (`episode_index`, `frame_index`, `global_frame_index`, `media_id`) get `BTREE`. Low-cardinality categorical filters (`task_index`, `camera_id`, `is_bad_frame`, `split`, `source_dataset`, `session_id`, `embodiment_id`) get `BITMAP`. The earlier "BTREE on everything" plan was over-indexed for low-cardinality columns. |

## 10. Versioning and migration

v2.0 is the **current stable contract**. There is **no v1 backward
compatibility surface** in canonical readers.

- `format` is the contract discriminator. v2 readers strict-match
  `rllab_published_lance_dataset_v2`; bundles that advertise the v1 format
  string are out of contract and MUST be loaded through a separate v1 reader
  path if at all.
- `schema_version` is `MAJOR.MINOR`. Within v2.x:
  - Additive changes (new optional columns, new manifest keys, new
    sidecars, new capability flags) are v2.x-compatible. Older v2 readers
    must continue to function by ignoring unknown fields.
  - Removals and renames are NOT v2.x-compatible and require a v3.

Migration policy:

- Bundles converted before v2 (any `rllab_published_lance_dataset_v1`
  bundle, or any bundle still emitting flat aliases / deprecated
  table columns / root `meta/tasks.json` / root `meta/stats.json`) MUST
  be re-converted from source.
- No in-place v1 -> v2 migration tool is shipped. The conversion path
  is the source -> Lance converter.
- Reverse conversion to LeRobot v2.1/v3 directory layouts is useful for
  ecosystem compatibility but is not part of the v2.0 contract.

## 11. Conformance checklist

A bundle can be treated as v2.0 stable only if these checks pass:

1. `manifest.json` includes every §3.1 canonical key.
2. `manifest.capabilities` uses only documented v2.0 flags or clearly ignored
   extension flags.
3. `manifest.lance.data_storage_version >= "2.2"`,
   `manifest.lance.blob_encoding == "lance.blob.v2"`, and
   `manifest.lance.external_blob_uris_allowed == false`.
4. `data/videos.lance.video_blob` is a Lance Blob v2 field, can be fetched
   through Lance blob access, and contains no external URI blob values.
5. State/action columns are FixedSizeList-based for single-shape datasets:
   episode trajectories use `large_list<fixed_size_list<...>>`, frame rows use
   `fixed_size_list<...>`.
6. Every `episodes.lance.trajectory_sha256` matches the §4.1 deterministic
   binary hash recipe when recomputed.
7. Canonical sidecars exist: `meta/info.json`, `meta/tasks.jsonl`,
   `meta/episodes.jsonl`, `meta/splits.json`,
   `meta/sessions.json`, `meta/stats/state_body.json`, and
   `meta/stats/action_body.json`.
8. `manifest.indexes.created` lists any created indexes with column, type, and
   status; listed indexes must exist in Lance.
9. `primary_training_table` exists and contains the required trajectory
   columns from §4.1.

## 12. Compatibility constants

These values are reflected in `manifest.reader_hints` and
`manifest.indexes`. They are producer guidance, not reader assumptions.

```python
RLLAB_PUBLISHED_FORMAT = "rllab_published_lance_dataset_v2"
RLLAB_PUBLISHED_LAYOUT = "rllab_published_dataset_v2"
LANCE_DATA_STORAGE_VERSION = "2.2"
LANCE_BLOB_ENCODING = "lance.blob.v2"
PUBLISHED_BLOB_POLICY = "inline_bytes_only"
VIDEOS_MAX_BYTES_PER_FILE = 2 * 1024 * 1024 * 1024
VIDEOS_MAX_ROWS_PER_FILE = 4096
TRAJ_MAX_ROWS_PER_FILE = 100_000
SCALAR_INDEXES = {
    "episodes": [
        {"column": "episode_index", "type": "BTREE"},
        {"column": "task_index", "type": "BITMAP"},
        {"column": "split", "type": "BITMAP", "optional": True},
        {"column": "source_dataset", "type": "BITMAP", "optional": True},
        {"column": "session_id", "type": "BITMAP", "optional": True},
        {"column": "embodiment_id", "type": "BITMAP", "optional": True},
    ],
    "frames": [
        {"column": "global_frame_index", "type": "BTREE"},
        {"column": "episode_index", "type": "BTREE"},
        {"column": "frame_index", "type": "BTREE"},
        {"column": "task_index", "type": "BITMAP"},
        {"column": "split", "type": "BITMAP", "optional": True},
        {"column": "source_dataset", "type": "BITMAP", "optional": True},
        {"column": "session_id", "type": "BITMAP", "optional": True},
        {"column": "embodiment_id", "type": "BITMAP", "optional": True},
        {"column": "is_bad_frame", "type": "BITMAP"},
    ],
    "videos": [
        {"column": "media_id", "type": "BTREE"},
        {"column": "episode_index", "type": "BTREE"},
        {"column": "camera_id", "type": "BITMAP"},
        {"column": "source_dataset", "type": "BITMAP", "optional": True},
        {"column": "session_id", "type": "BITMAP", "optional": True},
        {"column": "embodiment_id", "type": "BITMAP", "optional": True},
    ],
}
```

## 13. Out of scope

The following are intentionally outside the core v2 standard:

- External MP4 file trees as primary storage.
- Per-frame keyframe materialization (`keyframes.lance`).
- Encrypted blobs or access-control policy.
- Streaming append semantics. Bundles are immutable per version; new data is
  published as new HF revisions or new datasets.
