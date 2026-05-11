# RLLAB Lance v2.0 Implementation Checklist

v2.0 is the **current stable candidate** for the RLLAB published Lance bundle
contract. The format identifier is `rllab_published_lance_dataset_v2` and the
schema version is `2.0`. There is **no backward-compatibility path for v1
bundles** — any bundle that was emitted under `rllab_published_lance_dataset_v1`
or `schema_version` `1.x` must be re-converted from its source LeRobot dataset.
Readers may probe the format discriminator to refuse v1 input cleanly, but they
do not provide a translation layer.

This checklist is the working tracker for promoting the v2 contract to stable.
The full design rationale lives in [`STANDARD.md`](STANDARD.md) and the
deeper plan in [`checklist.md`](checklist.md).

## Status (in-repo)

**`lerobot2lance` converter + validator: complete for new-data creation.**
A fresh LeRobot dataset can be converted with
`lerobot2lance --layout hf --target <out>` and verified with
`scripts/validate_bundle.py <out>`; both produce a v2-clean bundle that
passes every conformance test in this repo (20 / 20, last run on 2026-05-11).

Legacy-bundle migration and the pretrain merge re-run that depended on it
are out of scope — the lab is creating fresh v2 bundles from current
source LeRobot data, not migrating old `data/converted_19d/*` artifacts.

Cross-repo status: `rllab-training` reads v2 bundles without v1 aliases,
exposes `task_segments` with the canonical
`[start_frame, end_frame_exclusive)` shape, refuses to train on unknown
action semantics, and ships a `verify_trajectory_sha256` recompute path
(plus a `DatasetConfig.verify_trajectory_hashes` opt-in to run it on every
episode at construction time). The `robo_dataview` backend opens v2
bundles, fetches videos via
`camera_segments[*].media_id -> videos.media_id -> take_blobs`, and now
surfaces `manifest.actions.action.body.semantics` through
`DatasetSummary.action_semantics`; the episode viewer renders a single-line
semantics badge under the action plot. The `rllab-data-collection` publish
path emits v2-clean published bundles from raw session data. A fresh ubless
BG2 v2 probe validates, loads in `rllab-training`, completes a 2-step
optimizer smoke run, and reloads through `rllab-infer` with a 19-D action
prediction. The only cross-repo follow-on left is the optional
`media_id + sha256` decoder cache key (B2.4) and the HF publish / remote-read
smoke tasks in §D.

Legend:

```text
[x]   shipped in lerobot2lance and verified locally
[~]   partially shipped; remaining work is called out inline
[ ]   still to do
[def] explicitly deferred from v2.0 (will not gate stable promotion)
```

---

## Snapshot

| Section | Scope | Done | Partial | Open | Deferred |
| --- | --- | --- | --- | --- | --- |
| A | `lerobot2lance` v2 contract | 12 | 0 | 0 | 0 |
| B | Cross-repo consumers (`rllab-training`, `robo_dataview`, `rllab-data-collection`) | 20 | 1 | 0 | 0 |
| C | Conformance / validator suite | 7 | 0 | 0 | 0 |
| D | Operational data tasks (smoke tests, publish) | 4 | 0 | 4 | 0 |
| E | Open questions | 0 | 0 | 1 | 0 |

---

## A. `lerobot2lance` v2 contract

The converter and the published manifest schema have been bumped to v2 and the
v1-era compatibility surfaces have been dropped. Only the validator strict-mode
catalog and the cross-repo `unknown` action-semantics upgrade remain.

* [x] **A1.** Lance Blob v2 only: `videos.video_blob` is `lance.blob.v2`,
  `data_storage_version >= "2.2"`, `external_blob_uris_allowed = false`,
  `blob_read_api = "take_blobs"`. Legacy `large_binary + lance-encoding:blob=true`
  emission is gone.
* [x] **A2.** `frames.observation_state` / `frames.action` are
  `fixed_size_list<float32, dim>`; `episodes.observation_state` /
  `episodes.actions` are `large_list<fixed_size_list<float32, dim>>`. No ragged
  `list<float32>` fallback.
* [x] **A3.** `trajectory_sha256` is the deterministic little-endian binary
  digest with magic prefix and `length / state_dim / action_dim` headers; NaN
  and Inf are rejected at hash time.
* [x] **A4.** `task_segments` use half-open `[start_frame, end_frame_exclusive)`
  and `[from_timestamp, to_timestamp)` with no inclusive `end_frame` legacy
  field.
* [x] **A5.** Typed scalar indexes are recorded under `manifest.indexes.created`
  with explicit `BTREE` / `BITMAP` types; `frames.global_frame_index` BTREE is
  created by default.
* [x] **A6.** Format and schema bumped: `RLLAB_PUBLISHED_FORMAT =
  "rllab_published_lance_dataset_v2"` and `schema_version = "2.0"` are the only
  values the converter emits for HF-published bundles.
* [x] **A7.** Flat manifest aliases are no longer emitted. The published
  manifest produced by `lerobot2lance/converter.py::_write_hf_manifest_json`
  contains only `format`, `schema_version`, `dataset_id`, `created_at`,
  `source`, `source_format`, `lance`, `counts`, `tables`, `modalities`,
  `actions`, `rates`, `state_action_alignment`, `indexes`, `reader_hints`,
  `capabilities`, `meta`, and `primary_training_table`. None of `state_column`,
  `action_column`, `training_columns`, `frame_columns`, `state_dim`,
  `action_dim`, `camera_keys`, `camera_columns`, `camera_key_to_column`, top-
  level `fps`, `media_mode`, `camera_storage`, `blob_storage`, `total_*`,
  `meta.stats`, or `meta.tasks` appear. The `scripts/build_pretrain_19d_lance.py`
  merge path emits the same canonical manifest.
* [x] **A8.** Deprecated table alias columns removed from the published
  schema: `videos.lance` no longer carries `episode_id`, `media_type`, `uri`,
  or `video_path`, and `episodes.lance` no longer emits per-camera
  `{camera}_from_timestamp` / `{camera}_to_timestamp` aliases. `camera_segments`
  is the only canonical per-camera timestamp source.
* [x] **A9.** Compatibility sidecars removed. The converter no longer writes
  `meta/tasks.json` or `meta/stats.json`; canonical sidecars are
  `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`,
  `meta/splits.json`, `meta/sessions.json`, and the per-modality stats files
  under `meta/stats/`.
* [x] **A10.** `manifest.capabilities` advertises the v2 mandatory flags:
  `modality_registry_v2 = true`, `lance_blob_v2 = true`,
  `fixed_size_state_action = true`, `action_semantics = true`. The legacy
  `modality_registry_v1` flag is no longer emitted.
* [x] **A11.** Validator strict-v2 mode in
  `scripts/validate_bundle.py`. The validator rejects v1 capability flags, the
  legacy alias key set, root `meta/tasks.json` / `meta/stats.json`, deprecated
  `videos.lance` columns, and v1 format/schema markers with an explicit
  re-conversion diagnostic. Covered by `tests/test_validate_bundle.py`.
* [x] **A12.** Producers upgrade the selected action target's
  `semantics.command_type` from `"unknown"` to a specific value for the known
  BG2/FFW paths. `lerobot2lance` infers concrete joint-position semantics for
  ROBOTIS/FFW-family 16-D/19-D/25-D LeRobot sources, the 19-D pretrain builder
  emits the same concrete `action.body` contract, and unknown non-FFW sources
  remain intentionally non-training-compatible until explicitly annotated.

---

## B. Cross-repo consumers

The contract is finalized inside `lerobot2lance`; v2.0 only ships once the
downstream consumers can read a freshly converted v2 bundle without falling
back to v1 aliases.

### B1. `rllab-training`

* [x] **B1.1.** Recognize `format == "rllab_published_lance_dataset_v2"` as the
  primary discriminator and refuse v1 bundles with a clear error.
* [x] **B1.2.** Resolve state / action / video columns via
  `manifest.modalities` and `manifest.actions` only. No flat-alias fallback.
* [x] **B1.3.** Read normalization stats only from `meta/stats/*.json`; remove
  any `meta/stats.json` aggregate fallback.
* [x] **B1.4.** Honor `task_segments.end_frame_exclusive` everywhere range math
  is performed. `EpisodeDataset.task_segments(episode_index)` exposes the
  half-open structs verbatim and the v2 test suite asserts
  `end_frame_exclusive - start_frame == frame_count` and that the inclusive
  legacy `end_frame` field is not present. Covered by
  `test_episode_dataset_exposes_task_segments_half_open`.
* [x] **B1.5.** Tolerate / verify the binary `trajectory_sha256` shape.
  `EpisodeDataset` mirrors the v2 magic-prefixed little-endian recipe in
  `_compute_trajectory_sha256`, exposes `trajectory_sha256(episode_index)` /
  `verify_trajectory_sha256(episode_index)`, and adds a
  `DatasetConfig.verify_trajectory_hashes` opt-in that re-derives every
  episode digest at construction time. Tamper paths fail with a clear
  mismatch error. Covered by three round-trip / tamper / config tests in
  `test_published_layout.py`.
* [x] **B1.6.** Fetch video bytes via `videos.media_id` + Lance Blob v2
  `take_blobs`; never project `video_blob` in metadata scans.
* [x] **B1.7.** Use the selected training action target's `semantics` block to
  configure the action contract, and refuse to silently train when required
  semantics are missing or `unknown`. `EpisodeDataset` resolves the target from
  `manifest.training_targets` when present, otherwise from the single entry in
  `manifest.actions`; it does not require the registry key to be
  `action.body`. It validates `command_type`, `absolute_or_delta`, `units`,
  `control_frame`, `applies_to_interval`, and `normalized`, exposes
  `dataset.action_target` / `dataset.action_semantics`, and includes them in
  `describe()`. Covered by `tests/test_published_layout.py`.

### B2. `robo_dataview`

(Deprioritized for ship-blocking, but tracked here so v2.0 promotion does not
silently break the viewer.)

* [x] **B2.1.** Recognize the v2 format identifier.
* [x] **B2.2.** Discover cameras from `manifest.modalities.video.*`; stop
  reading the `camera_keys` / `camera_columns` flat aliases.
* [x] **B2.3.** Resolve `camera_segments[*].media_id` against `videos.media_id`
  for video lookup.
* [~] **B2.4.** Use `media_id + sha256` as the decoder cache key. Backend
  lookup now uses stable `media_id`; no persistent viewer decoder cache is
  currently maintained in `lance_store.py`, so `sha256` cache-key integration
  remains for the frontend / Rerun cache path if a decoder cache is added.
* [x] **B2.5.** Fetch video bytes via Blob v2 `take_blobs`; do not reach for
  the removed `uri` / `video_path` columns.
* [x] **B2.6.** Render `task_segments` as half-open ranges in the UI.
  Backend `EpisodeDetail` includes typed `task_segments`; the web parser maps
  them to `Episode.taskSegments`; `EpisodeViewer` renders a clickable
  half-open segment strip and active task label. Covered by the v2
  `test_lance_store` media-id regression for API exposure.
* [x] **B2.7.** Surface action semantics (joint vs. EE pose, delta vs.
  absolute, units, normalization) to the user. Backend
  `_action_semantics_from_manifest` lifts
  `manifest.actions.action.body.semantics` into a typed `ActionSemantics`
  block on `DatasetSummary`; the web `toDatasetSummary` parser maps it to
  `actionSemantics`; `EpisodeViewer` accepts an optional `actionSemantics`
  prop and renders a single-line `summarizeActionSemantics` summary
  (command_type · absolute/delta · units · control_frame · normalized)
  under the action plot, threaded through both `browse-mode` and
  `annotation-mode`. Covered by `test_v2_dataset_summary_surfaces_action_semantics`.

### B3. `rllab-data-collection`

* [x] **B3.1.** Publish path emits `rllab_published_lance_dataset_v2` /
  `schema_version = "2.0"` from `tools/publish_dataset.py`.
* [x] **B3.2.** Manifest includes `lance`, `modalities`, `actions`, `rates`,
  `capabilities`, and `reader_hints` blocks in the same registry-first shape as
  `lerobot2lance`.
* [x] **B3.3.** Action semantics emitted with concrete values
  (`command_type`, `absolute_or_delta`, `units`, `control_frame`,
  `applies_to_interval`, `normalized`); no `"unknown"` placeholders for
  RLLAB-collected data. Current publish semantics are joint-position,
  absolute, mixed-unit, robot-base, unnormalized actions.
* [x] **B3.4.** Video published as Lance Blob v2 inline bytes using
  `lance.blob_field` / `lance.blob_array`; publish rejects non-inline blob
  values before writing.
* [x] **B3.5.** State / action emitted with the FixedSizeList schema
  documented in A2.
* [x] **B3.6.** Denormalized columns `split`, `source_dataset`, `session_id`,
  `embodiment_id` populated on `episodes.lance` and `frames.lance`.
* [x] **B3.7.** Sidecars limited to canonical files: `meta/info.json`,
  `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/splits.json`,
  `meta/sessions.json`, and `meta/stats/{name}.json`. Root
  `meta/tasks.json` / `meta/stats.json` are not emitted.

  Verification: `tests/test_publish_dataset.py` passes under the
  `rllab-training` venv (`3 passed`), and a synthetic published bundle produced
  by `tools/publish_dataset.py` passes
  `lerobot2lance/scripts/validate_bundle.py`.

---

## C. Conformance / validator suite

* [x] **C1.** `scripts/validate_bundle.py` rejects bundles that carry forbidden
  flat-alias manifest keys, `modality_registry_v1`, legacy `reader_hints`
  fields, root `meta/tasks.json` / `meta/stats.json`, and the deprecated
  `videos.lance` legacy columns. Verified locally by `tests/test_validate_bundle.py`.
* [x] **C2.** Manifest must expose `format ==
  "rllab_published_lance_dataset_v2"`, `schema_version` starting with `2.`,
  `lance.blob_encoding == "lance.blob.v2"`,
  `lance.data_storage_version >= "2.2"`, `lance.external_blob_uris_allowed ==
  false`, `capabilities.lance_blob_v2 == true`, and
  `capabilities.action_semantics == true`. Validator covers this.
* [x] **C3.** `episodes.lance` / `frames.lance` / `videos.lance` schemas:
  state and action are FixedSizeList for `shape_policy="single"`; required
  non-null columns are populated; `videos.media_id` is unique;
  `camera_segments[*].media_id` resolves to `videos.media_id`;
  `sha256(video_blob bytes) == videos.sha256`;
  `len(video_blob bytes) == videos.byte_size`. Covered by
  `tests/test_converter.py` and `tests/test_validate_bundle.py`.
* [x] **C4.** Frame ↔ episode alignment: `counts.frames ==
  sum(episodes.length)`; `frames.timestamp == episodes.timestamps[frame_index]`;
  `frames.observation_state == episodes.observation_state[frame_index]`;
  `frames.action == episodes.actions[frame_index]`; converter does not shift
  actions. Covered by converter tests.
* [x] **C5.** Index entries declared under `manifest.indexes.created` exist
  on the actual datasets (BTREE / BITMAP types match), and stale entries are
  rejected. Covered by validator + converter tests.
* [x] **C6.** Explicit hard-fail with a labeled error when a bundle still
  carries v1 markers (e.g. legacy `videos.lance` alias columns reintroduced
  by a downstream tool). Covered by `test_v1_marker_gets_labeled_error`.
* [x] **C7.** Hard-fail when root-level `meta/tasks.json` or `meta/stats.json`
  reappears, even if the rest of the bundle is otherwise v2-clean. Covered by
  `test_root_tasks_json_forbidden_in_v2` and `test_root_stats_json_forbidden_in_v2`.

---

## D. Operational data tasks

Forward-looking smoke / publish tasks for fresh v2 bundles. Legacy-bundle
re-conversion and the pretrain merge re-run that depended on it are out
of scope — the lab starts from current source data.

* [x] **D1.** Run `scripts/validate_bundle.py` on each freshly converted
  v2 bundle and require zero errors. Verified on
  `data/probes/ubless_v2_final`.
* [x] **D2.** Run `scripts/validate_converted_root.py` over the converted
  root and require zero errors per bundle. Verified over `data/probes`
  (`ubless_v2_final`, `ubless_v2_verify`: 2 passed, 0 failed).
* [x] **D3.** Sample `rllab-training` smoke run on a fresh v2 bundle.
  `rllab-inspect-dataset` reports 10 episodes / 3150 frames, 19-D state/action,
  3 cameras, `action_target = action.body`, and concrete joint-position
  semantics. A 2-step optimizer smoke run completed
  (`step=1 loss=1.122010`, `step=2 loss=1.123513`, `val_loss=1.113266`) and
  `rllab-infer` reloaded the checkpoint and produced a 19-D action.
* [x] **D4.** Sample `robo_dataview` backend smoke run on a converted bundle:
  ubless v2 probe opens, summarizes 10 episodes / 3150 frames, lists 3 cameras,
  returns 322-frame state/action timeseries for episode 0, and fetches the
  `cam_head` MP4 blob via Blob v2.
* [ ] **D5.** HF publish dry-run for one converted bundle.
* [ ] **D6.** HF published-bundle remote read test (manifest + sidecars +
  schema).
* [ ] **D7.** Remote video blob `take_blobs` round-trip from the published
  bundle.
* [ ] **D8.** Confirm that remote metadata scans never project `video_blob`
  (network-traffic check).

---

## E. Open questions

* [ ] **E1.** Cross-repo manifest tolerance window. Decide whether
  `rllab-training` and `rllab-data-collection` keep a transient grace mode
  that accepts manifests where one of the new mandatory blocks
  (`modalities`, `actions`, `capabilities.fixed_size_state_action`, action
  semantics) is missing or `unknown`, and for how long. This is the gate
  before B1 / B3 ship — the longer the window, the more v1 contamination
  bleeds back into v2 datasets.

---

## Definition of done

A bundle is RLLAB Lance v2.0 stable when:

1. `format == "rllab_published_lance_dataset_v2"` and `schema_version`
   starts with `2.`.
2. Lance Blob v2 only; no legacy blob metadata anywhere.
3. Registry-only manifest: `modalities` and `actions` are mandatory and no
   flat aliases survive.
4. Canonical sidecars only: no root `meta/tasks.json` / `meta/stats.json`.
5. No deprecated table-alias columns on `videos.lance` or `episodes.lance`.
6. FixedSizeList state / action for `shape_policy="single"` datasets.
7. Binary `trajectory_sha256` with NaN/Inf rejection.
8. Half-open `task_segments`.
9. Logical `media_id`-based video lookup.
10. Typed scalar indexes recorded and verified.
11. `rllab-training`, `robo_dataview`, and `rllab-data-collection` all read
    a freshly converted v2 bundle end-to-end without flat-alias fallbacks.
