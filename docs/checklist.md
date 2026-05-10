## 전제

현재 문서는 이미 **Blob v2, FixedSizeList, typed scalar index, binary trajectory hash, half-open task segment, action semantics**까지 들어간 상태라서 기술적으로는 꽤 v2에 가깝습니다. 다만 문서 자체는 아직 “v1.0 final, no planned v1.1/v2.0”이라고 못 박고 있고, flat alias와 compatibility sidecar를 영구 유지하는 방향으로 되어 있습니다.  

따라서 “한번에 2.0까지 간다”는 것은 **새 기능을 많이 추가한다는 뜻이 아니라, v1 호환용 잔재를 지금 정리해서 v2.0 stable contract로 승격한다는 뜻**으로 잡는 게 맞습니다.

---

# RLLAB Lance v2.0 One-Shot Checklist

Legend:

```text
[x] 이미 현재 문서/구현에 반영됨
[~] 일부 반영됨, v2.0 stable 전에 정리 필요
[ ] 해야 함
[drop] v2.0에서도 하지 않음
```

---

## 0. v2.0 목표 정의

### 0.1 v2.0의 핵심 원칙

* [ ] v2.0은 **registry-first / Blob v2 only / canonical sidecar only / no deprecated aliases**로 정의한다.
* [ ] LeRobot과는 계속 **semantic-compatible**이지, LeRobot v2.1/v3 directory layout과 byte-layout compatible이 아님을 유지한다.
* [ ] v2.0은 “로봇 데이터 의미를 새로 발명하는 포맷”이 아니라, **LeRobot semantics를 Lance-native storage로 고정하는 포맷**으로 유지한다.
* [ ] timestamp_ns, quality_flags, calibration, embodiments sidecar는 v2.0 mandatory가 아니다. 현재 논의처럼 과한 항목은 gate에서 제외한다. 

---

## 1. 문서 버전 리셋

현재 문서에는 “v1.0 final format, no planned v1.1 or v2.0”이라는 방향이 남아 있습니다. v2로 한 번에 갈 거면 제일 먼저 이걸 정리해야 합니다. 

### 1.1 파일명 / 문서명

* [ ] `docs/V1_IMPLEMENTATION_CHECKLIST.md`를 `docs/V2_IMPLEMENTATION_CHECKLIST.md`로 변경.
* [ ] 기존 V1 checklist의 상태 snapshot을 v2 기준으로 재작성.
* [ ] `docs/STANDARD.md` 상단을 다음처럼 변경.

```text
Format identifier: rllab_published_lance_dataset_v2
Current schema version: 2.0
Status: v2.0 stable candidate
```

### 1.2 format identifier

강력 추천:

```text
rllab_published_lance_dataset_v2
```

체크리스트:

* [ ] `RLLAB_PUBLISHED_FORMAT`을 `rllab_published_lance_dataset_v2`로 변경.
* [ ] `schema_version`을 `"2.0"`으로 변경.
* [ ] v1 reader가 실수로 v2 bundle을 읽지 않도록 format strict-match 유지.
* [ ] v1 bundle과 v2 bundle을 같은 loader에서 받을 경우, loader가 format으로 분기하도록 구현.

v1 format string을 유지한 채 schema_version만 2.0으로 바꾸는 것은 비추천입니다. 현재 문서가 `format`을 contract discriminator로 쓰고 있기 때문에, major break라면 format도 바꾸는 편이 안전합니다. 

---

## 2. v2.0에서 유지할 것

이 항목들은 이미 방향이 맞습니다. v2에서도 그대로 유지하면 됩니다.

### 2.1 Lance Blob v2

현재 STANDARD는 이미 v1.0 stable bundle에서 Blob v2를 요구하고, legacy `large_binary + lance-encoding:blob=true`를 금지하고 있습니다. 

* [x] `video_blob`는 `lance.blob.v2`.
* [x] `data_storage_version >= "2.2"`.
* [x] `lance.blob_field("video_blob")`.
* [x] `lance.blob_array(...)`.
* [x] external URI blob 금지.
* [x] inline bytes only.
* [x] metadata scan에서 `video_blob` projection 금지.
* [ ] v2 validator에서 legacy blob metadata dataset을 hard-fail 처리.

v2 conformance rule:

```text
If videos.video_blob is large_binary with lance-encoding:blob=true,
the bundle is not v2.0 conformant.
```

---

### 2.2 FixedSizeList state/action

현재 STANDARD는 single-shape dataset에서 episode trajectory를 `large_list<fixed_size_list<float32, dim>>`, frame row를 `fixed_size_list<float32, dim>`로 정의합니다. 이 방향은 v2에서도 그대로 유지합니다. 

* [x] `frames.observation_state`: `fixed_size_list<float32, state_dim>`.
* [x] `frames.action`: `fixed_size_list<float32, action_dim>`.
* [x] `episodes.observation_state`: `large_list<fixed_size_list<float32, state_dim>>`.
* [x] `episodes.actions`: `large_list<fixed_size_list<float32, action_dim>>`.
* [x] 16D / 19D silent mixing 금지.
* [~] multi-embodiment policy는 문서에 있지만 실제 production merge policy는 아직 별도 확인 필요.

v2 rule:

```text
shape_policy == "single"이면 FixedSizeList 기반 schema가 mandatory.
shape_policy 없이 ragged vector를 쓰면 v2 non-conformant.
```

---

### 2.3 Binary `trajectory_sha256`

현재 STANDARD는 JSON hash를 버리고 deterministic little-endian binary hash를 정의합니다. 이건 v2에서도 유지합니다. 

* [x] magic prefix 사용.
* [x] length/state_dim/action_dim 포함.
* [x] timestamp float64 LE.
* [x] state/action float32 LE.
* [x] NaN / Inf hard-reject.
* [ ] validator가 모든 episode hash를 재계산해서 확인.
* [ ] pretrain merge 후에도 hash 재검증.

---

### 2.4 Half-open `task_segments`

현재 문서는 `end_frame_exclusive`, `end_timestamp_exclusive`를 사용합니다. v2에서도 유지합니다. 

* [x] `task_segments.start_frame`.
* [x] `task_segments.end_frame_exclusive`.
* [x] `[start, end)` convention.
* [ ] v1 legacy `end_frame` inclusive field가 있으면 v2 validator hard-fail.
* [ ] training/viewer가 half-open range만 사용하도록 수정.

---

### 2.5 Typed scalar indexes

현재 checklist는 BTREE/BITMAP typed index와 `frames.global_frame_index` BTREE 추가를 완료 상태로 둡니다. 

* [x] `episode_index`, `frame_index`, `global_frame_index`, `media_id` → BTREE.
* [x] `task_index`, `camera_id`, `is_bad_frame`, `split`, `source_dataset`, `session_id`, `embodiment_id` → BITMAP.
* [x] `indexes.created`와 `indexes.recommended` 분리.
* [ ] v2에서는 index record schema를 고정.

권장 v2 schema:

```json
"indexes": {
  "created": {
    "data/frames.lance": [
      {
        "name": "idx_frames_global_frame_index",
        "column": "global_frame_index",
        "type": "BTREE",
        "status": "ready"
      }
    ]
  },
  "recommended": {
    "data/frames.lance": [
      {"column": "global_frame_index", "type": "BTREE"},
      {"column": "is_bad_frame", "type": "BITMAP"}
    ]
  }
}
```

---

## 3. v2.0에서 제거할 것

이게 “한번에 2.0”의 핵심입니다.

---

## 3.1 flat manifest aliases 제거

현재 v1 문서는 flat aliases를 “kept permanently”로 두고 있습니다. v2로 갈 거면 이 결정을 뒤집어야 합니다. 

### 제거 대상

* [ ] `state_column`
* [ ] `action_column`
* [ ] `training_columns`
* [ ] `frame_columns`
* [ ] `state_dim`
* [ ] `action_dim`
* [ ] `camera_keys`
* [ ] `camera_columns`
* [ ] `camera_key_to_column`
* [ ] `fps` alias
* [ ] `media_mode`
* [ ] `camera_storage`
* [ ] `blob_storage`
* [ ] `total_episodes`
* [ ] `total_frames`
* [ ] `total_videos`
* [ ] `total_video_segments`
* [ ] `meta.stats`
* [ ] `meta.tasks`

### v2 대체

```text
state/action location  -> manifest.modalities / manifest.actions
dimensions             -> registry shape
cameras                -> video.* registry entries
fps                    -> manifest.rates
media storage          -> manifest.tables + manifest.lance + capabilities
counts                 -> manifest.counts
stats/tasks paths      -> manifest.meta canonical paths
```

### reader rule

* [ ] v2 reader는 flat aliases를 읽지 않는다.
* [ ] v2 producer는 flat aliases를 emit하지 않는다.
* [ ] v1 compatibility loader가 필요하면 별도 `read_v1_bundle()` 경로로 분리한다.

---

## 3.2 deprecated video alias columns 제거

현재 STANDARD는 `videos.lance.episode_id`와 `media_type`을 deprecated alias로 남깁니다. v2에서는 제거해야 합니다. 

제거 대상:

* [ ] `videos.episode_id`
* [ ] `videos.media_type`
* [x] `videos.uri`는 이미 제거됨.
* [x] `videos.video_path`는 이미 제거됨.

v2 canonical columns:

```text
episode_index
media_id
camera_id
camera_name
relative_path
source_uri
source
source_dataset
source_episode_index
source_dataset_url
source_media_id
source_relative_path
session_id
embodiment_id
chunk_index
file_index
```

Validator:

```text
If videos.lance contains episode_id, media_type, uri, or video_path,
the bundle is not v2.0 conformant.
```

---

## 3.3 per-camera timestamp alias columns 제거

현재 `episodes.lance`에는 deprecated alias로 `{camera}_from_timestamp`, `{camera}_to_timestamp`가 남아 있습니다. v2에서는 `camera_segments`만 canonical로 둡니다. 

제거 대상:

* [ ] `{camera}_from_timestamp`
* [ ] `{camera}_to_timestamp`

v2 canonical:

```text
episodes.camera_segments[*].from_timestamp
episodes.camera_segments[*].to_timestamp
episodes.camera_segments[*].frame_start
episodes.camera_segments[*].frame_count
episodes.camera_segments[*].media_id
```

Validator:

```text
If any episode column ends with _from_timestamp or _to_timestamp and refers to
a camera alias, fail v2 conformance.
```

---

## 3.4 compatibility sidecar 정리

현재 문서는 `meta/tasks.json`과 `meta/stats.json`을 compatibility surface로 유지합니다. v2에서는 source-of-truth 혼란을 없애려면 제거하는 것이 맞습니다. 

### 제거 또는 contract 밖으로 이동

* [ ] `meta/tasks.json` 제거.
* [ ] `meta/stats.json` 제거.

v2 canonical sidecars:

```text
meta/info.json
meta/sessions.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/splits.json
meta/stats/state_body.json
meta/stats/action_body.json
```

현실적인 타협안:

```text
meta/compat/tasks.json
meta/compat/stats.json
```

를 optional export로 허용할 수는 있습니다. 하지만 v2 contract의 canonical `meta` map에는 넣지 않는 것이 좋습니다.

권장 v2 rule:

```text
Root-level meta/tasks.json and meta/stats.json are not part of v2.0.
If compatibility exports are emitted, they must live under meta/compat/
and validators must not treat them as canonical.
```

---

## 4. manifest v2 구조 정리

## 4.1 v2 required top-level keys

v2에서는 top-level을 다음으로만 고정하세요.

```json
{
  "format": "rllab_published_lance_dataset_v2",
  "schema_version": "2.0",
  "dataset_id": "...",
  "created_at": "...",
  "source": "...",
  "source_format": "...",
  "lance": {},
  "counts": {},
  "tables": {},
  "modalities": {},
  "actions": {},
  "rates": {},
  "state_action_alignment": {},
  "indexes": {},
  "reader_hints": {},
  "capabilities": {},
  "meta": {}
}
```

체크리스트:

* [ ] `primary_training_table` 유지 여부 결정.

  * 추천: **유지**. `train_episodes.lance` 같은 future merged table을 가리키는 실용성이 있음.
* [ ] 단, `primary_training_table`은 alias가 아니라 canonical access hint로 정의.
* [ ] `source_dataset` top-level alias 제거.
* [ ] `fps` top-level alias 제거.
* [ ] `counts`만 canonical count source로 유지.
* [ ] `tables`만 canonical table path source로 유지.

---

## 4.2 `capabilities` 정리

v2 capability flags:

```json
"capabilities": {
  "modality_registry_v2": true,
  "frames_table": true,
  "videos_table": true,
  "inline_video_blobs": true,
  "lance_blob_v2": true,
  "camera_segments": true,
  "task_segments": true,
  "trajectory_sha256": true,
  "per_modality_stats": true,
  "fixed_size_state_action": true,
  "action_semantics": true
}
```

체크리스트:

* [ ] `modality_registry_v1` → `modality_registry_v2`.
* [ ] `lance_blob_v2 == true` mandatory.
* [ ] `action_semantics == true` mandatory.
* [ ] `fixed_size_state_action == true` mandatory for `shape_policy="single"`.

---

## 4.3 `reader_hints` 정리

v2에서는 `legacy_aliases_available`을 제거하거나 false로 고정합니다.

```json
"reader_hints": {
  "prefer_registry": true,
  "video_lookup": "videos.media_id",
  "normalization": "meta/stats",
  "lazy_blob_columns": {
    "data/videos.lance": ["video_blob"]
  },
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
  }
}
```

체크리스트:

* [ ] `legacy_aliases_available` 제거.
* [ ] `legacy_normalization` 제거.
* [ ] `normalization`은 `meta/stats`만 가리킴.
* [ ] video metadata projection에 `video_blob`이 없어야 함.
* [ ] `blob_read_api = take_blobs` 유지.

---

## 5. table schema v2 정리

## 5.1 `episodes.lance`

현재 유지:

* [x] `episode_index`
* [x] `task_index`
* [x] `fps`
* [x] `length`
* [x] `timestamps`
* [x] `observation_state`
* [x] `actions`
* [x] `language_instruction`
* [x] `camera_segments`
* [x] `task_segments`
* [x] `trajectory_sha256`
* [x] `split`
* [x] `source_dataset`
* [x] `source_episode_index`
* [x] `session_id`
* [x] `embodiment_id`

v2에서 제거:

* [ ] `{camera}_from_timestamp`
* [ ] `{camera}_to_timestamp`

v2 validator:

* [ ] `camera_segments` list must be non-null.
* [ ] `task_segments` list must be non-null.
* [ ] `camera_segments[*].media_id` must resolve to `videos.media_id`.
* [ ] no `*_video_blob`.
* [ ] no per-camera timestamp aliases.

---

## 5.2 `frames.lance`

현재 구조 유지:

* [x] `episode_index`
* [x] `frame_index`
* [x] `global_frame_index`
* [x] `timestamp`
* [x] `task_index`
* [x] `observation_state`
* [x] `action`
* [x] `state_norm`
* [x] `action_norm`
* [x] `is_bad_frame`
* [x] `split`
* [x] `source_dataset`
* [x] `session_id`
* [x] `embodiment_id`

v2 유지 방침:

* [x] `state_norm`, `action_norm`은 optional.
* [x] `is_bad_frame`은 required non-null.
* [drop] `quality_flags`는 v2 mandatory 아님.
* [drop] `quality_score`는 v2 mandatory 아님.
* [drop] `timestamp_ns`는 v2 mandatory 아님.

현재 논의상 `timestamp_ns`와 `quality_flags`는 실제 워크플로가 생길 때 additive optional로 추가하는 것이 맞습니다. LeRobot도 ns timestamp나 일반 quality flag 시스템을 쓰지 않고, 현재 RLLAB Lance는 이미 float64 timestamp라 실용상 충분하다는 판단이 있었습니다. 

---

## 5.3 `videos.lance`

유지:

* [x] `media_id`
* [x] `episode_index`
* [x] `camera_id`
* [x] `camera_name`
* [x] `video_blob`
* [x] `from_timestamp`
* [x] `to_timestamp`
* [x] `num_frames`
* [x] `sha256`
* [x] `byte_size`
* [x] `width_pixels`
* [x] `height_pixels`
* [x] `fps`
* [x] `codec`
* [x] provenance fields

v2에서 제거:

* [ ] `episode_id`
* [ ] `media_type`
* [x] `uri`
* [x] `video_path`

v2 hard rules:

* [ ] `media_id` unique.
* [ ] `media_id` is stable logical key, never row id.
* [ ] `video_blob` is Blob v2.
* [ ] no external URI blob values.
* [ ] `sha256 == sha256(video_blob bytes)`.
* [ ] `byte_size == len(video_blob bytes)`.

---

## 6. modality / action registry v2

## 6.1 registry mandatory

현재 문서는 registry를 canonical로 두지만 flat alias도 유지합니다. v2에서는 registry만 남깁니다. 

체크리스트:

* [ ] `modalities` mandatory.
* [ ] `actions` mandatory.
* [ ] every state/action/video entry has `kind`.
* [ ] every state/action entry has `shape`.
* [ ] every state/action entry has `shape_policy`.
* [ ] every video entry has `camera_key`, `camera_column`, `media_id_column`, `blob_column`.
* [ ] unknown `kind`는 reader가 skip 가능.
* [ ] requested unknown modality는 clear error.

---

## 6.2 action semantics mandatory

현재 checklist는 action semantics block emit을 완료 상태로 두지만, RLLAB-collected dataset에서 `unknown`을 specific value로 올리는 cross-repo 작업이 남아 있습니다. 

체크리스트:

* [x] `actions.action.body.semantics` block exists.
* [x] required keys:

  * `command_type`
  * `absolute_or_delta`
  * `units`
  * `control_frame`
  * `applies_to_interval`
  * `normalized`
* [ ] `rllab-data-collection`에서 known semantics를 specific value로 emit.
* [ ] at least one RLLAB-collected v2 dataset has `command_type != "unknown"`.
* [ ] training loader refuses to silently assume semantics when value is `unknown`.

---

## 7. cross-repo consumer checklist

현재 checklist에서 가장 큰 남은 일은 cross-repo 소비자입니다. converter 쪽만 끝나도 `rllab-training`, `robo_dataview`, `rllab-data-collection`이 v2를 못 읽으면 stable이 아닙니다. 

---

## 7.1 `rllab-training`

* [ ] format discriminator가 v2를 인식.
* [ ] flat alias를 사용하지 않음.
* [ ] `manifest.modalities` / `manifest.actions` registry로 state/action/video column 해석.
* [ ] `meta/stats/{name}.json`만 canonical normalization으로 사용.
* [ ] `meta/stats.json` fallback 제거.
* [ ] `task_segments.end_frame_exclusive` 지원.
* [ ] binary `trajectory_sha256` shape 무시 또는 검증 가능.
* [ ] `media_id`로 video lookup.
* [ ] Lance Blob v2 `take_blobs` path 사용.
* [ ] metadata scan에서 `video_blob` projection 금지.
* [ ] predicate pushdown에 `split`, `task_index`, `source_dataset`, `embodiment_id`, `is_bad_frame` 사용.
* [ ] `action_semantics`를 model head / action decoder 설정에 반영.
* [ ] `unknown` semantics면 warning 또는 explicit config 요구.

---

## 7.2 `robo_dataview`

* [ ] format discriminator가 v2를 인식.
* [ ] camera discovery를 `modalities.video.*`에서 수행.
* [ ] `camera_keys` / `camera_columns` flat alias 사용 제거.
* [ ] `camera_segments[*].media_id -> videos.media_id` lookup 사용.
* [ ] `media_id + sha256`를 decoder cache key로 사용.
* [ ] Blob v2 `take_blobs`로 video fetch.
* [ ] `episode_id`, `media_type`, `uri`, `video_path` alias에 의존하지 않음.
* [ ] `task_segments` UI가 half-open range를 사용.
* [ ] action semantics 표시:

  * joint position
  * EE pose
  * delta
  * units
  * normalized 여부
* [ ] unsupported modality는 UI에서 graceful skip.

---

## 7.3 `rllab-data-collection`

* [ ] publish path가 v2 format을 emit.
* [ ] `manifest.lance` block emit.
* [ ] `modalities` / `actions` registry emit.
* [ ] `action_semantics`를 specific values로 emit.
* [ ] Blob v2 inline bytes로 video emit.
* [ ] external URI blob reject.
* [ ] FixedSizeList state/action schema emit.
* [ ] `split`, `source_dataset`, `session_id`, `embodiment_id` denormalized columns emit.
* [ ] canonical sidecars only:

  * `tasks.jsonl`
  * `episodes.jsonl`
  * `splits.json`
  * per-modality stats
* [ ] deprecated root `tasks.json` / `stats.json` emit 중단 또는 `meta/compat/`로 이동.

---

## 8. converter / builder checklist

## 8.1 `lerobot2lance`

* [ ] `RLLAB_PUBLISHED_FORMAT = "rllab_published_lance_dataset_v2"`.
* [ ] `schema_version = "2.0"`.
* [ ] flat manifest alias emission 제거.
* [ ] `meta/tasks.json` emission 제거 또는 `meta/compat/tasks.json`으로 이동.
* [ ] `meta/stats.json` emission 제거 또는 `meta/compat/stats.json`으로 이동.
* [ ] `videos.episode_id` 제거.
* [ ] `videos.media_type` 제거.
* [ ] episode per-camera timestamp alias 제거.
* [x] Blob v2 유지.
* [x] FixedSizeList 유지.
* [x] binary trajectory hash 유지.
* [x] half-open task segments 유지.
* [x] typed indexes 유지.
* [ ] validator를 v2 strict mode로 업데이트.

---

## 8.2 pretrain builder

* [ ] v2 manifest만 입력으로 받는 mode 추가.
* [ ] v1 bundle 입력 시 hard-fail 또는 explicit `--allow-v1-source` 요구.
* [ ] shape_policy compatibility 확인.
* [ ] 16D / 19D silent merge 금지.
* [ ] merged output도 `rllab_published_lance_dataset_v2`.
* [ ] `train_episodes.lance`가 있으면 §4.4 trajectory contract 만족.
* [ ] merged stats는 per-modality canonical stats로 emit.
* [ ] merged sidecar에서 deprecated fallback 생성 금지.
* [ ] merged videos Blob v2 integrity 재검증.

---

## 9. validator / conformance checklist

v2 stable gate는 validator가 통과해야 합니다.

### 9.1 manifest

* [ ] `format == "rllab_published_lance_dataset_v2"`.
* [ ] `schema_version == "2.0"` 또는 startswith `"2."`.
* [ ] 모든 required top-level key 존재.
* [ ] flat aliases 없음.
* [ ] `lance.blob_encoding == "lance.blob.v2"`.
* [ ] `lance.data_storage_version >= "2.2"`.
* [ ] `lance.external_blob_uris_allowed == false`.
* [ ] `capabilities.lance_blob_v2 == true`.
* [ ] `capabilities.action_semantics == true`.

### 9.2 sidecars

* [ ] `meta/info.json` exists.
* [ ] `meta/tasks.jsonl` exists.
* [ ] `meta/episodes.jsonl` exists.
* [ ] `meta/splits.json` exists.
* [ ] `meta/sessions.json` exists.
* [ ] `meta/stats/state_body.json` exists.
* [ ] `meta/stats/action_body.json` exists.
* [ ] root `meta/tasks.json` absent.
* [ ] root `meta/stats.json` absent.
* [ ] optional compat exports, if any, live under `meta/compat/`.

### 9.3 table schema

* [ ] `episodes.lance` has no `*_video_blob`.
* [ ] `episodes.lance` has no `{camera}_from_timestamp`.
* [ ] `episodes.lance` has no `{camera}_to_timestamp`.
* [ ] `videos.lance` has no `episode_id`.
* [ ] `videos.lance` has no `media_type`.
* [ ] `videos.lance` has no `uri`.
* [ ] `videos.lance` has no `video_path`.
* [ ] state/action are FixedSizeList-based for `shape_policy="single"`.
* [ ] required non-null columns contain no nulls.
* [ ] required nullable columns exist.

### 9.4 alignment

* [ ] `counts.frames == sum(episodes.length)`.
* [ ] `frames.timestamp == episodes.timestamps[frame_index]`.
* [ ] `frames.observation_state == episodes.observation_state[frame_index]`.
* [ ] `frames.action == episodes.actions[frame_index]`.
* [ ] converter does not shift actions.
* [ ] `task_segments` use `[start, end_exclusive)`.
* [ ] no legacy inclusive `end_frame`.

### 9.5 media integrity

* [ ] `videos.media_id` unique.
* [ ] every `camera_segments[*].media_id` resolves to `videos.media_id`.
* [ ] `media_id` not interpreted as row id.
* [ ] `video_blob` is Blob v2.
* [ ] no external URI blob values.
* [ ] `sha256(video_blob bytes) == videos.sha256`.
* [ ] `len(video_blob bytes) == videos.byte_size`.
* [ ] at least one video decodes successfully per camera.

### 9.6 indexes

* [ ] `indexes.created` lists actual created indexes.
* [ ] `frames.global_frame_index` BTREE exists.
* [ ] `videos.media_id` BTREE exists.
* [ ] `episodes.task_index` BITMAP exists when indexed.
* [ ] `frames.task_index` BITMAP exists when indexed.
* [ ] `frames.is_bad_frame` BITMAP exists when indexed.
* [ ] `videos.camera_id` BITMAP exists when indexed.
* [ ] no stale index entries in manifest.

---

## 10. operational data checklist

기존 converted bundle은 v2가 아닙니다. 현재 checklist도 기존 bundle은 재변환이 필요하다고 봅니다. 

* [ ] 모든 `data/converted_19d/*` 삭제 또는 archive.
* [ ] source LeRobot dataset에서 v2로 전량 재변환.
* [ ] 각 converted bundle에 `validate_bundle.py --strict-v2`.
* [ ] converted root에 `validate_converted_root.py --strict-v2`.
* [ ] pretrain merge 재실행.
* [ ] merged pretrain bundle validation.
* [ ] sample training run.
* [ ] sample viewer run.
* [ ] HF publish dry-run.
* [ ] HF published bundle remote read test.
* [ ] remote video blob `take_blobs` test.
* [ ] remote metadata scan이 `video_blob`를 project하지 않는지 확인.

---

## 11. 하지 않을 것

v2.0로 가도 아래는 gate에 넣지 마세요. 지금 넣으면 포맷은 커지지만 실제 이득은 작습니다.

* [drop] `timestamp_ns` mandatory.
* [drop] `quality_flags` mandatory.
* [drop] `quality_score` mandatory.
* [drop] `meta/embodiments.json` mandatory.
* [drop] `meta/calibration/cameras.json` mandatory.
* [drop] `primary_access_patterns` mandatory.
* [drop] keyframes table mandatory.
* [drop] external MP4 primary storage.
* [drop] in-place v1 → v2 migration CLI.

이 항목들은 필요해지면 v2.x additive optional로 추가하면 됩니다. 현재 논의에서도 `timestamp_ns`, `quality_flags`, embodiments, calibration은 당장 gate에 넣지 않는 쪽이 실용적이라고 판단했습니다. 

---

# 우선순위별 실행 순서

## 1순위 — 문서/contract 변경

* [ ] `format`을 v2로 변경.
* [ ] `schema_version`을 2.0으로 변경.
* [ ] “v1.0 final, no v2” 문구 제거.
* [ ] flat aliases 제거 정책 반영.
* [ ] deprecated table alias 제거 정책 반영.
* [ ] root `tasks.json` / `stats.json` 제거 정책 반영.
* [ ] `V1_IMPLEMENTATION_CHECKLIST.md` → `V2_IMPLEMENTATION_CHECKLIST.md`.

## 2순위 — converter 변경

* [ ] v2 manifest emit.
* [ ] flat aliases emit 중단.
* [ ] deprecated table columns emit 중단.
* [ ] compatibility sidecar emit 중단.
* [ ] strict v2 validator 업데이트.

## 3순위 — consumers 변경

* [ ] `rllab-training` registry-only reader.
* [ ] `robo_dataview` registry-only camera/video path.
* [ ] `rllab-data-collection` v2 publisher.
* [ ] Blob v2 read path end-to-end 확인.

## 4순위 — data 재생성

* [ ] converted root validation.
* [ ] pretrain merge 재실행.
* [ ] training/viewer smoke test.
* [ ] publish dry-run.

---

# 최종 v2.0 definition of done

```text
A bundle is RLLAB Lance v2.0 stable only if:

1. format == rllab_published_lance_dataset_v2
2. schema_version starts with 2.
3. Blob v2 only; no legacy blob metadata
4. registry-only manifest; no flat aliases
5. canonical sidecars only; no root tasks.json / stats.json
6. no deprecated table alias columns
7. FixedSizeList state/action for single-shape datasets
8. binary trajectory_sha256
9. half-open task_segments
10. logical media_id lookup
11. typed scalar indexes recorded and valid
12. rllab-training, robo_dataview, and rllab-data-collection all pass against a freshly converted v2 bundle
```

한 줄로 정리하면: **현재 문서는 기술적으로 v2에 가까우니, 남은 일은 “호환 alias를 계속 끌고 갈지”가 아니라 “v2에서 과감히 끊을 것들을 실제로 제거하고 cross-repo reader를 맞추는 것”입니다.**
