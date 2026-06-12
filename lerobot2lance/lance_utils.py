"""Shared Lance helpers for RLLAB published bundle writers."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any


LANCE_DATA_STORAGE_VERSION = "2.2"
LANCE_BLOB_ENCODING = "lance.blob.v2"
PUBLISHED_BLOB_POLICY = "inline_bytes_only"
COMPACT_MAX_BYTES_PER_FILE = 4 * 1024 * 1024 * 1024
COMPACT_NUM_THREADS = 8


def add_compaction_args(parser: argparse.ArgumentParser, *, compact_help: str) -> None:
    parser.add_argument(
        "--no-compact",
        dest="compact",
        action="store_false",
        help=compact_help,
    )
    parser.set_defaults(compact=True)
    parser.add_argument(
        "--compact-max-bytes",
        type=int,
        default=COMPACT_MAX_BYTES_PER_FILE,
        help=(
            "Target maximum bytes per compacted Lance file/blob batch. "
            f"Default: {COMPACT_MAX_BYTES_PER_FILE} (4 GiB)."
        ),
    )
    parser.add_argument(
        "--compact-num-threads",
        type=int,
        default=COMPACT_NUM_THREADS,
        help=f"Threads reserved for compaction implementations. Default: {COMPACT_NUM_THREADS}.",
    )
    parser.add_argument(
        "--keep-old-versions",
        action="store_true",
        help="Keep pre-compaction Lance versions instead of cleaning old versions.",
    )


def assert_lance_storage_version(lance: Any, path: Path) -> None:
    ds = lance.dataset(str(path))
    if str(getattr(ds, "data_storage_version", "")) != LANCE_DATA_STORAGE_VERSION:
        raise RuntimeError(
            f"{path} was not written with Lance data_storage_version "
            f"{LANCE_DATA_STORAGE_VERSION}"
        )


def is_blob_field(field: Any) -> bool:
    if field.metadata and field.metadata.get(b"lance-encoding:blob") == b"true":
        return True
    return getattr(field.type, "extension_name", None) == LANCE_BLOB_ENCODING


def table_from_pylist_with_blob_columns(
    pa: Any,
    lance: Any,
    rows: list[dict[str, Any]],
    *,
    schema: Any,
    blob_columns: set[str],
) -> Any:
    if not blob_columns:
        return pa.Table.from_pylist(rows, schema=schema)
    arrays = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if field.name in blob_columns:
            validate_inline_blob_values(values, field.name)
            arrays.append(lance.blob_array(values))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def validate_inline_blob_values(values: list[Any], column: str) -> None:
    for value in values:
        if value is None or isinstance(value, (bytes, bytearray, memoryview)):
            continue
        raise TypeError(
            f"{column} must contain inline bytes only for published v2 bundles; "
            f"got {type(value).__name__}"
        )


def clear_blobs(row: dict[str, Any], blob_columns: set[str]) -> None:
    for column in blob_columns:
        row[column] = None


def materialize_blobs(
    ds: Any,
    row: dict[str, Any],
    source_columns: set[str],
    blob_columns: set[str],
    source_row_index: int,
) -> None:
    for column in blob_columns:
        value = row.get(column)
        if value is None or isinstance(value, (bytes, bytearray, memoryview)):
            continue
        if column not in source_columns:
            row[column] = None
            continue
        handles = ds.take_blobs(column, indices=[source_row_index])
        if not handles:
            row[column] = None
            continue
        handle = handles[0]
        try:
            row[column] = handle.readall() if hasattr(handle, "readall") else handle.read()
        finally:
            handle.close()


def scan_batches(ds: Any, *, batch_size: int, columns: list[str] | None = None) -> Iterable[Any]:
    scanner = ds.scanner(columns=columns, batch_size=batch_size)
    yield from scanner.to_batches()


def compact_lance_tables(
    lance: Any,
    output: Path,
    *,
    max_bytes_per_file: int,
    num_threads: int,
    blob_write_target_bytes: int,
) -> None:
    _ = num_threads
    for name in ("episodes", "train_episodes", "frames"):
        path = output / "data" / f"{name}.lance"
        if not path.exists():
            continue
        rewrite_table_compacted(
            lance,
            path,
            max_bytes_per_file=max_bytes_per_file,
            batch_size=100_000,
        )
    videos_path = output / "data" / "videos.lance"
    if videos_path.exists():
        rewrite_blob_table_compacted(
            lance,
            videos_path,
            max_bytes_per_file=max_bytes_per_file,
            target_blob_bytes=blob_write_target_bytes,
        )


def rewrite_table_compacted(
    lance: Any,
    path: Path,
    *,
    max_bytes_per_file: int,
    batch_size: int,
) -> None:
    import pyarrow as pa

    ds = lance.dataset(str(path))
    schema = ds.schema
    tmp_path = path.with_name(f"{path.name}.compact_tmp")
    backup_path = path.with_name(f"{path.name}.precompact_backup")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)

    mode = "overwrite"
    total_rows = 0
    writes = 0
    pending_batches: list[Any] = []
    pending_bytes = 0

    def flush_batches() -> None:
        nonlocal pending_batches, pending_bytes, mode, total_rows, writes
        if not pending_batches:
            return
        table = pa.Table.from_batches(pending_batches, schema=schema)
        if table.num_rows == 0:
            pending_batches = []
            pending_bytes = 0
            return
        lance.write_dataset(
            table,
            str(tmp_path),
            mode=mode,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
            max_bytes_per_file=max_bytes_per_file,
        )
        mode = "append"
        total_rows += table.num_rows
        writes += 1
        print(
            f"rewrote {path.name} compact write {writes}: "
            f"rows={table.num_rows} bytes={pending_bytes}",
            flush=True,
        )
        pending_batches = []
        pending_bytes = 0

    for batch in scan_batches(ds, batch_size=batch_size):
        if batch.num_rows == 0:
            continue
        pending_batches.append(batch)
        pending_bytes += int(getattr(batch, "nbytes", 0) or 0)
        if pending_bytes >= max_bytes_per_file:
            flush_batches()
    flush_batches()

    if total_rows:
        assert_lance_storage_version(lance, tmp_path)
        path.rename(backup_path)
        tmp_path.rename(path)
        shutil.rmtree(backup_path)
    else:
        shutil.rmtree(tmp_path, ignore_errors=True)
    print(f"rewrote {path}: rows={total_rows} batches={writes}", flush=True)


def rewrite_blob_table_compacted(
    lance: Any,
    path: Path,
    *,
    max_bytes_per_file: int,
    target_blob_bytes: int,
) -> None:
    import pyarrow as pa

    ds = lance.dataset(str(path))
    schema = ds.schema
    blob_columns = {field.name for field in schema if is_blob_field(field)}
    if not blob_columns:
        return

    tmp_path = path.with_name(f"{path.name}.compact_tmp")
    backup_path = path.with_name(f"{path.name}.precompact_backup")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)

    source_columns = set(schema.names)
    rows: list[dict[str, Any]] = []
    batch_blob_bytes = 0
    mode = "overwrite"
    total_rows = 0
    writes = 0

    def flush_rows() -> None:
        nonlocal rows, batch_blob_bytes, mode, writes
        if not rows:
            return
        table = table_from_pylist_with_blob_columns(
            pa,
            lance,
            rows,
            schema=schema,
            blob_columns=blob_columns,
        )
        lance.write_dataset(
            table,
            str(tmp_path),
            mode=mode,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
            max_bytes_per_file=max_bytes_per_file,
        )
        mode = "append"
        writes += 1
        print(
            f"rewrote {path.name} compact batch {writes}: "
            f"rows={len(rows)} blob_bytes={batch_blob_bytes}",
            flush=True,
        )
        rows = []
        batch_blob_bytes = 0

    for batch in scan_batches(ds, batch_size=256):
        for row in batch.to_pylist():
            source_row_index = total_rows
            out = dict(row)
            materialize_blobs(ds, out, source_columns, blob_columns, source_row_index)
            row_blob_bytes = sum(
                len(value)
                for column in blob_columns
                for value in [out.get(column)]
                if isinstance(value, (bytes, bytearray, memoryview))
            )
            if rows and batch_blob_bytes + row_blob_bytes > target_blob_bytes:
                flush_rows()
            rows.append(out)
            batch_blob_bytes += row_blob_bytes
            total_rows += 1
    flush_rows()

    if total_rows:
        assert_lance_storage_version(lance, tmp_path)
        path.rename(backup_path)
        tmp_path.rename(path)
        shutil.rmtree(backup_path)
    else:
        shutil.rmtree(tmp_path, ignore_errors=True)
    print(f"rewrote {path}: rows={total_rows} batches={writes}", flush=True)


def cleanup_lance_tables(lance: Any, output: Path) -> None:
    for name in ("episodes", "train_episodes", "frames", "videos"):
        path = output / "data" / f"{name}.lance"
        if path.exists():
            cleanup_lance_table(lance, path)


def cleanup_lance_table(lance: Any, path: Path) -> None:
    ds = lance.dataset(str(path))
    stats = ds.cleanup_old_versions(
        retain_versions=1,
        delete_unverified=True,
        error_if_tagged_old_versions=False,
    )
    print(f"cleaned old versions {path}: {stats}", flush=True)
