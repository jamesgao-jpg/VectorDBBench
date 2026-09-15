#!/usr/bin/env python3

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vectordb_bench.backend.turbopuffer_multitenant import (
    PREPARED_SCHEMA as OUTPUT_SCHEMA,
    SOURCE_COLUMNS,
    SOURCE_FIELDS,
)

DEFAULT_SOURCE = "s3://file-transfering-bucket/widetablebenchmark/1b-clean/"
TARGET_ROWS = 5_000_000
DEFAULT_BATCH_ROWS = 16_384

def parse_s3_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    prefix = parsed.path.strip("/")
    return f"{parsed.netloc}/{prefix}" if prefix else parsed.netloc


def validate_source_schema(schema: pa.Schema) -> None:
    missing = [field.name for field in SOURCE_FIELDS if schema.get_field_index(field.name) < 0]
    if missing:
        raise ValueError(f"source Parquet is missing required columns: {missing}")
    for expected in SOURCE_FIELDS:
        actual = schema.field(expected.name)
        if actual.type != expected.type or actual.nullable != expected.nullable:
            raise ValueError(
                f"source column {expected.name!r} must be {expected.type} "
                f"(nullable={expected.nullable}), got {actual.type} (nullable={actual.nullable})"
            )


def _download_file(filesystem: Any, remote_path: str, remote_root: str, download_dir: Path) -> Path:
    relative = PurePosixPath(remote_path).relative_to(PurePosixPath(remote_root))
    local_path = download_dir.joinpath(*relative.parts)
    remote_size = int(filesystem.info(remote_path)["size"])
    if local_path.exists() and local_path.stat().st_size == remote_size:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_name(f"{local_path.name}.partial")
    partial_path.unlink(missing_ok=True)
    filesystem.download(remote_path, str(partial_path))
    if partial_path.stat().st_size != remote_size:
        raise IOError(f"downloaded size does not match S3 object: {remote_path}")
    partial_path.replace(local_path)
    return local_path


def download_until_rows(
    source_uri: str,
    download_dir: Path,
    target_rows: int,
    *,
    filesystem: Any | None = None,
) -> tuple[list[Path], int]:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    if filesystem is None:
        import s3fs

        filesystem = s3fs.S3FileSystem(anon=False)

    remote_root = parse_s3_uri(source_uri)
    source_objects = sorted(path for path in filesystem.find(remote_root) if path.lower().endswith(".parquet"))
    if not source_objects:
        raise FileNotFoundError(f"no Parquet files found under {source_uri}")

    downloaded = []
    available_rows = 0
    for remote_path in source_objects:
        local_path = _download_file(filesystem, remote_path, remote_root, download_dir)
        parquet_file = pq.ParquetFile(local_path, memory_map=True, pre_buffer=False)
        validate_source_schema(parquet_file.schema_arrow)
        rows = parquet_file.metadata.num_rows
        if rows <= 0:
            raise ValueError(f"source Parquet has no rows: {remote_path}")
        downloaded.append(local_path)
        available_rows += rows
        if available_rows >= target_rows:
            return downloaded, available_rows

    raise ValueError(f"source contains {available_rows} rows, fewer than required {target_rows}")


def _validate_batch(batch: pa.RecordBatch, first_output_id: int) -> None:
    for field in SOURCE_FIELDS:
        column = batch.column(batch.schema.get_field_index(field.name))
        if not field.nullable and column.null_count:
            raise ValueError(f"source column {field.name!r} contains nulls near output id {first_output_id}")

    vectors = batch.column(batch.schema.get_field_index("emb_768"))
    dimensions = pc.list_value_length(vectors)
    if not pc.all(pc.equal(dimensions, 768)).as_py():
        raise ValueError(f"emb_768 contains a vector with a dimension other than 768 near output id {first_output_id}")
    if vectors.values.null_count or not pc.all(pc.is_finite(vectors.values)).as_py():
        raise ValueError(f"emb_768 contains a null or non-finite value near output id {first_output_id}")

    for field_name in ("bluesky_json", "$meta"):
        values = batch.column(batch.schema.get_field_index(field_name)).to_pylist()
        for offset, value in enumerate(values):
            try:
                decoded = json.loads(value)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{field_name} contains invalid JSON at output id {first_output_id + offset}"
                ) from error
            if not isinstance(decoded, dict):
                raise ValueError(f"{field_name} must contain a JSON object at output id {first_output_id + offset}")


def _prepared_batches(source_files: list[Path], target_rows: int, batch_rows: int):
    output_id = 0
    for source_file in source_files:
        parquet_file = pq.ParquetFile(source_file, memory_map=True, pre_buffer=True)
        validate_source_schema(parquet_file.schema_arrow)
        for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=list(SOURCE_COLUMNS)):
            remaining = target_rows - output_id
            if remaining <= 0:
                return
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            _validate_batch(batch, output_id)
            arrays = [pa.array(range(output_id, output_id + batch.num_rows), type=pa.int64())]
            arrays.extend(batch.column(batch.schema.get_field_index(name)) for name in SOURCE_COLUMNS)
            yield pa.RecordBatch.from_arrays(arrays, schema=OUTPUT_SCHEMA)
            output_id += batch.num_rows

    if output_id != target_rows:
        raise ValueError(f"downloaded Parquet files contain {output_id} usable rows, expected {target_rows}")


def write_prepared_parquet(
    source_files: list[Path],
    output: Path,
    target_rows: int,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_name(f"{output.name}.partial")
    partial_output.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial_output, OUTPUT_SCHEMA, compression="zstd")
    try:
        for batch in _prepared_batches(source_files, target_rows, batch_rows):
            writer.write_batch(batch)
    except Exception:
        writer.close()
        partial_output.unlink(missing_ok=True)
        raise
    writer.close()

    prepared = pq.ParquetFile(partial_output, memory_map=True, pre_buffer=False)
    if prepared.metadata.num_rows != target_rows or prepared.schema_arrow != OUTPUT_SCHEMA:
        partial_output.unlink(missing_ok=True)
        raise RuntimeError("prepared Parquet verification failed")
    partial_output.replace(output)


def prepare_dataset(
    source_uri: str,
    download_dir: Path,
    output: Path,
    *,
    target_rows: int = TARGET_ROWS,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    filesystem: Any | None = None,
) -> dict[str, int | str]:
    source_files, available_rows = download_until_rows(
        source_uri,
        download_dir,
        target_rows,
        filesystem=filesystem,
    )
    write_prepared_parquet(source_files, output, target_rows, batch_rows=batch_rows)
    return {
        "source_files": len(source_files),
        "downloaded_rows": available_rows,
        "output_rows": target_rows,
        "output": str(output),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the 5M-row turbopuffer multi-tenant Parquet dataset.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Private S3 prefix; credentials use the AWS chain.")
    parser.add_argument("--download-dir", required=True, type=Path, help="Directory for downloaded source Parquets.")
    parser.add_argument("--output", required=True, type=Path, help="New prepared Parquet file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(prepare_dataset(args.source, args.download_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
