"""Extract the shared out-of-sample query set for the turbopuffer multi-tenant case.

Reads the first ``--count`` rows of a wide-table Parquet (a source file NOT used
by the prepared insert data, so the query vectors are out-of-sample) and writes
a deterministic JSON query file. The dense vectors and BM25 strings come from
the same rows, so the query set is consistent across the dense and BM25 modes.

The output JSON is consumed by the setup operation of
``TurboPufferMultiTenantColdStart`` via ``--multitenant-queries-file`` and is
recorded in the setup manifest. The script touches only local files; it does
not access S3 or turbopuffer.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DENSE_FIELD = "emb_768"
DEFAULT_BM25_FIELD = "content"
DEFAULT_COUNT = 100
DENSE_DIMENSIONS = 768


def _finite(value: float) -> bool:
    return math.isfinite(value)


def extract_queries(
    source: Path,
    *,
    count: int,
    dense_field: str,
    bm25_field: str,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"query Parquet does not exist: {source}")
    parquet_file = pq.ParquetFile(source, memory_map=True, pre_buffer=False)
    if parquet_file.metadata.num_rows < count:
        raise ValueError(f"query Parquet has {parquet_file.metadata.num_rows} rows, but {count} queries are required")
    columns = {field.name: field for field in parquet_file.schema_arrow}
    if dense_field not in columns or bm25_field not in columns:
        raise ValueError(f"query Parquet must declare {dense_field} and {bm25_field} columns")
    if str(columns[bm25_field].type) != "string":
        raise ValueError(f"query {bm25_field} column must be string")
    dense_type = columns[dense_field].type
    if not (pa.types.is_list(dense_type) and pa.types.is_floating(dense_type.value_type)):
        raise ValueError(f"query {dense_field} column must be a list of floats")

    batch = parquet_file.read_row_group(0, columns=[dense_field, bm25_field]).slice(0, count)
    dense_values = batch.column(dense_field).to_pylist()
    bm25_values = batch.column(bm25_field).to_pylist()

    queries = []
    for index, (vector, text) in enumerate(zip(dense_values, bm25_values, strict=True)):
        if len(vector) != DENSE_DIMENSIONS or not all(_finite(value) for value in vector):
            raise ValueError(f"query {index} {dense_field} vector must be finite {DENSE_DIMENSIONS}-dimensional")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"query {index} {bm25_field} value must be a non-empty string")
        queries.append(
            {
                "index": index,
                "dense": [float(value) for value in vector],
                "bm25": text,
            }
        )
    return {
        "version": 1,
        "count": len(queries),
        "dense_field": dense_field,
        "bm25_field": bm25_field,
        "queries": queries,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w") as output:
        json.dump(value, output, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="wide-table Parquet source")
    parser.add_argument("--output", type=Path, required=True, help="JSON query file to write")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"number of queries (default {DEFAULT_COUNT})")
    parser.add_argument("--dense-field", default=DEFAULT_DENSE_FIELD, help="dense vector column")
    parser.add_argument("--bm25-field", default=DEFAULT_BM25_FIELD, help="BM25 text column")
    args = parser.parse_args()

    queries = extract_queries(
        args.input,
        count=args.count,
        dense_field=args.dense_field,
        bm25_field=args.bm25_field,
    )
    write_json(args.output, queries)
    print(
        f"wrote {queries['count']} queries to {args.output} "
        f"(dense={queries['dense_field']}, bm25={queries['bm25_field']})"
    )


if __name__ == "__main__":
    sys.exit(main())
