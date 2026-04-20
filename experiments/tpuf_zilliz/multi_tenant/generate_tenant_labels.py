"""
Generate the shared tenant-to-row mapping for the multi-tenant test.

Reads Cohere 10M's shuffle_train parquet files, assigns each row to one
of NUM_TENANTS tenants deterministically (tenant_id = id % NUM_TENANTS),
and writes a parquet file with [id, labels] columns that both the
Zilliz loader and the Turbopuffer loader consume.

Assignment is deterministic and stateless: given the same id, both
systems route it to the same tenant. No need to preserve file order.

Layout:
  NUM_TENANTS = 1000
  ROWS_PER_TENANT = 10_000   (10M / 1000)
  Labels = "tenant_0000" ... "tenant_0999"
  Output: tenant_labels_1000x10k.parquet
"""

import pyarrow.parquet as pq
import polars as pl
from pathlib import Path

DATA_DIR = Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m")
OUTPUT = DATA_DIR / "tenant_labels_1000x10k.parquet"
NUM_TENANTS = 1000


def main():
    ids = []
    for i in range(10):
        p = DATA_DIR / f"shuffle_train-{i:02d}-of-10.parquet"
        print(f"Reading {p.name}")
        t = pq.read_table(p, columns=["id"])
        ids.extend(t.column("id").to_pylist())
    print(f"Total ids: {len(ids):,}")
    assert len(ids) == 10_000_000, f"expected 10M, got {len(ids):,}"
    # Assign: deterministic, uniform 10K per tenant
    labels = [f"tenant_{(i % NUM_TENANTS):04d}" for i in ids]
    df = pl.DataFrame({"id": ids, "labels": labels})
    df.write_parquet(OUTPUT)
    # Sanity check
    counts = df.group_by("labels").len().sort("labels")
    mn, mx = counts["len"].min(), counts["len"].max()
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print(f"Tenants: {counts.height}, rows/tenant min={mn} max={mx}")


if __name__ == "__main__":
    main()
