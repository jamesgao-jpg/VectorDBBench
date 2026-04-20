"""
One-time prep: split the 10 Cohere shuffle_train files into 1000
per-tenant parquet files.

Output:
  /tmp/vectordb_bench/dataset/cohere/cohere_large_10m/per_tenant/tenant_XXXX.parquet

Each output file:
  Schema: id (int64), vector (list<float32>)
  Rows:   exactly 10,000 (one tenant's share)
  Size:   ~30 MB (f32 basis, cosine-normalized)

The vectors are:
  - cast from the source's float64 to float32 (TB + Zilliz f32 indexing)
  - L2-normalized (Cohere is cosine distance; pre-normalizing makes
    both `cosine` and `dot_product` metrics give equivalent rankings)

This replaces the streaming accumulator in load_multitenant_tpuf.py.
Once done, both TB and Zilliz loaders become trivial: iterate 1000
files, one write per file, per-tenant = per-batch.

Runtime: ~10 min on the m6i.2xlarge client (I/O-bound on the source
reads + per-group writes).

Usage:
  python3 split_by_tenant.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA_DIR = Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m")
TENANT_LABELS = DATA_DIR / "tenant_labels_1000x10k.parquet"
TRAIN_FILES = [DATA_DIR / f"shuffle_train-{i:02d}-of-10.parquet" for i in range(10)]
OUT_DIR = DATA_DIR / "per_tenant"


def build_id_to_label() -> dict[int, str]:
    print(f"Reading tenant mapping: {TENANT_LABELS.name}")
    df = pl.read_parquet(TENANT_LABELS)
    return dict(zip(df["id"].to_list(), df["labels"].to_list()))


def process_file(path: Path, file_idx: int, id_to_label: dict[int, str]):
    """Read one shuffle_train file, cast f64→f32, normalize, split into
    per-tenant shards written to OUT_DIR/<tenant>_fileNN.parquet."""
    t_read = time.perf_counter()
    t = pq.read_table(path)
    ids = t.column("id").to_pylist()
    n_rows = len(ids)
    print(f"  [read] {n_rows:,} rows in {time.perf_counter()-t_read:.1f}s")

    t_conv = time.perf_counter()
    # f64 → f32 at Arrow layer (avoids 6 GB transient numpy f64)
    flat_f32 = pc.cast(pc.list_flatten(t.column("emb")), pa.float32())
    embs_np = flat_f32.to_numpy(zero_copy_only=False).reshape(n_rows, 768)
    if not embs_np.flags.writeable:
        embs_np = embs_np.copy()
    del flat_f32, t
    # Normalize in-place for cosine
    embs_np /= np.linalg.norm(embs_np, axis=1, keepdims=True)
    print(f"  [convert+normalize] in {time.perf_counter()-t_conv:.1f}s")

    # Group row indices by tenant
    t_group = time.perf_counter()
    by_tenant: dict[str, list[int]] = {}
    for row_idx, pk_id in enumerate(ids):
        by_tenant.setdefault(id_to_label[pk_id], []).append(row_idx)
    print(f"  [group] {len(by_tenant)} tenants in {time.perf_counter()-t_group:.1f}s")

    # Write one shard per tenant for this file
    t_write = time.perf_counter()
    for tenant, idxs in by_tenant.items():
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        tenant_ids = [ids[i] for i in idxs]
        tenant_embs = embs_np[idxs_arr]
        # Build a pyarrow Table with proper list<float32> column
        emb_flat_arr = pa.array(tenant_embs.reshape(-1), type=pa.float32())
        emb_list_arr = pa.ListArray.from_arrays(
            pa.array(np.arange(len(idxs) + 1) * 768, type=pa.int32()),
            emb_flat_arr,
        )
        out_table = pa.table({
            "id": pa.array(tenant_ids, type=pa.int64()),
            "vector": emb_list_arr,
        })
        out_path = OUT_DIR / f"{tenant}_file{file_idx:02d}.parquet"
        pq.write_table(out_table, out_path, compression="zstd")
    print(
        f"  [write-shards] {len(by_tenant)} shards in "
        f"{time.perf_counter()-t_write:.1f}s"
    )


def merge_shards(tenants: list[str]):
    """Concat the 10 per-file shards into one file per tenant, delete shards."""
    print(f"\n[merge] consolidating shards for {len(tenants)} tenants")
    t0 = time.perf_counter()
    for i, tenant in enumerate(tenants):
        shard_paths = sorted(OUT_DIR.glob(f"{tenant}_file*.parquet"))
        if not shard_paths:
            continue
        merged = pa.concat_tables([pq.read_table(p) for p in shard_paths])
        out_path = OUT_DIR / f"{tenant}.parquet"
        pq.write_table(merged, out_path, compression="zstd")
        for p in shard_paths:
            p.unlink()
        if (i + 1) % 100 == 0:
            dur = time.perf_counter() - t0
            print(f"  [merge] {i+1}/{len(tenants)} | {dur:.1f}s | {(i+1)/dur:.0f}/s")
    print(f"[merge] done in {time.perf_counter()-t0:.1f}s")


def main():
    assert TENANT_LABELS.exists(), f"Missing {TENANT_LABELS}"
    for p in TRAIN_FILES:
        assert p.exists(), f"Missing {p}"

    if OUT_DIR.exists():
        # Clean slate — a partial prior run is cheaper to redo than reason about
        print(f"[clean] removing existing {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    id_to_label = build_id_to_label()
    tenants = sorted(set(id_to_label.values()))
    print(f"  {len(id_to_label):,} id→label, {len(tenants)} tenants")

    t0 = time.perf_counter()
    for file_idx, path in enumerate(TRAIN_FILES):
        print(f"\n[{path.name}]")
        process_file(path, file_idx, id_to_label)

    merge_shards(tenants)

    print(f"\n[done] total {time.perf_counter()-t0:.1f}s")
    # Sanity summary
    files = list(OUT_DIR.glob("tenant_*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"  {len(files)} tenant files | {total_bytes/1e9:.1f} GB on disk")
    if files:
        sample = pq.read_table(files[0])
        print(f"  sample {files[0].name}: {len(sample)} rows, schema={sample.schema}")


if __name__ == "__main__":
    main()
