"""
Multi-tenant Zilliz loader for Cohere 10M.

Schema: pk (int64, primary) + vector (float[768]) + labels (varchar,
is_partition_key=True). 1000 tenants × 10K rows, mapped via
tenant_labels_1000x10k.parquet.

Environment:
  ZILLIZ_URI        (required)
  ZILLIZ_TOKEN      (required)
  ZILLIZ_USER       (default: db_admin)
  ZILLIZ_PASSWORD   (required if non-token auth; usually ignored)
  COLLECTION        (default: cohere10m_multitenant)
  NUM_PARTITIONS    (default: 64)
  DROP_OLD          (default: 1 = drop existing collection first)
  BATCH_SIZE        (default: 500; Milvus internal wire batch target)
  MAX_WORKERS       (default: cpu_count; threads for concurrent insert,
                     mirrors VDBBench ConcurrentInsertRunner)

Usage:
  ZILLIZ_URI=... ZILLIZ_TOKEN=... python3 load_multitenant_zilliz.py
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pymilvus import DataType, MilvusClient

DATA_DIR = Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m")
TENANT_LABELS = DATA_DIR / "tenant_labels_1000x10k.parquet"
TRAIN_FILES = [DATA_DIR / f"shuffle_train-{i:02d}-of-10.parquet" for i in range(10)]

URI = os.environ["ZILLIZ_URI"]
TOKEN = os.environ["ZILLIZ_TOKEN"]
USER = os.environ.get("ZILLIZ_USER", "db_admin")
PASSWORD = os.environ.get("ZILLIZ_PASSWORD", "")
COLLECTION = os.environ.get("COLLECTION", "cohere10m_multitenant")
NUM_PARTITIONS = int(os.environ.get("NUM_PARTITIONS", "64"))
DROP_OLD = os.environ.get("DROP_OLD", "1") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", str(os.cpu_count() or 8)))


def build_id_to_label():
    print(f"Reading tenant mapping: {TENANT_LABELS}")
    df = pl.read_parquet(TENANT_LABELS)
    id_to_label = dict(zip(df["id"].to_list(), df["labels"].to_list()))
    print(f"  {len(id_to_label):,} id→label mappings")
    return id_to_label


def ensure_collection(client: MilvusClient):
    if DROP_OLD and client.has_collection(COLLECTION):
        print(f"Dropping existing collection {COLLECTION}")
        client.drop_collection(COLLECTION)

    if client.has_collection(COLLECTION):
        print(f"Reusing existing collection {COLLECTION}")
        return

    schema = MilvusClient.create_schema()
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("id", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("labels", DataType.VARCHAR, max_length=32, is_partition_key=True)

    idx = MilvusClient.prepare_index_params()
    idx.add_index(
        field_name="vector",
        index_name="vector_idx",  # match VDBBench's _vector_index_name convention
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    print(f"Creating collection {COLLECTION} (num_partitions={NUM_PARTITIONS})")
    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        num_partitions=NUM_PARTITIONS,
        consistency_level="Session",
    )
    client.create_index(COLLECTION, idx)
    client.load_collection(COLLECTION)


def insert_file(client: MilvusClient, path: Path, id_to_label: dict):
    """Concurrent insert: ThreadPoolExecutor with shared MilvusClient (thread-safe).
    Mirrors VDBBench's ConcurrentInsertRunner (threading backend)."""
    print(f"[{path.name}] reading", flush=True)
    t = pq.read_table(path)  # full file in memory; ~4.5 GB -> large but doable at 32 GB RAM
    ids = t.column("id").to_pylist()
    embs_col = t.column("emb")  # list<float> arrays
    total = len(ids)
    batch_starts = list(range(0, total, BATCH_SIZE))
    log_every = max(1, len(batch_starts) // 20)  # ~5% progress granularity

    start = time.perf_counter()
    counter = {"n": 0}
    lock = threading.Lock()

    def insert_batch(idx: int) -> int:
        b = batch_starts[idx]
        e = min(b + BATCH_SIZE, total)
        batch = [
            {
                "pk": ids[i],
                "id": ids[i],
                "vector": embs_col[i].as_py(),
                "labels": id_to_label[ids[i]],
            }
            for i in range(b, e)
        ]
        res = client.insert(COLLECTION, batch)
        n = res["insert_count"]
        with lock:
            counter["n"] += n
            if idx % log_every == 0:
                dur = time.perf_counter() - start
                print(
                    f"  [{path.name}] batch {idx}/{len(batch_starts)} | "
                    f"{counter['n']:>8,}/{total:,} rows | {dur:.1f}s | "
                    f"{counter['n']/dur:.0f} r/s",
                    flush=True,
                )
        return n

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(insert_batch, range(len(batch_starts))))

    dur = time.perf_counter() - start
    print(f"[{path.name}] DONE: {counter['n']:,} rows in {dur:.1f}s ({counter['n']/dur:.0f} r/s)", flush=True)


def wait_for_index(client: MilvusClient, collection: str, index_name: str = "vector_idx"):
    while True:
        info = client.describe_index(collection, index_name)
        if info is None:
            # pymilvus returns None on INDEX_NOT_FOUND (grpc_handler.py:1572-1573);
            # transient after flush() while index metadata re-registers. Same bug
            # exists in VDBBench milvus.py:159-164 — we just hit it first.
            print("  describe_index returned None (transient INDEX_NOT_FOUND), sleeping 5s", flush=True)
            time.sleep(5)
            continue
        pending = info.get("pending_index_rows", -1)
        if pending == 0:
            return
        print(f"  index pending_index_rows={pending}, sleeping 5s", flush=True)
        time.sleep(5)


def wait_for_compaction(client: MilvusClient, compaction_id: int):
    while True:
        state = client.get_compaction_state(compaction_id)
        if state == "Completed":
            return
        time.sleep(0.5)


def optimize_collection(client: MilvusClient, collection: str):
    """Replicates VDBBench ZillizCloud._optimize() flow.
    flush -> wait_for_index -> compact (force merge) -> wait -> wait_for_index -> refresh_load.
    Skips _wait_for_segments_sorted (Zilliz Cloud denies GetPersistentSegmentInfo).
    """
    print(f"[optimize] flush {collection}")
    client.flush(collection)
    print("[optimize] wait_for_index (pre-compact)")
    wait_for_index(client, collection)
    try:
        print("[optimize] compact (force-merge, target_size=2^63-1)")
        compaction_id = client.compact(collection, target_size=(2**63 - 1))
        if compaction_id > 0:
            print(f"[optimize] wait_for_compaction id={compaction_id}")
            wait_for_compaction(client, compaction_id)
        print("[optimize] wait_for_index (post-compact)")
        wait_for_index(client, collection)
    except Exception as e:
        print(f"[optimize] compact error: {e} — skipping compact")
    print("[optimize] refresh_load")
    client.refresh_load(collection)
    print("[optimize] done")


def main():
    assert TENANT_LABELS.exists(), f"Missing {TENANT_LABELS}; run generate_tenant_labels.py first"
    for p in TRAIN_FILES:
        assert p.exists(), f"Missing {p}"

    print(f"[config] BATCH_SIZE={BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS}", flush=True)
    id_to_label = build_id_to_label()
    client = MilvusClient(uri=URI, user=USER, password=PASSWORD, token=TOKEN, timeout=60)
    ensure_collection(client)

    t0 = time.perf_counter()
    for p in TRAIN_FILES:
        insert_file(client, p, id_to_label)
    insert_dur = time.perf_counter() - t0

    print(f"\n[insert] total: {insert_dur:.1f}s ({10_000_000 / insert_dur:.0f} r/s avg)")

    t_opt = time.perf_counter()
    optimize_collection(client, COLLECTION)
    opt_dur = time.perf_counter() - t_opt

    stats = client.get_collection_stats(COLLECTION)
    idx = client.describe_index(COLLECTION, "vector_idx")
    print(f"\n[summary]")
    print(f"  insert duration:   {insert_dur:.1f}s")
    print(f"  optimize duration: {opt_dur:.1f}s")
    print(f"  row_count:         {stats.get('row_count')}")
    print(f"  indexed_rows:      {idx.get('indexed_rows')}")
    print(f"  index state:       {idx.get('state')}")
    client.close()


if __name__ == "__main__":
    main()
