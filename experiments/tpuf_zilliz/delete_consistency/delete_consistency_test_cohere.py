"""Delete-consistency reproduction on Cohere 10M (faster variant).

Smaller-corpus counterpart to `delete_consistency_test.py` (LAION 100M).
Same protocol: delete the top-500 pool of 5 fixed queries, then poll top-100
for each query and record when the count recovers.

Prerequisites:
  - namespace `cohere10m_stream_bpOFF` fully indexed (unindexed_bytes == 0)
  - `/tmp/vectordb_bench/dataset/cohere/cohere_large_10m/` populated

Restore reads all 10 shuffled parquet files once (IDs are not sequentially
packed per file, unlike LAION).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import turbopuffer as tpuf

API_KEY = "tpuf_cxMjXQvBFFJjCpEGcrJj5SAisrI5ufc4"
REGION = "aws-us-west-2"
NAMESPACE = "cohere10m_stream_bpOFF"
DATASET_DIR = Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m")
TEST_PATH = DATASET_DIR / "test.parquet"
TRAIN_GLOB = "shuffle_train-*-of-10.parquet"

QUERY_COUNT = 5
TOP_K = 100
WIDE_TOP_K = 500

# Env overrides let us extend the observation window without modifying the
# core protocol (e.g. POLL_ITERATIONS=60 for t+60 min). Defaults match the
# original 30-minute run so the unchanged invocation reproduces it exactly.
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", 60))
POLL_ITERATIONS = int(os.environ.get("POLL_ITERATIONS", 30))
EARLY_STOP_STREAK = int(os.environ.get("EARLY_STOP_STREAK", 3))

_tag = f"_poll{POLL_ITERATIONS}x{POLL_INTERVAL_SEC}s" if os.environ.get("POLL_ITERATIONS") or os.environ.get("POLL_INTERVAL_SEC") else ""
RESULTS_PATH = Path(__file__).parent / f"delete_consistency_results_cohere{_tag}.json"


def client() -> tpuf.Turbopuffer:
    return tpuf.Turbopuffer(api_key=API_KEY, region=REGION)


def assert_fully_indexed(ns) -> None:
    meta = ns.metadata()
    unindexed = getattr(meta.index, "unindexed_bytes", 0) or 0
    rows = meta.approx_row_count
    if unindexed != 0:
        raise SystemExit(
            f"Namespace {ns.id!r} not fully indexed: "
            f"unindexed_bytes={unindexed/1e9:.2f}GB, rows={rows:,}."
        )
    print(f"[prereq] namespace fully indexed: rows={rows:,} unindexed=0", flush=True)


def load_queries(n: int) -> list[list[float]]:
    df = pl.read_parquet(TEST_PATH).head(n)
    return np.stack(df["emb"]).tolist()


def top_k_ids(ns, q: list[float], k: int) -> list[int]:
    res = ns.query(rank_by=("vector", "ANN", q), top_k=k)
    return [int(r.id) for r in (res.rows or [])]


def run_query_batch(ns, queries: list[list[float]], k: int) -> list[list[int]]:
    return [top_k_ids(ns, q, k) for q in queries]


def reinsert_by_ids(ns, ids_to_reinsert: set[int], batch_size: int = 50_000):
    """Scan all shuffled train parquet files once, filter to needed IDs, upsert."""
    if not ids_to_reinsert:
        return
    print(f"[restore] Re-inserting {len(ids_to_reinsert)} deleted IDs from parquet...", flush=True)

    remaining = set(ids_to_reinsert)
    total_restored = 0
    for fpath in sorted(DATASET_DIR.glob(TRAIN_GLOB)):
        if not remaining:
            break
        df = pl.read_parquet(fpath)
        sub = df.filter(pl.col("id").is_in(list(remaining)))
        if sub.height == 0:
            continue
        ids = sub["id"].to_list()
        vecs = np.stack(sub["emb"]).tolist()
        for i in range(0, len(ids), batch_size):
            bids = ids[i:i + batch_size]
            bvecs = vecs[i:i + batch_size]
            ns.write(
                upsert_columns={"id": bids, "vector": bvecs},
                distance_metric="cosine_distance",
                disable_backpressure=True,
            )
            total_restored += len(bids)
        remaining -= set(ids)
        print(f"  [restore] {fpath.name}: +{len(ids)} ({total_restored}/{len(ids_to_reinsert)} done, {len(remaining)} remaining)", flush=True)
    if remaining:
        print(f"  [warn] {len(remaining)} IDs not found in any train file", flush=True)
    print(f"[restore] Done: {total_restored} rows re-uploaded", flush=True)


def main():
    c = client()
    ns = c.namespace(NAMESPACE)

    results: dict = {
        "config": {
            "namespace": NAMESPACE,
            "dataset": "cohere_large_10m",
            "query_count": QUERY_COUNT,
            "top_k": TOP_K,
            "wide_top_k": WIDE_TOP_K,
            "poll_interval_sec": POLL_INTERVAL_SEC,
            "poll_iterations": POLL_ITERATIONS,
        },
        "events": [],
    }

    print("=== Phase 1: prerequisite check ===", flush=True)
    assert_fully_indexed(ns)

    print("\n=== Phase 2/3: baseline queries ===", flush=True)
    queries = load_queries(QUERY_COUNT)
    baseline_top_k = run_query_batch(ns, queries, TOP_K)
    wide_top_k = run_query_batch(ns, queries, WIDE_TOP_K)
    for i, (top100, top500) in enumerate(zip(baseline_top_k, wide_top_k)):
        print(f"  Q{i}: top100 returned {len(top100)}, top500 returned {len(top500)}", flush=True)
    results["baseline"] = {
        "top_k_counts": [len(x) for x in baseline_top_k],
        "wide_k_counts": [len(x) for x in wide_top_k],
    }

    deletion_set: set[int] = set()
    for ids in wide_top_k:
        deletion_set.update(ids)
    delete_ids = sorted(deletion_set)
    print(f"\n[delete] will delete {len(delete_ids)} unique IDs", flush=True)
    results["deletion_set_size"] = len(delete_ids)

    print("\n=== Phase 4: delete ===", flush=True)
    t_del_start = time.perf_counter()
    ns.write(deletes=delete_ids, disable_backpressure=True)
    t_del_dur = time.perf_counter() - t_del_start
    t_delete_completed = time.time()
    print(f"  delete API took {t_del_dur:.2f}s", flush=True)
    results["delete_api_duration_sec"] = round(t_del_dur, 3)

    def measure_iteration(iter_idx: int, label: str) -> dict:
        t_query_start = time.perf_counter()
        counts = []
        returned_any_deleted = []
        for q in queries:
            ids = top_k_ids(ns, q, TOP_K)
            counts.append(len(ids))
            returned_any_deleted.append(any(rid in deletion_set for rid in ids))
        t_query_dur = time.perf_counter() - t_query_start
        elapsed_since_delete = time.time() - t_delete_completed
        event = {
            "iteration": iter_idx,
            "label": label,
            "elapsed_since_delete_sec": round(elapsed_since_delete, 2),
            "query_duration_sec": round(t_query_dur, 3),
            "counts_per_query": counts,
            "any_deleted_returned_per_query": returned_any_deleted,
        }
        results["events"].append(event)
        print(f"[iter {iter_idx} {label}] t+{elapsed_since_delete:>6.1f}s counts={counts} "
              f"any_deleted_in_result={any(returned_any_deleted)}", flush=True)
        return event

    print("\n=== Phase 5: immediate post-delete query ===", flush=True)
    measure_iteration(0, "immediate")

    print(f"\n=== Phase 6: poll loop (every {POLL_INTERVAL_SEC}s, up to {POLL_INTERVAL_SEC*POLL_ITERATIONS//60} min) ===", flush=True)
    streak = 0
    for i in range(1, POLL_ITERATIONS + 1):
        time.sleep(POLL_INTERVAL_SEC)
        ev = measure_iteration(i, f"poll-{i}")
        if all(c >= TOP_K for c in ev["counts_per_query"]):
            streak += 1
            if streak >= EARLY_STOP_STREAK:
                print(f"[early stop] all queries at {TOP_K} for {streak} consecutive iterations", flush=True)
                break
        else:
            streak = 0

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] results → {RESULTS_PATH}", flush=True)

    print("\n=== Phase 7: restore ===", flush=True)
    reinsert_by_ids(ns, deletion_set)

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
