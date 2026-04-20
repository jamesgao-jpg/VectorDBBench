"""Reproduce the Zilliz blog's claim about turbopuffer delete consistency.

Design: delete the top-K candidate pool for 5 fixed queries, then repeatedly
query top-100 for each and watch whether the count returns fewer than 100
(the "drop") and how long the count takes to recover to 100 (the "convergence").

Prerequisites (see delete_consistency_test.md):
  - namespace fully indexed (unindexed_bytes == 0)
  - test.parquet available locally

At the end, re-insert the deleted rows so the namespace is restored.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import turbopuffer as tpuf

API_KEY = "tpuf_cxMjXQvBFFJjCpEGcrJj5SAisrI5ufc4"
REGION = "aws-us-west-2"
NAMESPACE = "laion100m_bulk"
DATASET_DIR = Path("/tmp/vectordb_bench/dataset/laion/laion_large_100m")
TEST_PATH = DATASET_DIR / "test.parquet"

QUERY_COUNT = 5
TOP_K = 100
WIDE_TOP_K = 500  # deletion-pool size per query
POLL_INTERVAL_SEC = 180  # 3 minutes
POLL_ITERATIONS = 20  # 60 minutes total
EARLY_STOP_STREAK = 3  # stop if all queries return 100 for N consecutive polls

RESULTS_PATH = Path(__file__).parent / "delete_consistency_results.json"


def client() -> tpuf.Turbopuffer:
    return tpuf.Turbopuffer(api_key=API_KEY, region=REGION)


def assert_fully_indexed(ns) -> None:
    """Fail fast if the namespace has any unindexed backlog.

    The delete-consistency test assumes a clean fully-indexed starting state so
    that observed count drops can be attributed to the delete, not lingering
    indexing lag. Use wait_for_index_drain.py to satisfy this prerequisite.
    """
    meta = ns.metadata()
    unindexed = getattr(meta.index, "unindexed_bytes", 0) or 0
    rows = meta.approx_row_count
    if unindexed != 0:
        raise SystemExit(
            f"Namespace {ns.id!r} not fully indexed yet: "
            f"unindexed_bytes={unindexed/1e9:.2f}GB, rows={rows:,}. "
            "Run wait_for_index_drain.py first."
        )
    print(f"[prereq] namespace fully indexed: rows={rows:,} unindexed=0", flush=True)


def load_queries(n: int) -> tuple[list[list[float]], list[int]]:
    df = pl.read_parquet(TEST_PATH)
    rows = df.head(n)
    # LAION is L2, no normalization needed
    vecs = np.stack(rows["emb"]).tolist()
    ids = rows["id"].to_list()
    return vecs, ids


def top_k_ids(ns, q: list[float], k: int) -> list[int]:
    res = ns.query(rank_by=("vector", "ANN", q), top_k=k)
    return [int(r.id) for r in (res.rows or [])]


def run_query_batch(ns, queries: list[list[float]], k: int) -> list[list[int]]:
    return [top_k_ids(ns, q, k) for q in queries]


def reinsert_by_ids(ns, ids_to_reinsert: set[int], batch_size: int = 50_000):
    """Re-insert rows by reading the corresponding parquet files and upserting."""
    if not ids_to_reinsert:
        return
    print(f"[restore] Re-inserting {len(ids_to_reinsert)} deleted IDs from parquet...", flush=True)

    # LAION IDs are globally sequential across the 100 files; each file has 1M rows
    # with IDs [file_idx*1_000_000, (file_idx+1)*1_000_000)
    by_file: dict[int, set[int]] = {}
    for rid in ids_to_reinsert:
        file_idx = rid // 1_000_000
        by_file.setdefault(file_idx, set()).add(rid)

    total_restored = 0
    for file_idx in sorted(by_file.keys()):
        fpath = DATASET_DIR / f"train-{file_idx:02d}-of-100.parquet"
        if not fpath.exists():
            print(f"  [warn] {fpath} missing, skipping {len(by_file[file_idx])} IDs", flush=True)
            continue
        wanted_ids = by_file[file_idx]
        df = pl.read_parquet(fpath)
        sub = df.filter(pl.col("id").is_in(list(wanted_ids)))
        ids = sub["id"].to_list()
        vecs = np.stack(sub["emb"]).tolist()
        # Upload in batches
        for i in range(0, len(ids), batch_size):
            bids = ids[i:i + batch_size]
            bvecs = vecs[i:i + batch_size]
            ns.write(
                upsert_columns={"id": bids, "vector": bvecs},
                distance_metric="euclidean_squared",
                disable_backpressure=True,
            )
            total_restored += len(bids)
            print(f"  [restore] re-uploaded {total_restored}/{len(ids_to_reinsert)}", flush=True)
    print(f"[restore] Done: {total_restored} rows re-uploaded", flush=True)


def main():
    c = client()
    ns = c.namespace(NAMESPACE)

    results: dict = {
        "config": {
            "namespace": NAMESPACE,
            "query_count": QUERY_COUNT,
            "top_k": TOP_K,
            "wide_top_k": WIDE_TOP_K,
            "poll_interval_sec": POLL_INTERVAL_SEC,
            "poll_iterations": POLL_ITERATIONS,
        },
        "events": [],
    }

    # Phase 1: prerequisite check (must be fully indexed)
    print("=== Phase 1: prerequisite check ===", flush=True)
    assert_fully_indexed(ns)

    # Phase 2+3: baseline + wide capture
    print("\n=== Phase 2/3: baseline queries ===", flush=True)
    queries, query_source_ids = load_queries(QUERY_COUNT)
    baseline_top_k = run_query_batch(ns, queries, TOP_K)
    wide_top_k = run_query_batch(ns, queries, WIDE_TOP_K)
    for i, (top100, top500) in enumerate(zip(baseline_top_k, wide_top_k)):
        print(f"  Q{i}: top100 returned {len(top100)}, top500 returned {len(top500)}", flush=True)
    results["baseline"] = {
        "top_k_counts": [len(x) for x in baseline_top_k],
        "wide_k_counts": [len(x) for x in wide_top_k],
    }

    # Union of wide-top-k IDs → deletion set
    deletion_set: set[int] = set()
    for ids in wide_top_k:
        deletion_set.update(ids)
    delete_ids = sorted(deletion_set)
    print(f"\n[delete] will delete {len(delete_ids)} unique IDs", flush=True)
    results["deletion_set_size"] = len(delete_ids)

    # Phase 4: delete
    print("\n=== Phase 4: delete ===", flush=True)
    t_del_start = time.perf_counter()
    ns.write(deletes=delete_ids, disable_backpressure=True)
    t_del_dur = time.perf_counter() - t_del_start
    t_delete_completed = time.time()  # epoch
    print(f"  delete API took {t_del_dur:.2f}s", flush=True)
    results["delete_api_duration_sec"] = round(t_del_dur, 3)

    # Phase 5: immediate query
    def measure_iteration(iter_idx: int, label: str) -> dict:
        t_query_start = time.perf_counter()
        counts = []
        returned_any_deleted = []
        for i, q in enumerate(queries):
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

    # Phase 6: poll loop
    print("\n=== Phase 6: poll loop (every 3 min, up to 60 min) ===", flush=True)
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

    # Save results
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] results → {RESULTS_PATH}", flush=True)

    # Phase 7: restore
    print("\n=== Phase 7: restore ===", flush=True)
    reinsert_by_ids(ns, deletion_set)

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
