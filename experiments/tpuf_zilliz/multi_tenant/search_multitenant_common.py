"""
Shared helpers for the per-backend multi-tenant search drivers
(search_multitenant_tpuf.py, search_multitenant_zilliz.py).

What lives here:
  - Dataset paths + small parquet reads (test vectors, tenant list)
  - Latency percentile aggregation
  - Generic concurrent runner using ProcessPoolExecutor (spawn) +
    Manager queue/condition for synchronized starts
  - Generic serial runner

Backend-specific code (client construction, search call, optional
namespace warming) lives in the per-backend file.
"""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pyarrow.parquet as pq

DATA_DIR = Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m")
TENANT_LABELS = DATA_DIR / "tenant_labels_1000x10k.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"


def load_test_vectors() -> list[list[float]]:
    return list(pq.read_table(TEST_PARQUET).column("emb").to_pylist())


def load_tenants() -> list[str]:
    df = pl.read_parquet(TENANT_LABELS)
    return sorted(set(df["labels"].to_list()))


def percentiles(latencies_s: list[float]) -> dict:
    if not latencies_s:
        return {"n": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
    ms = np.array(latencies_s) * 1000
    return {
        "n": len(ms),
        "avg_ms": float(np.mean(ms)),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
    }


# ---------- Single-process serial runner ----------

def run_serial(make_searcher: Callable, k: int, n_queries: int,
               vectors: list[list[float]], tenants: list[str]) -> dict:
    """Single-threaded; one query at a time. `make_searcher` returns
    `(search_fn, close_fn)` and is invoked once in the main process.
    `vectors` and `tenants` are loaded once by the caller and passed in."""
    search, close = make_searcher(k)
    rng = random.Random(0)
    n = min(n_queries, len(vectors))
    t0 = time.perf_counter()
    latencies: list[float] = []
    for i in range(n):
        vec = vectors[i % len(vectors)]
        tenant = tenants[rng.randrange(len(tenants))]
        s = time.perf_counter()
        try:
            search(vec, tenant)
            latencies.append(time.perf_counter() - s)
        except Exception as e:
            print(f"[serial] err: {e}", flush=True)
    elapsed = time.perf_counter() - t0
    close()
    out = {"mode": "serial", "elapsed_s": elapsed, "qps": len(latencies) / elapsed}
    out.update(percentiles(latencies))
    return out


# ---------- Multiprocess concurrent runner ----------

def run_concurrent_one(
    worker_fn: Callable,
    k: int,
    conc: int,
    duration: float,
    vectors: list[list[float]],
    tenants: list[str],
    concurrency_timeout: int = 300,
) -> dict:
    """Spawn `conc` subprocesses, each owns its own client. All workers
    block on a shared Condition until released, then run for `duration`
    seconds. `worker_fn` must be a module-level function importable in
    the spawned subprocess.

    `vectors` and `tenants` are pickled in from the caller and shipped
    to each worker as submit args — workers do NOT re-read parquet
    files. This mirrors VDBBench's mp_runner.py:130 pattern and avoids
    the 80-way parquet-read + client-init thundering herd that would
    otherwise push worker setup past the `concurrency_timeout` bound
    (caught by `q.get(timeout=...)`) at high concurrency."""
    with mp.Manager() as manager:
        q, cond = manager.Queue(), manager.Condition()
        ctx = mp.get_context("spawn")
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=ctx, max_workers=conc
        ) as executor:
            futures = [
                executor.submit(worker_fn, i, duration, k, vectors, tenants, q, cond)
                for i in range(conc)
            ]
            for _ in range(conc):
                q.get(timeout=concurrency_timeout)
            with cond:
                cond.notify_all()
            for f in concurrent.futures.as_completed(
                futures, timeout=duration + concurrency_timeout
            ):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"[conc={conc}] worker error: {e}", flush=True)
    total_count = sum(r[0] for r in results)
    all_lats = [l for r in results for l in r[2]]
    actual_dur = max((r[1] for r in results), default=duration)
    out = {
        "mode": "concurrent", "concurrency": conc,
        "duration_s": actual_dur, "qps": total_count / actual_dur,
    }
    out.update(percentiles(all_lats))
    return out


def run_sweep(
    make_searcher: Callable,
    worker_fn: Callable,
    backend_label: str,
    k: int,
    concurrencies: list[int],
    duration: float,
    serial_n: int | None = None,
) -> dict:
    # Load once in the parent; pickle to workers per submit (matches
    # VDBBench's mp_runner, avoids per-worker parquet reads).
    print("[sweep] loading test vectors + tenant list (once in parent)", flush=True)
    vectors = load_test_vectors()
    tenants = load_tenants()
    print(f"[sweep] loaded {len(vectors)} test vectors, {len(tenants)} tenants", flush=True)

    out = {"backend": backend_label, "k": k, "runs": []}
    if serial_n:
        print(f"\n[serial] {serial_n} queries...", flush=True)
        r = run_serial(make_searcher, k, serial_n, vectors, tenants)
        print(json.dumps(r, indent=2), flush=True)
        out["runs"].append(r)
    for conc in concurrencies:
        print(f"\n[concurrent] conc={conc}, duration={duration}s...", flush=True)
        r = run_concurrent_one(worker_fn, k, conc, duration, vectors, tenants)
        print(json.dumps(r, indent=2), flush=True)
        out["runs"].append(r)
    return out


def print_summary(result: dict) -> None:
    print("\n=== SWEEP SUMMARY ===")
    for item in result["runs"]:
        tag = item.get("mode", "?")
        extra = item.get("concurrency", "")
        print(
            f"  {tag} c={extra}: qps={item.get('qps', 0):.2f}, "
            f"p99={item.get('p99_ms', 0):.0f}ms, n={item.get('n', 0)}"
        )
