"""
Multi-tenant search driver — Zilliz Cloud backend (pymilvus).

Single collection with `labels` as `is_partition_key=True`. Each query
supplies `filter='labels == "tenant_XXXX"'`, which Zilliz uses to route
to the appropriate partition (via partition_key's hash).

Environment:
  ZILLIZ_URI       (required)
  ZILLIZ_TOKEN     (required)
  ZILLIZ_USER      (default: db_admin)
  ZILLIZ_PASSWORD  (optional)
  COLLECTION       (default: cohere10m_multitenant)

Common knobs (shared with search_multitenant_tpuf.py):
  MODE            = serial | concurrent | sweep   (default: sweep)
  SERIAL_N        = queries for serial leg        (default: 1000)
  CONCURRENCIES   = csv list of concurrency levels (default: 1,5,20,40,60,80)
  DURATION        = seconds per concurrency level (default: 30)
  K               = top-K                         (default: 100)
  OUTPUT          = path for results JSON         (optional)

Usage:
  ZILLIZ_URI=... ZILLIZ_TOKEN=... python3 search_multitenant_zilliz.py
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from search_multitenant_common import (
    load_test_vectors,
    load_tenants,
    print_summary,
    run_concurrent_one,
    run_serial,
    run_sweep,
)

COLLECTION = os.environ.get("COLLECTION", "cohere10m_multitenant")


def make_searcher(k: int):
    """Returns (search_fn, close_fn). Called once per process/worker."""
    from pymilvus import MilvusClient
    client = MilvusClient(
        uri=os.environ["ZILLIZ_URI"],
        user=os.environ.get("ZILLIZ_USER", "db_admin"),
        password=os.environ.get("ZILLIZ_PASSWORD", ""),
        token=os.environ["ZILLIZ_TOKEN"],
        timeout=60,
    )

    def search(vec, tenant_id):
        return client.search(
            COLLECTION, [vec], limit=k,
            search_params={"metric_type": "COSINE"},
            filter=f'labels == "{tenant_id}"',
            output_fields=["pk"],
        )

    return search, client.close


def _worker(worker_id: int, duration: float, k: int,
            vectors: list[list[float]], tenants: list[str], q, cond):
    """ProcessPoolExecutor worker. Must be module-level for spawn pickling.
    Startup ordering matches VDBBench's mp_runner.search_by_dur: signal
    ready IMMEDIATELY (before heavy init) so the parent's `q.get()` can
    drain fast even at conc=80. Client creation + TLS happens AFTER the
    starting gun fires."""
    # Ready signal first — keeps queue-fill timeout trivial at any conc
    q.put(1)
    with cond:
        cond.wait()

    # Heavy init only after the starting gun
    search, close = make_searcher(k)

    rng = random.Random(worker_id * 7919 + 1)
    n = len(vectors)
    n_ten = len(tenants)

    t0 = time.perf_counter()
    count = 0
    latencies: list[float] = []
    while time.perf_counter() < t0 + duration:
        vec = vectors[count % n]
        tenant = tenants[rng.randrange(n_ten)]
        s = time.perf_counter()
        try:
            search(vec, tenant)
            count += 1
            latencies.append(time.perf_counter() - s)
        except Exception as e:
            print(f"[worker {worker_id}] error: {e}", flush=True)
    total = time.perf_counter() - t0
    close()
    return count, total, latencies


def main():
    mode = os.environ.get("MODE", "sweep")
    k = int(os.environ.get("K", "100"))
    duration = float(os.environ.get("DURATION", "30"))
    serial_n = int(os.environ.get("SERIAL_N", "1000"))
    concs = [int(x) for x in os.environ.get("CONCURRENCIES", "1,5,20,40,60,80").split(",")]
    output = os.environ.get("OUTPUT")

    print(
        f"backend=zilliz collection={COLLECTION} mode={mode} k={k} "
        f"duration={duration} concs={concs}",
        flush=True,
    )

    if mode == "serial":
        vectors = load_test_vectors()
        tenants = load_tenants()
        r = run_serial(make_searcher, k, serial_n, vectors, tenants)
        print(json.dumps(r, indent=2))
        if output:
            Path(output).write_text(json.dumps(r, indent=2))
    elif mode == "concurrent":
        vectors = load_test_vectors()
        tenants = load_tenants()
        r = run_concurrent_one(_worker, k, concs[0], duration, vectors, tenants)
        print(json.dumps(r, indent=2))
        if output:
            Path(output).write_text(json.dumps(r, indent=2))
    elif mode == "sweep":
        r = run_sweep(make_searcher, _worker, "zilliz", k, concs, duration, serial_n=serial_n)
        print_summary(r)
        if output:
            Path(output).write_text(json.dumps(r, indent=2))
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
