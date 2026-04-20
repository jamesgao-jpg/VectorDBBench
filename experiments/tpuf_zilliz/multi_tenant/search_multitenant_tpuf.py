"""
Multi-tenant search driver — Turbopuffer backend.

One namespace per tenant. Before each query we look up the namespace
handle (cached per-worker), then issue a top-K ANN query against it.

Optional namespace warming:
  TPUF_WARM=1 calls `ns.hint_cache_warm()` on every selected tenant
  before measurements start. This is TB's API for signalling "prepare
  for low-latency requests" — useful before measuring steady-state
  warm QPS so results aren't dragged down by first-touch cold paths.

Environment:
  TPUF_API_KEY        (required)
  TPUF_REGION         (default: aws-us-west-2)
  NAMESPACE_PREFIX    (default: cohere10m_tenant_)
  TPUF_WARM           (default: 0 = skip warming; set 1 to pre-warm)
  TPUF_WARM_WORKERS   (default: 16; thread pool for the warming pass)
  TPUF_WARM_SLEEP     (default: 0; seconds to sleep after warming
                       before running measurements — gives TB a moment
                       to finish loading state into caches)

Common knobs (shared with search_multitenant_zilliz.py):
  MODE            = serial | concurrent | sweep   (default: sweep)
  SERIAL_N        = queries for serial leg        (default: 1000)
  CONCURRENCIES   = csv list of concurrency levels (default: 1,5,20,40,60,80)
  DURATION        = seconds per concurrency level (default: 30)
  K               = top-K                         (default: 100)
  OUTPUT          = path for results JSON         (optional)

Usage:
  TPUF_API_KEY=... TPUF_WARM=1 python3 search_multitenant_tpuf.py
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from search_multitenant_common import (
    load_test_vectors,
    load_tenants,
    percentiles,
    print_summary,
    run_concurrent_one,
    run_serial,
    run_sweep,
)

NAMESPACE_PREFIX = os.environ.get("NAMESPACE_PREFIX", "cohere10m_tenant_")
TPUF_WARM = os.environ.get("TPUF_WARM", "0") == "1"
TPUF_WARM_WORKERS = int(os.environ.get("TPUF_WARM_WORKERS", "16"))
TPUF_WARM_SLEEP = float(os.environ.get("TPUF_WARM_SLEEP", "0"))


def _client():
    import turbopuffer as tpuf
    return tpuf.Turbopuffer(
        api_key=os.environ["TPUF_API_KEY"],
        region=os.environ.get("TPUF_REGION", "aws-us-west-2"),
    )


def make_searcher(k: int):
    """Returns (search_fn, close_fn). Called once per process/worker."""
    client = _client()
    ns_cache: dict[str, object] = {}

    def search(vec, tenant_id):
        ns = ns_cache.get(tenant_id)
        if ns is None:
            ns = client.namespace(f"{NAMESPACE_PREFIX}{tenant_id}")
            ns_cache[tenant_id] = ns
        return ns.query(
            rank_by=("vector", "ANN", vec),
            top_k=k,
            include_attributes=[],
        )

    return search, lambda: None


def _worker(worker_id: int, duration: float, k: int,
            vectors: list[list[float]], tenants: list[str], q, cond):
    """ProcessPoolExecutor worker. Must be module-level for spawn pickling.
    Startup ordering matches VDBBench's mp_runner.search_by_dur: signal
    ready IMMEDIATELY (before heavy init) so the parent's `q.get()` can
    drain fast even at conc=80. Client creation happens AFTER the
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


def warm_all_namespaces(tenants: list[str]):
    """Call ns.hint_cache_warm() on every tenant's namespace in parallel.
    Best-effort: errors are logged but don't abort the run."""
    client = _client()
    errors = 0

    def warm_one(t: str):
        nonlocal errors
        try:
            client.namespace(f"{NAMESPACE_PREFIX}{t}").hint_cache_warm()
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"[warm] {t}: {e}", flush=True)

    t0 = time.perf_counter()
    print(f"[warm] hinting {len(tenants)} namespaces (workers={TPUF_WARM_WORKERS})", flush=True)
    with ThreadPoolExecutor(max_workers=TPUF_WARM_WORKERS) as ex:
        list(ex.map(warm_one, tenants))
    dur = time.perf_counter() - t0
    print(
        f"[warm] done in {dur:.1f}s ({len(tenants)/dur:.0f} hints/s, "
        f"{errors} errors)",
        flush=True,
    )
    if TPUF_WARM_SLEEP > 0:
        print(f"[warm] sleeping {TPUF_WARM_SLEEP}s for cache settle", flush=True)
        time.sleep(TPUF_WARM_SLEEP)


def main():
    mode = os.environ.get("MODE", "sweep")
    k = int(os.environ.get("K", "100"))
    duration = float(os.environ.get("DURATION", "30"))
    serial_n = int(os.environ.get("SERIAL_N", "1000"))
    concs = [int(x) for x in os.environ.get("CONCURRENCIES", "1,5,20,40,60,80").split(",")]
    output = os.environ.get("OUTPUT")

    print(
        f"backend=turbopuffer mode={mode} k={k} duration={duration} "
        f"concs={concs} warm={TPUF_WARM} warm_sleep={TPUF_WARM_SLEEP}",
        flush=True,
    )

    if TPUF_WARM:
        warm_all_namespaces(load_tenants())

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
        r = run_sweep(make_searcher, _worker, "turbopuffer", k, concs, duration, serial_n=serial_n)
        print_summary(r)
        if output:
            Path(output).write_text(json.dumps(r, indent=2))
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
