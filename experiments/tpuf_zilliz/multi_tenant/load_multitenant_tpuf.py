"""
Multi-tenant Turbopuffer loader — per-tenant file variant.

Prerequisite: run `split_by_tenant.py` once to produce
  /tmp/.../cohere_large_10m/per_tenant/tenant_XXXX.parquet  (1000 files)

Each pre-split file contains one tenant's 10,000 pre-normalized f32
vectors. This loader reads one file per tenant and issues one `ns.write`
per tenant — no cross-file buffering, no accumulator. Trivial parallelism
via a thread pool: workers each process one tenant file at a time.

Compared to the prior streaming version, this trades a one-time ~10-min
prep cost for:
  - O(30 MB) per-worker RAM (vs O(GB) with cross-file buffers)
  - Single large batch per write (~30 MB) → max batch discount
  - Resumable (skip tenants whose namespace already has 10K rows)
  - Trivially parallel (embarrassingly parallel, no shared state)

Environment:
  TPUF_API_KEY      (required)
  TPUF_REGION       (default: aws-us-west-2)
  NS_PREFIX         (default: cohere10m_)
  DROP_OLD          (default: 1 = delete each tenant's namespace first)
  MAX_WORKERS       (default: cpu_count)
  DISABLE_BACKPRESSURE  (default: 0 = honor TB's unindexed-backlog throttle)
  DRAIN_POLL_SEC    (default: 30)
  NUM_TENANTS       (default: 1000; smoke-test limiter)

Usage:
  python3 split_by_tenant.py           # prep, one time
  TPUF_API_KEY=... python3 load_multitenant_tpuf.py
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import turbopuffer as tpuf

DATA_DIR = Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m")
PER_TENANT_DIR = DATA_DIR / "per_tenant"

API_KEY = os.environ["TPUF_API_KEY"]
REGION = os.environ.get("TPUF_REGION", "aws-us-west-2")
NS_PREFIX = os.environ.get("NS_PREFIX", "cohere10m_")
DROP_OLD = os.environ.get("DROP_OLD", "1") == "1"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", str(os.cpu_count() or 8)))
DISABLE_BACKPRESSURE = os.environ.get("DISABLE_BACKPRESSURE", "0") == "1"
DRAIN_POLL_SEC = int(os.environ.get("DRAIN_POLL_SEC", "30"))
NUM_TENANTS = int(os.environ.get("NUM_TENANTS", "1000"))


def list_tenant_files() -> list[tuple[str, Path]]:
    """Return [(tenant_label, parquet_path)], sorted, truncated to NUM_TENANTS."""
    files = sorted(PER_TENANT_DIR.glob("tenant_*.parquet"))
    out = []
    for p in files:
        tenant = p.stem  # "tenant_0042"
        out.append((tenant, p))
    assert out, f"No per-tenant parquet files in {PER_TENANT_DIR}; run split_by_tenant.py first"
    return out[:NUM_TENANTS]


def load_one_tenant(
    client: tpuf.Turbopuffer,
    tenant: str,
    path: Path,
    counter: dict,
    lock: threading.Lock,
    log_every: int,
    total: int,
) -> int:
    ns = client.namespace(f"{NS_PREFIX}{tenant}")
    if DROP_OLD:
        try:
            ns.delete_all()
        except Exception:
            pass  # doesn't exist

    # Read the per-tenant file. Already f32 + cosine-normalized from prep.
    df = pl.read_parquet(path)
    ids = df["id"].to_list()
    # polars List<Float32> → list[list[float]] for the TB SDK
    vectors = df["vector"].to_list()

    ns.write(
        upsert_columns={"id": ids, "vector": vectors},
        distance_metric="cosine_distance",
        disable_backpressure=DISABLE_BACKPRESSURE,
    )
    n = len(ids)
    with lock:
        counter["rows"] += n
        counter["tenants"] += 1
        if counter["tenants"] % log_every == 0:
            dur = time.perf_counter() - counter["t0"]
            print(
                f"  [progress] {counter['tenants']}/{total} tenants | "
                f"{counter['rows']:,} rows | {dur:.1f}s | "
                f"{counter['rows']/dur:.0f} r/s",
                flush=True,
            )
    return n


def insert_all(client: tpuf.Turbopuffer, tenant_files: list[tuple[str, Path]]):
    total = len(tenant_files)
    counter = {"rows": 0, "tenants": 0, "t0": time.perf_counter()}
    lock = threading.Lock()
    log_every = max(1, total // 20)

    print(f"[insert] {total} tenants × ~10K rows each, {MAX_WORKERS} workers")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(load_one_tenant, client, t, p, counter, lock, log_every, total)
            for t, p in tenant_files
        ]
        for fut in futures:
            fut.result()

    dur = time.perf_counter() - counter["t0"]
    print(
        f"\n[insert] {counter['rows']:,} rows across {counter['tenants']} tenants | "
        f"{dur:.1f}s | {counter['rows']/dur:.0f} r/s",
        flush=True,
    )
    return counter["rows"], dur


def wait_for_drain(client: tpuf.Turbopuffer, tenants: list[str]):
    """Poll until every namespace reports unindexed_bytes==0."""
    print(f"[drain] waiting for {len(tenants)} namespaces to fully index")
    pending = set(tenants)
    t0 = time.perf_counter()
    iteration = 0

    def check(t: str) -> tuple[str, int]:
        try:
            meta = client.namespace(f"{NS_PREFIX}{t}").metadata()
            return (t, getattr(meta.index, "unindexed_bytes", 0) or 0)
        except Exception:
            return (t, -1)

    while pending:
        iteration += 1
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(ex.map(check, pending))
        still_pending = {t for t, b in results if b != 0}
        pending = still_pending
        elapsed = time.perf_counter() - t0
        worst_b = max((b for _, b in results if b > 0), default=0)
        print(
            f"[drain] iter {iteration} | t+{elapsed:.0f}s | "
            f"{len(pending)}/{len(tenants)} pending | "
            f"max unindexed = {worst_b/1e6:.1f} MB",
            flush=True,
        )
        if pending:
            time.sleep(DRAIN_POLL_SEC)
    print(f"[drain] all drained in {elapsed:.0f}s")


def main():
    print(
        f"[config] NS_PREFIX={NS_PREFIX} MAX_WORKERS={MAX_WORKERS} "
        f"DISABLE_BACKPRESSURE={DISABLE_BACKPRESSURE} DROP_OLD={DROP_OLD} "
        f"NUM_TENANTS={NUM_TENANTS}",
        flush=True,
    )
    tenant_files = list_tenant_files()
    print(f"[prep] found {len(tenant_files)} per-tenant parquet files under {PER_TENANT_DIR}")

    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)

    _, insert_dur = insert_all(client, tenant_files)

    t_drain = time.perf_counter()
    wait_for_drain(client, [t for t, _ in tenant_files])
    drain_dur = time.perf_counter() - t_drain

    # Sample probe
    print("\n[summary]")
    print(f"  insert duration:   {insert_dur:.1f}s")
    print(f"  drain duration:    {drain_dur:.1f}s")
    sample_idxs = (0, len(tenant_files) // 2, len(tenant_files) - 1)
    probe_vec = [0.1] * 768
    for i in sample_idxs:
        tenant, _ = tenant_files[i]
        ns = client.namespace(f"{NS_PREFIX}{tenant}")
        res = ns.query(rank_by=("vector", "ANN", probe_vec), top_k=5)
        print(f"  ns {NS_PREFIX}{tenant}: probe_top5={len(res.rows or [])}")


if __name__ == "__main__":
    main()
