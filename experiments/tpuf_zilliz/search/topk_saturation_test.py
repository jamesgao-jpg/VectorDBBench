"""top_k saturation test for turbopuffer under narrow int filter.

For each top_k in [100, 500, 1200], run all 1000 test queries with
filter_rate=0.999 (id >= 99,900,000 → 100k eligible rows) and record
how many results are actually returned per query.

Zilliz blog claim: turbopuffer returns ~500 when top_k=1200 under
extreme filters because the ANN candidate pool is exhausted after
post-filtering.

Usage:
    .venv/bin/python -u topk_saturation_test.py | tee /tmp/topk_sat.log
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
NAMESPACE = "laion100m_bulk"
TEST_PATH = Path("/tmp/vectordb_bench/dataset/laion/laion_large_100m/test.parquet")

# Env overrides let us rerun the same protocol at different filter rates / k values
# without duplicating the script (e.g. FILTER_RATE=0.99 TOP_KS=1200 to probe
# intermediate selectivity).
FILTER_RATE = float(os.environ.get("FILTER_RATE", 0.999))
FILTER_VALUE = int(FILTER_RATE * 100_000_000)
TOP_KS = [int(x) for x in os.environ.get("TOP_KS", "100,500,1200").split(",")]

_tag = f"_filter{FILTER_RATE}_k{'-'.join(map(str, TOP_KS))}" if os.environ.get("FILTER_RATE") or os.environ.get("TOP_KS") else ""
RESULTS_PATH = Path(__file__).parent / f"topk_saturation_results{_tag}.json"


def main():
    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    ns = client.namespace(NAMESPACE)

    # Verify namespace is ready
    meta = ns.metadata()
    unindexed = getattr(meta.index, "unindexed_bytes", 0) or 0
    print(f"Namespace: {NAMESPACE}", flush=True)
    print(f"  rows: {meta.approx_row_count:,}", flush=True)
    print(f"  unindexed: {unindexed/1e9:.3f} GB", flush=True)
    print(f"  filter: id >= {FILTER_VALUE} (rate={FILTER_RATE})", flush=True)

    # Load test queries
    df = pl.read_parquet(TEST_PATH)
    queries = np.stack(df["emb"]).tolist()  # LAION is L2, no normalization
    print(f"  test queries: {len(queries)}\n", flush=True)

    results = {
        "namespace": NAMESPACE,
        "filter_rate": FILTER_RATE,
        "filter_value": FILTER_VALUE,
        "n_queries": len(queries),
        "sweeps": [],
    }

    for k in TOP_KS:
        print(f"=== top_k={k} ===", flush=True)
        counts = []
        latencies = []

        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            res = ns.query(
                rank_by=("vector", "ANN", q),
                top_k=k,
                filters=("id", "Gte", FILTER_VALUE),
            )
            lat = time.perf_counter() - t0
            n = len(res.rows) if res.rows else 0
            counts.append(n)
            latencies.append(lat)

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(queries)}] last_count={n} last_lat={lat*1000:.0f}ms", flush=True)

        counts_arr = np.array(counts)
        lat_arr = np.array(latencies)

        sweep = {
            "top_k": k,
            "returned_min": int(counts_arr.min()),
            "returned_median": int(np.median(counts_arr)),
            "returned_max": int(counts_arr.max()),
            "returned_mean": round(float(counts_arr.mean()), 1),
            "returned_lt_k_count": int((counts_arr < k).sum()),
            "returned_lt_k_pct": round(float((counts_arr < k).sum() / len(counts_arr) * 100), 1),
            "latency_avg_ms": round(float(lat_arr.mean() * 1000), 1),
            "latency_p95_ms": round(float(np.percentile(lat_arr, 95) * 1000), 1),
            "latency_p99_ms": round(float(np.percentile(lat_arr, 99) * 1000), 1),
        }
        results["sweeps"].append(sweep)

        print(f"  returned: min={sweep['returned_min']} med={sweep['returned_median']} "
              f"max={sweep['returned_max']} mean={sweep['returned_mean']}", flush=True)
        print(f"  under-saturated: {sweep['returned_lt_k_count']}/{len(queries)} "
              f"({sweep['returned_lt_k_pct']}%) queries returned < {k}", flush=True)
        print(f"  latency: avg={sweep['latency_avg_ms']}ms "
              f"p95={sweep['latency_p95_ms']}ms p99={sweep['latency_p99_ms']}ms\n", flush=True)

    # Summary
    print("=== SUMMARY ===", flush=True)
    print(f"{'top_k':>6} | {'min':>5} | {'med':>5} | {'max':>5} | {'mean':>6} | {'<k queries':>12} | {'avg_ms':>7}", flush=True)
    print("-" * 70, flush=True)
    for s in results["sweeps"]:
        print(f"{s['top_k']:>6} | {s['returned_min']:>5} | {s['returned_median']:>5} | "
              f"{s['returned_max']:>5} | {s['returned_mean']:>6} | "
              f"{s['returned_lt_k_count']:>4}/{len(queries)} ({s['returned_lt_k_pct']:>4}%) | "
              f"{s['latency_avg_ms']:>7}", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
