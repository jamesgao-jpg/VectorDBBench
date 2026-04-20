"""Serial search latency stability test — LAION 100M (TB-only).

Measures how TB's serial-search latency drifts and jitters over an extended
window. Single worker, back-to-back queries, no concurrency, no payload
retrieval. When the query set is exhausted it loops from index 0.

Output:
  - per-query trace (timestamp, latency_us, returned_count)
  - windowed summary (p50 / p95 / p99 / max / mean / QPS) every WINDOW_QUERIES
    or WINDOW_SEC, whichever hits first
  - PNG plot of window p99 / p95 / mean over elapsed time

Run:
  .venv/bin/python -u serial_stability_test.py | tee /tmp/serial_stability.log

Env overrides (additive, defaults match the originally-committed run):
  DURATION_SEC      = 3600    (1 h wall-clock cap)
  WINDOW_QUERIES    = 1000
  WINDOW_SEC        = 120
  TOP_K             = 100
  FILTER_RATE       = ""       (unfiltered by default; set e.g. 0.9)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import turbopuffer as tpuf

API_KEY = "tpuf_cxMjXQvBFFJjCpEGcrJj5SAisrI5ufc4"
REGION = "aws-us-west-2"
NAMESPACE = "laion100m_bulk"
QUERY_PARQUET = Path("/tmp/vectordb_bench/dataset/laion/laion_large_100m/train-00-of-100.parquet")

DURATION_SEC = int(os.environ.get("DURATION_SEC", 3600))
WINDOW_QUERIES = int(os.environ.get("WINDOW_QUERIES", 1000))
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", 120))
TOP_K = int(os.environ.get("TOP_K", 100))
FILTER_RATE = os.environ.get("FILTER_RATE", "").strip()
FILTER_VALUE = int(float(FILTER_RATE) * 100_000_000) if FILTER_RATE else None

OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "serial_stability_results.json"
PLOT_PATH = OUT_DIR / "serial_stability.png"


def load_queries() -> tuple[np.ndarray, np.ndarray]:
    print(f"[load] reading {QUERY_PARQUET.name}", flush=True)
    df = pl.read_parquet(QUERY_PARQUET)
    ids = df["id"].to_numpy()
    vecs = np.stack(df["emb"].to_numpy()).astype(np.float32)
    print(f"[load] {len(ids):,} queries, vec shape {vecs.shape}", flush=True)
    return ids, vecs


def percentiles(arr_us: np.ndarray) -> dict:
    if arr_us.size == 0:
        return {"n": 0}
    return {
        "n": int(arr_us.size),
        "mean_ms": round(float(arr_us.mean()) / 1000.0, 3),
        "p50_ms": round(float(np.percentile(arr_us, 50)) / 1000.0, 3),
        "p95_ms": round(float(np.percentile(arr_us, 95)) / 1000.0, 3),
        "p99_ms": round(float(np.percentile(arr_us, 99)) / 1000.0, 3),
        "max_ms": round(float(arr_us.max()) / 1000.0, 3),
        "min_ms": round(float(arr_us.min()) / 1000.0, 3),
    }


def main():
    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    ns = client.namespace(NAMESPACE)
    _, vecs = load_queries()
    n_q = len(vecs)

    print(f"\n=== serial stability run ===", flush=True)
    print(f"  namespace:   {NAMESPACE}", flush=True)
    print(f"  duration:    {DURATION_SEC} s ({DURATION_SEC/60:.1f} min)", flush=True)
    print(f"  window:      {WINDOW_QUERIES} queries or {WINDOW_SEC} s", flush=True)
    print(f"  top_k:       {TOP_K}", flush=True)
    print(f"  filter:      {'id >= ' + str(FILTER_VALUE) if FILTER_VALUE else 'none'}", flush=True)
    print(f"  query set:   {QUERY_PARQUET.name} ({n_q:,} vectors, loop-on-exhaust)", flush=True)

    latencies_us: list[int] = []          # full trace
    windows: list[dict] = []              # windowed summaries
    wall_start = time.perf_counter()
    win_start_elapsed = 0.0
    win_latencies: list[int] = []
    q_idx = 0

    def flush_window():
        nonlocal win_start_elapsed, win_latencies
        if not win_latencies:
            return
        arr = np.asarray(win_latencies, dtype=np.int64)
        now_elapsed = time.perf_counter() - wall_start
        win_dur = now_elapsed - win_start_elapsed
        win_qps = arr.size / win_dur if win_dur > 0 else 0.0
        stats = percentiles(arr)
        stats["t_end_sec"] = round(now_elapsed, 2)
        stats["window_dur_sec"] = round(win_dur, 2)
        stats["qps"] = round(win_qps, 2)
        windows.append(stats)
        print(f"  [win] t+{now_elapsed/60:5.1f}min  n={stats['n']:>4} qps={win_qps:>6.1f} "
              f"p50={stats['p50_ms']:>5.1f} p95={stats['p95_ms']:>6.1f} p99={stats['p99_ms']:>6.1f} "
              f"max={stats['max_ms']:>6.1f} mean={stats['mean_ms']:>5.1f} ms", flush=True)
        win_latencies = []
        win_start_elapsed = now_elapsed

    # Main loop
    try:
        while True:
            now = time.perf_counter() - wall_start
            if now >= DURATION_SEC:
                print(f"[cap] {DURATION_SEC}s wall-clock reached", flush=True)
                break
            q = vecs[q_idx % n_q].tolist()
            q_idx += 1

            query_kwargs = {"rank_by": ("vector", "ANN", q), "top_k": TOP_K}
            if FILTER_VALUE is not None:
                query_kwargs["filters"] = ("id", "Gte", FILTER_VALUE)

            t0 = time.perf_counter()
            try:
                res = ns.query(**query_kwargs)
                lat_us = int((time.perf_counter() - t0) * 1_000_000)
                _ = len(res.rows or [])
            except Exception as e:
                print(f"  [error] q#{q_idx}: {type(e).__name__}: {e}", flush=True)
                continue

            latencies_us.append(lat_us)
            win_latencies.append(lat_us)

            # Window flush: query-count OR time
            win_elapsed = (time.perf_counter() - wall_start) - win_start_elapsed
            if len(win_latencies) >= WINDOW_QUERIES or win_elapsed >= WINDOW_SEC:
                flush_window()
    except KeyboardInterrupt:
        print("[interrupt] flushing final window", flush=True)

    # Final partial window
    flush_window()

    overall = percentiles(np.asarray(latencies_us, dtype=np.int64))
    total_elapsed = time.perf_counter() - wall_start
    overall["t_end_sec"] = round(total_elapsed, 2)
    overall["qps"] = round(len(latencies_us) / total_elapsed, 2) if total_elapsed > 0 else 0

    result = {
        "namespace": NAMESPACE,
        "duration_sec": DURATION_SEC,
        "window_queries": WINDOW_QUERIES,
        "window_sec": WINDOW_SEC,
        "top_k": TOP_K,
        "filter_rate": FILTER_RATE or None,
        "filter_value": FILTER_VALUE,
        "query_parquet": QUERY_PARQUET.name,
        "query_pool_size": n_q,
        "q_idx_final": q_idx,
        "loops_completed": q_idx // n_q,
        "total_queries": len(latencies_us),
        "total_elapsed_sec": round(total_elapsed, 2),
        "overall": overall,
        "windows": windows,
        "latencies_us": latencies_us,   # full trace
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {RESULTS_PATH}", flush=True)

    # Plot
    if windows:
        xs = [w["t_end_sec"] / 60 for w in windows]
        ax_kwargs = {"linewidth": 2, "marker": "o", "markersize": 3}
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(xs, [w["p99_ms"] for w in windows], color="#d62728", label="p99",  **ax_kwargs)
        ax.plot(xs, [w["p95_ms"] for w in windows], color="#ff7f0e", label="p95",  **ax_kwargs)
        ax.plot(xs, [w["mean_ms"] for w in windows], color="#1f77b4", label="mean", **ax_kwargs)
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Latency per window (ms)")
        ax.set_title(f"Turbopuffer serial-search latency stability — {NAMESPACE}\n"
                     f"top_k={TOP_K}, {'filter=id>='+str(FILTER_VALUE) if FILTER_VALUE else 'unfiltered'}, "
                     f"window={WINDOW_QUERIES}q/{WINDOW_SEC}s")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=110)
        print(f"[plot] saved {PLOT_PATH.name}", flush=True)

    print(f"\n=== Summary ===")
    print(f"  total queries:   {len(latencies_us):,}  ({q_idx // n_q} full loops of {n_q:,})")
    print(f"  duration:        {total_elapsed/60:.2f} min")
    print(f"  overall QPS:     {overall['qps']}")
    print(f"  overall mean:    {overall['mean_ms']} ms")
    print(f"  overall p95:     {overall['p95_ms']} ms")
    print(f"  overall p99:     {overall['p99_ms']} ms")
    print(f"  overall max:     {overall['max_ms']} ms")
    if windows:
        p99s = [w["p99_ms"] for w in windows]
        print(f"  window p99 range: {min(p99s):.1f} – {max(p99s):.1f} ms  "
              f"(std {np.std(p99s):.1f})")
    print("=== Done ===")


if __name__ == "__main__":
    main()
