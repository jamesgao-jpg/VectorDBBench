"""Cache-eviction probe test (LAION 100M, Turbopuffer).

Every cycle = 1,000 "pressure" queries (diverse, never-repeated) + 100
"probe" queries (fixed set, warmed during baseline, re-measured every
cycle). If probe latency stays flat, TB's cache held the probes'
inverted lists through the pressure. If probe latency rises over time,
pressure has pushed probe-associated data out of cache.

  Probe   : test.parquet[0:100]
  Pressure: train-00-of-100.parquet + train-50-of-100.parquet (2 M
            vectors total, shuffled with fixed seed; file indices 0 and
            50 picked to span LAION's ID range)

Writes nothing. Reads only.

Run:
  .venv/bin/python -u cache_eviction_probe_test.py | tee /tmp/cache_probe.log

Env overrides (additive; defaults match the originally-committed run):
  DURATION_SEC        = 14400   (4 h)
  PRESSURE_PER_CYCLE  = 1000
  PROBE_PER_CYCLE     = 100
  TOP_K               = 100
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
DATA_DIR = Path("/tmp/vectordb_bench/dataset/laion/laion_large_100m")
PRESSURE_FILES = ["train-00-of-100.parquet", "train-50-of-100.parquet"]

DURATION_SEC = int(os.environ.get("DURATION_SEC", 14400))
PRESSURE_PER_CYCLE = int(os.environ.get("PRESSURE_PER_CYCLE", 1000))
PROBE_PER_CYCLE = int(os.environ.get("PROBE_PER_CYCLE", 100))
TOP_K = int(os.environ.get("TOP_K", 100))

OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "cache_eviction_probe_results.json"
PLOT_PATH = OUT_DIR / "cache_eviction_probe.png"


def load_probe_queries() -> np.ndarray:
    df = pl.read_parquet(DATA_DIR / "test.parquet").slice(0, 100)
    vecs = np.stack(df["emb"].to_numpy()).astype(np.float32)
    print(f"[load] probe queries: {vecs.shape[0]} (test.parquet[0:100])", flush=True)
    return vecs


def load_pressure_queries() -> np.ndarray:
    vecs_list = []
    for f in PRESSURE_FILES:
        print(f"[load] reading {f}", flush=True)
        df = pl.read_parquet(DATA_DIR / f)
        vecs_list.append(np.stack(df["emb"].to_numpy()).astype(np.float32))
    vecs = np.concatenate(vecs_list)
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(vecs.shape[0])
    vecs = vecs[idx]
    print(f"[load] pressure queries: {vecs.shape[0]:,} (shuffled, seed=42)", flush=True)
    return vecs


def run_query(ns, q: list[float]) -> int:
    t0 = time.perf_counter()
    res = ns.query(rank_by=("vector", "ANN", q), top_k=TOP_K)
    return int((time.perf_counter() - t0) * 1_000_000)


def pctl(arr_us: np.ndarray) -> dict:
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


def dump(cycles, baseline_probe_stats, p_idx, n_probes, n_pressure):
    result = {
        "namespace": NAMESPACE,
        "duration_sec": DURATION_SEC,
        "pressure_per_cycle": PRESSURE_PER_CYCLE,
        "probe_per_cycle": PROBE_PER_CYCLE,
        "top_k": TOP_K,
        "probe_set_size": n_probes,
        "pressure_set_size": n_pressure,
        "pressure_queries_issued": p_idx,
        "baseline_probe_stats": baseline_probe_stats,
        "cycles": cycles,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2))


def main():
    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    ns = client.namespace(NAMESPACE)

    probes = load_probe_queries()
    pressure = load_pressure_queries()
    n_probes = probes.shape[0]
    n_pressure = pressure.shape[0]

    print(f"\n=== cache-eviction probe — LAION 100M ===", flush=True)
    print(f"  namespace:           {NAMESPACE}", flush=True)
    print(f"  duration:            {DURATION_SEC} s ({DURATION_SEC/3600:.2f} h)", flush=True)
    print(f"  probe set size:      {n_probes}", flush=True)
    print(f"  pressure set size:   {n_pressure:,}", flush=True)
    print(f"  cycle:               {PRESSURE_PER_CYCLE} pressure + {PROBE_PER_CYCLE} probe", flush=True)
    print(f"  top_k:               {TOP_K}", flush=True)

    # Baseline: 3 rounds of probes
    print(f"\n=== Baseline: warming probes (3 rounds) ===", flush=True)
    baseline_lats: list[int] = []
    for rnd in range(3):
        round_lats = []
        for q in probes:
            lat_us = run_query(ns, q.tolist())
            round_lats.append(lat_us)
        baseline_lats.extend(round_lats)
        stats = pctl(np.asarray(round_lats, dtype=np.int64))
        print(f"  [baseline round {rnd+1}] n={stats['n']} mean={stats['mean_ms']:>6.1f} "
              f"p99={stats['p99_ms']:>6.1f} max={stats['max_ms']:>6.1f}", flush=True)

    baseline_probe_stats = pctl(np.asarray(baseline_lats[-n_probes:], dtype=np.int64))
    print(f"[baseline] round-3 probe stats: p99={baseline_probe_stats['p99_ms']} mean={baseline_probe_stats['mean_ms']} ms", flush=True)

    # Main loop
    print(f"\n=== Main loop ===", flush=True)
    cycles: list[dict] = []
    wall_start = time.perf_counter()
    p_idx = 0
    cycle_idx = 0

    while True:
        now = time.perf_counter() - wall_start
        if now >= DURATION_SEC:
            print(f"[cap] {DURATION_SEC}s wall-clock reached at cycle {cycle_idx}", flush=True)
            break

        cycle_start = time.perf_counter()
        pressure_lats = []
        for _ in range(PRESSURE_PER_CYCLE):
            if p_idx >= n_pressure:
                p_idx = 0
            q = pressure[p_idx].tolist()
            p_idx += 1
            pressure_lats.append(run_query(ns, q))
        pressure_dur = time.perf_counter() - cycle_start

        probe_lats = []
        probe_start = time.perf_counter()
        for i in range(PROBE_PER_CYCLE):
            q = probes[i % n_probes].tolist()
            probe_lats.append(run_query(ns, q))
        probe_dur = time.perf_counter() - probe_start

        cycle_idx += 1
        cycle_end = time.perf_counter() - wall_start
        p_arr = np.asarray(pressure_lats, dtype=np.int64)
        probe_arr = np.asarray(probe_lats, dtype=np.int64)
        cycle = {
            "cycle": cycle_idx,
            "t_end_sec": round(cycle_end, 2),
            "pressure_total": p_idx,
            "pressure": pctl(p_arr) | {"dur_sec": round(pressure_dur, 2)},
            "probe": pctl(probe_arr) | {"dur_sec": round(probe_dur, 2)},
        }
        cycles.append(cycle)
        print(f"  [c{cycle_idx:>4}] t+{cycle_end/60:5.1f}min pressure:{p_idx:>8,} "
              f"p_mean={cycle['pressure']['mean_ms']:>6.1f} p_p99={cycle['pressure']['p99_ms']:>6.1f} "
              f"probe_mean={cycle['probe']['mean_ms']:>6.1f} probe_p99={cycle['probe']['p99_ms']:>6.1f} "
              f"probe_max={cycle['probe']['max_ms']:>6.1f}", flush=True)

        if cycle_idx % 10 == 0:
            dump(cycles, baseline_probe_stats, p_idx, n_probes, n_pressure)

    dump(cycles, baseline_probe_stats, p_idx, n_probes, n_pressure)

    if cycles:
        xs = [c["t_end_sec"] / 60 for c in cycles]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(xs, [c["probe"]["p99_ms"] for c in cycles], color="#d62728", linewidth=2, marker="o", markersize=3, label="probe p99")
        ax.plot(xs, [c["probe"]["mean_ms"] for c in cycles], color="#1f77b4", linewidth=2, marker="o", markersize=3, label="probe mean")
        ax.axhline(y=baseline_probe_stats["p99_ms"], color="#d62728", linestyle="--", alpha=0.5, label=f"baseline probe p99 = {baseline_probe_stats['p99_ms']} ms")
        ax.axhline(y=baseline_probe_stats["mean_ms"], color="#1f77b4", linestyle="--", alpha=0.5, label=f"baseline probe mean = {baseline_probe_stats['mean_ms']} ms")
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Probe latency (ms)")
        ax.set_title(f"TB cache-eviction probe — {NAMESPACE}\n"
                     f"probe = 100 fixed queries, re-measured every {PRESSURE_PER_CYCLE} diverse pressure queries")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=110)
        print(f"[plot] saved {PLOT_PATH.name}", flush=True)

    print(f"\n=== Summary ===", flush=True)
    total_elapsed = time.perf_counter() - wall_start
    print(f"  cycles:               {len(cycles)}")
    print(f"  pressure queries:     {p_idx:,}")
    print(f"  duration:             {total_elapsed/60:.2f} min")
    if cycles:
        probe_p99s = [c["probe"]["p99_ms"] for c in cycles]
        print(f"  probe p99 range:      {min(probe_p99s):.1f} – {max(probe_p99s):.1f} ms")
        print(f"  probe p99 mean:       {np.mean(probe_p99s):.1f} ms")
        print(f"  probe p99 std:        {np.std(probe_p99s):.1f} ms")
        print(f"  baseline probe p99:   {baseline_probe_stats['p99_ms']} ms")
        rise = max(probe_p99s) - baseline_probe_stats["p99_ms"]
        print(f"  max rise vs baseline: {rise:+.1f} ms ({rise / baseline_probe_stats['p99_ms'] * 100:+.1f}%)")
    print(f"=== Done ===")


if __name__ == "__main__":
    main()
