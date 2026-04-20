"""Turbopuffer insertion-throughput sweep across batch sizes.

Holds corpus / workers / backpressure constant, sweeps batch_size
across [50, 500, 5000]. Per-10% stage throughput is logged for each run.
The 4th datapoint (batch=50,000) is sourced from the 100M run captured
in ../insertion/insertion_test.md.

Usage:
    .venv/bin/python -u batch_size_sweep.py | tee /tmp/batch_sweep.log
"""
from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import turbopuffer as tpuf

API_KEY = "tpuf_cxMjXQvBFFJjCpEGcrJj5SAisrI5ufc4"
REGION = "aws-us-west-2"
DATASET_DIR = Path("/tmp/vectordb_bench/dataset/laion/laion_large_100m")
FILES = ["train-00-of-100.parquet", "train-01-of-100.parquet"]

# Env override: `BATCH_SIZES=50000` runs just that batch size as a
# supplementary datapoint. Default preserves the original 3-way sweep.
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "50,500,5000").split(",")]
NUM_WORKERS = 8
TOTAL_ROWS = 2_000_000
STAGE_COUNT = 10
STAGE_SIZE = TOTAL_ROWS // STAGE_COUNT  # 200k = 10% of corpus

# Reference: batch=50,000 from the 100M LAION run (see ../insertion/insertion_test.md).
# Steady-state per-stage rate collected from that test.
REF_BATCH_50K = {
    "batch_size": 50_000,
    "overall_rate_rows_per_sec": 15_800,
    "source": "../insertion/insertion_test.md (LAION 100M)",
}

OUT_DIR = Path(__file__).parent


def load_2m_rows() -> tuple[np.ndarray, np.ndarray]:
    """Concat files 00 + 01 of LAION 100M into one 2M-row corpus in memory."""
    all_ids = np.empty(TOTAL_ROWS, dtype=np.int64)
    all_vecs = np.empty((TOTAL_ROWS, 768), dtype=np.float32)
    offset = 0
    for f in FILES:
        print(f"[load] reading {f}", flush=True)
        df = pl.read_parquet(DATASET_DIR / f)
        n = df.height
        all_ids[offset:offset + n] = df["id"].to_numpy()
        all_vecs[offset:offset + n] = np.stack(df["emb"].to_numpy())
        offset += n
        del df
        gc.collect()
    print(f"[load] total rows={offset:,} vecs_shape={all_vecs.shape}", flush=True)
    return all_ids[:offset], all_vecs[:offset]


def run_one_batch_size(batch_size: int, ids: np.ndarray, vecs: np.ndarray) -> dict:
    """Insert 2M rows into a fresh namespace at given batch_size; return stage data."""
    namespace = f"laion2m_batch{batch_size}"
    print(f"\n=== batch_size={batch_size} → namespace={namespace} ===", flush=True)

    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    ns = client.namespace(namespace)
    try:
        ns.delete_all()
        print(f"[setup] dropped prior {namespace}", flush=True)
    except Exception as e:
        print(f"[setup] no prior {namespace} ({type(e).__name__})", flush=True)

    num_batches = (TOTAL_ROWS + batch_size - 1) // batch_size
    print(f"[setup] {num_batches:,} batches of {batch_size}, {NUM_WORKERS} workers", flush=True)

    # Stage tracking shared across threads
    rows_written = 0
    stage_idx = 0
    last_stage_time = time.perf_counter()
    stages = []
    lock = threading.Lock()
    errors: list[str] = []
    overall_start = time.perf_counter()

    def submit_batch(start: int, end: int) -> int:
        nonlocal rows_written, stage_idx, last_stage_time
        bids = ids[start:end].tolist()
        bvecs = vecs[start:end].tolist()
        try:
            ns.write(
                upsert_columns={"id": bids, "vector": bvecs},
                distance_metric="euclidean_squared",
                disable_backpressure=True,
            )
            n = end - start
        except Exception as e:
            with lock:
                errors.append(f"{type(e).__name__}: {e}")
            return 0

        with lock:
            rows_written += n
            while stage_idx < STAGE_COUNT and rows_written >= (stage_idx + 1) * STAGE_SIZE:
                now = time.perf_counter()
                stage_elapsed = now - last_stage_time
                stage_rate = STAGE_SIZE / stage_elapsed if stage_elapsed > 0 else 0.0
                stage_idx += 1
                stages.append({
                    "stage_pct": stage_idx * STAGE_COUNT,
                    "rows_at_end": rows_written,
                    "stage_duration_sec": round(stage_elapsed, 3),
                    "stage_rate_rows_per_sec": round(stage_rate, 1),
                })
                print(f"  [{namespace}] {stage_idx * STAGE_COUNT:>3}% rows={rows_written:,} "
                      f"stage_dur={stage_elapsed:.2f}s rate={stage_rate:,.0f} r/s", flush=True)
                last_stage_time = now
        return n

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = []
        for bi in range(num_batches):
            start = bi * batch_size
            end = min(start + batch_size, TOTAL_ROWS)
            futures.append(pool.submit(submit_batch, start, end))
        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # already caught in submit_batch

    overall_elapsed = time.perf_counter() - overall_start
    overall_rate = rows_written / overall_elapsed if overall_elapsed > 0 else 0.0
    requests_per_sec = num_batches / overall_elapsed if overall_elapsed > 0 else 0.0

    result = {
        "batch_size": batch_size,
        "namespace": namespace,
        "num_workers": NUM_WORKERS,
        "target_rows": TOTAL_ROWS,
        "rows_written": rows_written,
        "total_batches": num_batches,
        "overall_duration_sec": round(overall_elapsed, 2),
        "overall_rate_rows_per_sec": round(overall_rate, 1),
        "overall_requests_per_sec": round(requests_per_sec, 2),
        "error_count": len(errors),
        "error_samples": errors[:5],
        "stages": stages,
    }

    result_path = OUT_DIR / f"results_batch{batch_size}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[done] batch={batch_size} rows={rows_written:,} dur={overall_elapsed:.1f}s "
          f"rate={overall_rate:,.0f} r/s requests/s={requests_per_sec:.1f} errors={len(errors)} "
          f"→ {result_path.name}", flush=True)
    return result


def cleanup_namespaces(results: list[dict]) -> None:
    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    for r in results:
        ns = client.namespace(r["namespace"])
        try:
            ns.delete_all()
            print(f"[cleanup] deleted {r['namespace']}", flush=True)
        except Exception as e:
            print(f"[cleanup] {r['namespace']}: {type(e).__name__}: {e}", flush=True)


def plot_results(results: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"50": "#1f77b4", "500": "#2ca02c", "5000": "#d62728"}
    for r in results:
        stages = r["stages"]
        xs = [s["stage_pct"] for s in stages]
        ys = [s["stage_rate_rows_per_sec"] for s in stages]
        key = str(r["batch_size"])
        ax.plot(xs, ys, marker="o", linewidth=2, color=colors.get(key, "black"),
                label=f"batch={r['batch_size']:,}  (avg {r['overall_rate_rows_per_sec']:,.0f} r/s)")
    # Reference line: batch=50k from 100M run
    ax.axhline(y=REF_BATCH_50K["overall_rate_rows_per_sec"], color="#9467bd",
               linestyle="--", linewidth=2,
               label=f"batch=50,000 ref ({REF_BATCH_50K['overall_rate_rows_per_sec']:,} r/s, LAION 100M)")
    ax.set_xlabel("Stage (% of 2M corpus inserted)")
    ax.set_ylabel("Rows / sec (per-stage)")
    ax.set_title("Turbopuffer insertion throughput vs batch size\n"
                 "LAION 2M, 8 workers, disable_backpressure=True")
    ax.set_xticks(list(range(10, 101, 10)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    out = OUT_DIR / "insertion_speed.png"
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"[plot] saved {out.name}", flush=True)
    return out


def main():
    print("=== Phase 1: load 2M LAION rows into RAM ===", flush=True)
    ids, vecs = load_2m_rows()

    print("\n=== Phase 2: insert sweep ===", flush=True)
    results: list[dict] = []
    for bs in BATCH_SIZES:
        results.append(run_one_batch_size(bs, ids, vecs))

    print("\n=== Phase 3: summary ===", flush=True)
    print(f"{'batch':>8} | {'dur (s)':>9} | {'rows/s':>10} | {'req/s':>7} | {'errs':>4}")
    print("-" * 50)
    for r in results:
        print(f"{r['batch_size']:>8} | {r['overall_duration_sec']:>9.1f} | "
              f"{r['overall_rate_rows_per_sec']:>10,.0f} | {r['overall_requests_per_sec']:>7.1f} | "
              f"{r['error_count']:>4}")
    print(f"{REF_BATCH_50K['batch_size']:>8} |      (ref) | "
          f"{REF_BATCH_50K['overall_rate_rows_per_sec']:>10,} | "
          f"~{REF_BATCH_50K['overall_rate_rows_per_sec']/REF_BATCH_50K['batch_size']:>6.2f} |    0   (from insertion/insertion_test.md)")

    print("\n=== Phase 4: plot ===", flush=True)
    plot_results(results)

    print("\n=== Phase 5: cleanup namespaces ===", flush=True)
    cleanup_namespaces(results)

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
