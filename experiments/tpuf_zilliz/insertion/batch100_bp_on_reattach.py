"""Re-attach supervisor for the already-running VDBBench batch=100 bp-on test.

The original supervisor (batch100_bp_on_supervisor.py) crashed on its first
metadata call because turbopuffer namespaces are created lazily on first
write, not eagerly — a metadata() call before any write landed returned
404 and uncaught the exception. VDBBench itself kept running after the
supervisor died (start_new_session=True orphaned it successfully).

This script adopts the running VDBBench child:
 - polls ns.metadata() every 60s, records approx_row_count + unindexed_bytes
 - SIGTERM's the child's process group at the original 2-hour wall-clock cap
   (anchored to ORIGINAL_START_UTC below, not to this script's start)
 - polls for drain to 0
 - deletes the namespace
 - dumps full trace + a JSON
"""
from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import turbopuffer as tpuf

API_KEY = "tpuf_cxMjXQvBFFJjCpEGcrJj5SAisrI5ufc4"
REGION = "aws-us-west-2"
NAMESPACE = "laion100m_bp_on_batch100_tpuf"
VDBBENCH_PID = 22132
ORIGINAL_START_UTC = datetime(2026, 4, 19, 10, 59, 12, tzinfo=timezone.utc)
INSERT_WALL_CAP_SEC = 2 * 60 * 60
CAP_AT_UTC = datetime(2026, 4, 19, 12, 59, 12, tzinfo=timezone.utc)

METADATA_POLL_SEC = 60
DRAIN_POLL_SEC = 30
DRAIN_SAFETY_MAX_SEC = 60 * 60

OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "batch100_bp_on_results.json"


def poll_meta_tolerant(ns) -> tuple[int, int] | None:
    try:
        m = ns.metadata()
    except tpuf.NotFoundError:
        return None
    ub = getattr(m.index, "unindexed_bytes", 0) or 0
    rc = m.approx_row_count
    return int(rc), int(ub)


def vdbbench_alive() -> bool:
    try:
        os.kill(VDBBENCH_PID, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    ns = client.namespace(NAMESPACE)

    start_utc = datetime.now(tz=timezone.utc)
    print(f"[reattach] started at {start_utc.isoformat()}", flush=True)
    print(f"[reattach] vdbbench pid={VDBBENCH_PID} original_start={ORIGINAL_START_UTC.isoformat()}", flush=True)
    print(f"[reattach] 2h cap lands at {CAP_AT_UTC.isoformat()}", flush=True)
    if not vdbbench_alive():
        print("[reattach] ERROR: vdbbench not alive; exiting", flush=True)
        return
    print(f"[reattach] vdbbench pid alive ✓", flush=True)

    poll_trace: list[dict] = []
    print(f"\n=== Phase 1: poll metadata until 2h cap ===", flush=True)
    while True:
        now = datetime.now(tz=timezone.utc)
        t_since_orig = (now - ORIGINAL_START_UTC).total_seconds()
        sample = poll_meta_tolerant(ns)
        if sample is None:
            rc, ub = -1, -1
        else:
            rc, ub = sample
        poll_trace.append({
            "utc": now.isoformat(),
            "t_since_original_start_sec": round(t_since_orig, 1),
            "approx_row_count": rc,
            "unindexed_bytes": ub,
            "phase": "insert",
        })
        print(f"  [insert] t+{t_since_orig/60:.1f}min rows={rc:,} unindexed={ub/1e9:.3f} GB "
              f"vdbbench_alive={vdbbench_alive()}", flush=True)

        if not vdbbench_alive():
            print("[reattach] vdbbench exited on its own before cap", flush=True)
            break
        if now >= CAP_AT_UTC:
            print(f"[reattach] 2h cap reached, SIGTERM pid {VDBBENCH_PID}", flush=True)
            try:
                os.killpg(VDBBENCH_PID, signal.SIGTERM)
            except ProcessLookupError:
                print("[reattach] process already gone", flush=True)
            break
        time.sleep(METADATA_POLL_SEC)

    # Wait for child to actually exit (up to 2 min)
    for _ in range(120):
        if not vdbbench_alive():
            break
        time.sleep(1)
    if vdbbench_alive():
        print("[reattach] SIGTERM didn't take; SIGKILL", flush=True)
        try:
            os.killpg(VDBBENCH_PID, signal.SIGKILL)
        except ProcessLookupError:
            pass

    insert_phase_end_utc = datetime.now(tz=timezone.utc)
    insert_phase_sec = (insert_phase_end_utc - ORIGINAL_START_UTC).total_seconds()
    print(f"[insert-end] insert phase total: {insert_phase_sec/60:.2f}min (from vdbbench start)", flush=True)

    print(f"\n=== Phase 2: drain wait (unindexed -> 0) ===", flush=True)
    drain_start = time.perf_counter()
    while True:
        now = datetime.now(tz=timezone.utc)
        sample = poll_meta_tolerant(ns)
        if sample is None:
            rc, ub = -1, -1
        else:
            rc, ub = sample
        t_drain = time.perf_counter() - drain_start
        poll_trace.append({
            "utc": now.isoformat(),
            "t_since_original_start_sec": round((now - ORIGINAL_START_UTC).total_seconds(), 1),
            "approx_row_count": rc,
            "unindexed_bytes": ub,
            "phase": "drain",
        })
        print(f"  [drain] t_drain+{t_drain/60:.2f}min rows={rc:,} unindexed={ub/1e9:.3f} GB", flush=True)
        if ub == 0 and sample is not None:
            print("[drain] unindexed=0 reached", flush=True)
            break
        if t_drain > DRAIN_SAFETY_MAX_SEC:
            print(f"[drain] safety cap {DRAIN_SAFETY_MAX_SEC/60:.0f}min hit", flush=True)
            break
        time.sleep(DRAIN_POLL_SEC)
    drain_elapsed = time.perf_counter() - drain_start

    total_elapsed = (datetime.now(tz=timezone.utc) - ORIGINAL_START_UTC).total_seconds()
    final_sample = poll_meta_tolerant(ns)
    final_rc, final_ub = final_sample if final_sample else (-1, -1)

    # Derive per-interval throughput
    throughput_windows = []
    prev = None
    for s in poll_trace:
        if prev is not None and s["phase"] == "insert" and s["approx_row_count"] >= 0 and prev["approx_row_count"] >= 0:
            dt = s["t_since_original_start_sec"] - prev["t_since_original_start_sec"]
            drc = s["approx_row_count"] - prev["approx_row_count"]
            if dt > 0:
                throughput_windows.append({
                    "t_end_sec": s["t_since_original_start_sec"],
                    "delta_rows": drc,
                    "delta_sec": round(dt, 1),
                    "rate_rows_per_sec": round(drc / dt, 1),
                    "unindexed_bytes": s["unindexed_bytes"],
                })
        prev = s

    result = {
        "namespace": NAMESPACE,
        "batch_size": 100,
        "disable_backpressure": False,
        "num_workers": "VDBBench load_concurrency=0 -> cpu_count = 8",
        "wall_clock_cap_sec": INSERT_WALL_CAP_SEC,
        "original_start_utc": ORIGINAL_START_UTC.isoformat(),
        "reattach_start_utc": start_utc.isoformat(),
        "insert_phase_sec": round(insert_phase_sec, 2),
        "drain_phase_sec": round(drain_elapsed, 2),
        "total_sec": round(total_elapsed, 2),
        "final_row_count": final_rc,
        "final_unindexed_bytes": final_ub,
        "overall_insert_rate_rows_per_sec": round(final_rc / insert_phase_sec, 1) if insert_phase_sec > 0 else 0,
        "poll_trace": poll_trace,
        "throughput_windows": throughput_windows,
        "notes": "Supervisor re-attached to an already-running VDBBench child after "
                 "the original supervisor crashed on a premature metadata() call.",
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {RESULTS_PATH}", flush=True)

    print(f"\n=== Summary ===", flush=True)
    print(f"  rows inserted:    {final_rc:,}")
    print(f"  insert phase:     {insert_phase_sec/60:.2f} min (from VDBBench start)")
    print(f"  drain phase:      {drain_elapsed/60:.2f} min")
    print(f"  total:            {total_elapsed/60:.2f} min")
    if insert_phase_sec > 0 and final_rc > 0:
        print(f"  overall rate:     {final_rc/insert_phase_sec:,.0f} rows/s")

    print(f"\n=== Phase 3: cleanup namespace ===", flush=True)
    try:
        ns.delete_all()
        print(f"[cleanup] deleted {NAMESPACE}", flush=True)
    except Exception as e:
        print(f"[cleanup] {NAMESPACE}: {type(e).__name__}: {e}", flush=True)

    print(f"\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
