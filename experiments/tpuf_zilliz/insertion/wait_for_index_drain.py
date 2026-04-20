"""Poll turbopuffer namespace metadata until the indexer fully drains.

Reports unindexed_bytes over time. Also reports when the namespace crosses
the 2 GiB strongly-consistent threshold. Exits when unindexed_bytes == 0.
"""
from __future__ import annotations

import sys
import time

import turbopuffer as tpuf

API_KEY = "tpuf_cxMjXQvBFFJjCpEGcrJj5SAisrI5ufc4"
REGION = "aws-us-west-2"

TWO_GIB = 2 * 1024 * 1024 * 1024
POLL_SEC = 30
LOG_EVERY_SEC = 60


def main():
    if len(sys.argv) < 2:
        print("usage: wait_for_index_drain.py <namespace>", file=sys.stderr)
        sys.exit(2)
    ns_name = sys.argv[1]

    client = tpuf.Turbopuffer(api_key=API_KEY, region=REGION)
    ns = client.namespace(ns_name)

    t0 = time.perf_counter()
    strong_ready_at: float | None = None
    last_log = 0.0

    print(f"[drain] waiting for {ns_name} to reach unindexed_bytes=0", flush=True)
    while True:
        meta = ns.metadata()
        unindexed = getattr(meta.index, "unindexed_bytes", 0) or 0
        rows = meta.approx_row_count
        status = meta.index.status
        elapsed = time.perf_counter() - t0

        if strong_ready_at is None and unindexed < TWO_GIB:
            strong_ready_at = elapsed
            print(f"[drain] STRONGLY_CONSISTENT_READY at t={elapsed:.1f}s "
                  f"(unindexed={unindexed/1e9:.3f}GB)", flush=True)

        if unindexed == 0:
            print(f"[drain] FULLY_INDEXED at t={elapsed:.1f}s rows={rows:,}", flush=True)
            return

        now = time.perf_counter()
        if now - last_log >= LOG_EVERY_SEC:
            print(f"[drain] t={elapsed:>7.1f}s unindexed={unindexed/1e9:>6.2f}GB "
                  f"rows={rows:>12,} status={status}", flush=True)
            last_log = now
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
