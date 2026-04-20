"""
Run the VDBBench ZillizCloud._optimize() sequence against an existing
Zilliz collection (flush -> wait_for_index -> compact -> wait ->
wait_for_index -> refresh_load). Use this after an insertion that
didn't already optimize.

Environment (same as load_multitenant_zilliz.py):
  ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_USER (default db_admin),
  ZILLIZ_PASSWORD (optional),
  COLLECTION (default cohere10m_multitenant)
"""
import os
import time

from pymilvus import MilvusClient

URI = os.environ["ZILLIZ_URI"]
TOKEN = os.environ["ZILLIZ_TOKEN"]
USER = os.environ.get("ZILLIZ_USER", "db_admin")
PASSWORD = os.environ.get("ZILLIZ_PASSWORD", "")
COLLECTION = os.environ.get("COLLECTION", "cohere10m_multitenant")
INDEX_NAME = os.environ.get("INDEX_NAME", "vector_idx")


def wait_for_index(client, collection, index_name=INDEX_NAME):
    while True:
        info = client.describe_index(collection, index_name)
        if info is None:
            # pymilvus returns None on INDEX_NOT_FOUND; transient after flush()
            # while index metadata re-registers.
            print("  describe_index returned None (transient INDEX_NOT_FOUND), sleeping 5s", flush=True)
            time.sleep(5)
            continue
        pending = info.get("pending_index_rows", -1)
        if pending == 0:
            return
        print(f"  index pending_index_rows={pending}, sleeping 5s", flush=True)
        time.sleep(5)


def wait_for_compaction(client, compaction_id):
    while True:
        state = client.get_compaction_state(compaction_id)
        if state == "Completed":
            return
        time.sleep(0.5)


def main():
    client = MilvusClient(uri=URI, user=USER, password=PASSWORD, token=TOKEN, timeout=60)

    t0 = time.perf_counter()

    print(f"[optimize] flush {COLLECTION}")
    client.flush(COLLECTION)
    print("[optimize] wait_for_index (pre-compact)")
    wait_for_index(client, COLLECTION)
    try:
        print("[optimize] compact (target_size=2^63-1)")
        compaction_id = client.compact(COLLECTION, target_size=(2**63 - 1))
        if compaction_id > 0:
            print(f"[optimize] wait_for_compaction id={compaction_id}")
            wait_for_compaction(client, compaction_id)
        print("[optimize] wait_for_index (post-compact)")
        wait_for_index(client, COLLECTION)
    except Exception as e:
        print(f"[optimize] compact error: {e} — skipping")
    print("[optimize] refresh_load")
    client.refresh_load(COLLECTION)

    dur = time.perf_counter() - t0

    stats = client.get_collection_stats(COLLECTION)
    idx = client.describe_index(COLLECTION, INDEX_NAME)
    print(f"\n[done] optimize: {dur:.1f}s")
    print(f"  row_count:    {stats.get('row_count')}")
    print(f"  indexed_rows: {idx.get('indexed_rows')}")
    print(f"  state:        {idx.get('state')}")
    client.close()


if __name__ == "__main__":
    main()
