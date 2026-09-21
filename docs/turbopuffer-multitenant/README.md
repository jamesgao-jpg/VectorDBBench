# Run the turbopuffer multi-tenant cold-latency case

This case measures the same 100 out-of-sample queries per namespace in two passes: pass `first` runs every query with turbopuffer's `disable_cache` flag (a genuinely cold, uncached read), then pass `repeat` runs every query again without the flag (warm). It supports dense ANN and BM25 as separate operations over the same wide-row schema.

## Choose the namespace size

Each setup run creates exactly **one** namespace of `--multitenant-namespace-rows` rows (any positive integer up to the prepared file's 5M rows). Run setup once per size you want to compare — e.g. one 1K, one 10K, one 15K, one 5M — each with a fresh prefix and manifest, then search each.

| Option | Default | Meaning |
| --- | --- | --- |
| `--multitenant-namespace-rows` | `15000` | Rows in the single namespace this setup creates |

The namespace is named `{run_prefix}_{rows}_0001` — the row size is part of the name, e.g. `tp_mt_dense_1000_0001`, `tp_mt_dense_15000_0001`, `tp_mt_dense_5000000_0001`.

## Prerequisites

Run commands on the designated remote client in `/home/ubuntu/VectorDBBench-stage1-test`. The prepared source should be:

```text
/home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet
```

Verify it without loading the dataset:

```bash
/home/ubuntu/VectorDBBench/.venv/bin/python -c \
  'import pyarrow.parquet as pq; print(pq.ParquetFile("/home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet").metadata.num_rows)'
```

Expected output:

```text
5000000
```

Set the runtime variables. Reading the API key interactively keeps it out of shell history:

```bash
export VDBBENCH_WORKDIR=/home/ubuntu/VectorDBBench-stage1-test
export VDBBENCH_BIN=/home/ubuntu/VectorDBBench/.venv/bin/vectordbbench
export MULTITENANT_DATA=/home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet
export MULTITENANT_RUN_DIR=/home/ubuntu/vdbbench-data-inspect/turbopuffer-runs
export QUERIES_FILE=/home/ubuntu/vdbbench-data-inspect/turbopuffer-queries/queries.json
export TURBOPUFFER_REGION=aws-us-west-2
read -rsp "Turbopuffer API key: " TURBOPUFFER_API_KEY; echo
export TURBOPUFFER_API_KEY
mkdir -p "$MULTITENANT_RUN_DIR"
cd "$VDBBENCH_WORKDIR"
```

## Dense run

Choose a unique prefix and manifest. Never reuse a prefix that may already exist in turbopuffer.

```bash
export DENSE_PREFIX=tp_mt_dense_20260918
export DENSE_MANIFEST="$MULTITENANT_RUN_DIR/dense.json"
```

Create the namespaces:

```bash
PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation setup \
  --multitenant-manifest "$DENSE_MANIFEST" \
  --multitenant-prepared-data "$MULTITENANT_DATA" \
  --multitenant-queries-file "$QUERIES_FILE" \
  --multitenant-run-prefix "$DENSE_PREFIX" \
  --multitenant-namespace-rows 15000
```

Expected setup summary values: `rows_per_namespace=15000`, `completed_namespaces=1`, `total_namespaces=1`, `total_rows=15000`, and `queries=100`. Setup does not issue measured search queries.

Run dense measurements across every namespace. Each namespace is queried with all 100 queries in pass `first` (each with `disable_cache: true`), then all 100 again in pass `repeat` (no flag):

```bash
PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation dense \
  --multitenant-manifest "$DENSE_MANIFEST" \
  --k 100
```

Expected search summary values are `status=complete`, `rows_per_namespace=15000`, `query_count=100`, and `completed_namespaces=1`. The summary reports two buckets — `first` (the cold pass: all 100 queries per namespace, sent with `disable_cache: true`) and `repeat` (the warm pass: all 100 queries per namespace, no flag) — each with client/server latency min/max/average/P50/P95/P99, outcome counts, and cache metrics (`cache_temperature` counts and `cache_hit_ratio` stats); a top-level `cache_temperature`/`cache_hit_ratio` aggregate covers all queries. Per the task policy, cache cold/warmness is classified strictly from turbopuffer's reported `cache_temperature`/`cache_hit_ratio`; latency metrics are recorded but never used to label cache state. With the flag, pass `first` reports `cold`/`0.0` and pass `repeat` reports `hot`/`1.0`.

## BM25 run

BM25 can reuse the same manifest and namespace as dense — the cold pass is forced by `disable_cache`, so prior dense queries do not contaminate it, and no new setup is needed:

```bash
PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation bm25 \
  --multitenant-manifest "$DENSE_MANIFEST" \
  --k 100
```

Expected BM25 search summary values are `status=complete`, `rows_per_namespace=15000`, and `search_field=content`. For a multi-size comparison, run BM25 on the same manifests as the dense runs (no new setup).

## Search fields and returned attributes

Setup records the authoritative search fields. Their defaults are:

```text
dense: emb_768
BM25:  content
```

Override them only during setup:

```text
--multitenant-dense-field emb_768
--multitenant-bm25-field content
```

Search returns IDs only by default. To exercise selected attributes, add:

```text
--multitenant-output-fields vc_uuid,vc_tag
```

VDBBench validates the returned columns and counts, then discards their values. Query vectors, query text, result IDs, and returned attribute values are not written to the measurement artifact.

A search can be **re-run on the same manifest/namespace with different `--multitenant-output-fields`** (no re-setup, no re-insert): the runner restarts that mode's measurement and writes a per-payload summary named `{manifest}.{mode}.{fields...}.summary.json` (e.g. `dense-5m.dense.vc_uuid-vc_tag.summary.json`), keeping earlier payload summaries intact. Only the returned-attribute set may change; a different mode (`dense` vs `bm25`) or search parameters still requires a separate setup manifest.

## Artifacts

For `dense.json`, setup creates:

```text
dense.json
dense.checkpoints.jsonl
dense.queries.json
```

Dense search creates:

```text
dense.dense.search.jsonl
dense.dense.summary.json
```

The summary separates `first` (the cold pass, `disable_cache: true`) and `repeat` (the warm pass) results across all namespaces. Each bucket contains outcomes plus client, server-total, and query-execution min/max/average/P50/P95/P99, `cache_temperature` counts, and `cache_hit_ratio` stats.

## Cold-warm semantics

Turbopuffer honors an undocumented per-query `disable_cache` request-body flag (verified 2026-09-18 for this account; see the task tracker). Pass `first` sends it, so every first-pass query is a genuinely cold, uncached read reported as `cache_temperature="cold"` / `cache_hit_ratio=0.0`; pass `repeat` omits it and runs warm (`hot`/`1.0`). No idle-eviction wait is needed, and `hint_cache_warm`/namespace pinning are not used by this case. Per the task policy (AGENTS.md), turbopuffer's `cache_temperature`/`cache_hit_ratio` are the SOLE indicators of cache cold/warmness; latency metrics are recorded but never used to classify cache state.

## Resume and failure behavior

Repeat the exact setup command to resume setup. The manifest must match the profile, fields, queries file, prepared file, and prefix. Completed namespaces are skipped.

Repeat the exact search command to resume search. A recorded `started` event without a terminal event becomes `indeterminate` and is never rerun. If a first-pass query failed or became indeterminate, its repeat is marked `skipped`. Other namespaces and other queries continue, and an incomplete invocation still writes its summary before reporting failure.

Measured queries have no benchmark-level or SDK-level retries. The case never deletes namespaces automatically; use a new unique prefix for another run and clean up old namespaces separately after reviewing the artifacts.
