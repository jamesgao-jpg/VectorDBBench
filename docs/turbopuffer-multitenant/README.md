# Run the turbopuffer multi-tenant cold-latency case

This case measures one first query and one immediate repeat query against each namespace, sequentially. It supports dense ANN and BM25 as separate operations over the same wide-row schema.

## Choose a profile

The profile changes only the number of namespaces. Rows per namespace remain 1K, 3K, 15K, and 5M.

| Profile | A/B/C namespaces each | D namespaces | Total namespaces | Total rows |
| --- | ---: | ---: | ---: | ---: |
| `small` | 20 | 1 | 61 | 5.38M |
| `medium` | 100 | 1 | 301 | 6.9M |
| `large` | 300 | 1 | 901 | 10.7M |

Start with `small` for the live pilot. Use `medium` for the normal comparison. Use `large` only when Medium's confidence intervals are too wide. P99 is descriptive rather than an acceptance metric because there are too few tail observations even in Large.

### Skip the 5M-row namespace

Add `--multitenant-exclude-5m` to the **setup** command to skip the single 5M-row D namespace. The A/B/C namespace counts are unchanged, so the profile reports as `small-no-5m`, `medium-no-5m`, or `large-no-5m`:

| Profile + flag | Namespaces | Total rows |
| --- | ---: | ---: |
| `small` + `--multitenant-exclude-5m` | 60 | 380K |
| `medium` + `--multitenant-exclude-5m` | 300 | 1.9M |
| `large` + `--multitenant-exclude-5m` | 900 | 5.7M |

Setup summary for the flagged Small run is `profile=small-no-5m`, `completed_namespaces=60`, `total_namespaces=60`, and `total_rows=380000`. Search commands stay unchanged and read the namespaces from the manifest; do not pass `--multitenant-group D` for a manifest created with the flag.

### Shared out-of-sample query set

Every namespace is measured with the same 100 out-of-sample queries: pass `first` runs all 100 (query 0 is the cold sample, queries 1-99 are the warm-up ramp), then pass `repeat` runs all 100 again while the namespace is warm. The queries come from a wide-table Parquet that was NOT used to build the prepared insert file, so no query vector exists inside any namespace.

Extract them once from any unused source file (e.g. `wide_table_0010.parquet`, which is beyond the 5M rows used for inserts):

```bash
export QUERIES_FILE=/home/ubuntu/vdbbench-data-inspect/turbopuffer-queries/queries.json
/home/ubuntu/VectorDBBench/.venv/bin/python scripts/extract_turbopuffer_multitenant_queries.py \
  --input /home/ubuntu/vdbbench-data-inspect/turbopuffer-queries/wide_table_0010.parquet \
  --output "$QUERIES_FILE" --count 100
```

The setup operation records the query set in the manifest (version 5) and copies it to a `<manifest>.queries.json` sidecar, so dense and BM25 search commands need no query arguments.

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
export DENSE_PREFIX=tp_mt_dense_small_20260916
export DENSE_MANIFEST="$MULTITENANT_RUN_DIR/dense-small.json"
```

Create the Small profile:

```bash
PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation setup \
  --multitenant-profile small \
  --multitenant-manifest "$DENSE_MANIFEST" \
  --multitenant-prepared-data "$MULTITENANT_DATA" \
  --multitenant-queries-file "$QUERIES_FILE" \
  --multitenant-run-prefix "$DENSE_PREFIX"
```

Add `--multitenant-exclude-5m` to skip the 5M-row D namespace for a cheaper pilot.

Expected setup summary values for Small without the flag are `profile=small`, `completed_namespaces=61`, `total_namespaces=61`, `total_rows=5380000`, and `queries=100`; with `--multitenant-exclude-5m` they are `profile=small-no-5m`, `completed_namespaces=60`, `total_rows=380000`. Setup does not issue measured search queries.

Run dense measurements across every group. Each namespace is queried with all 100 queries in pass `first`, then all 100 again in pass `repeat`:

```bash
PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation dense \
  --multitenant-manifest "$DENSE_MANIFEST" \
  --multitenant-group all \
  --k 100
```

Expected search summary values are `status=complete`, `profile=small`, `query_count=100`, `namespace_count=61`, and `completed_namespaces=61`. The per-group summary reports three buckets per distribution — `cold` (query 0 of pass `first`, one genuinely-cold sample per namespace), `first` (the full cold pass: all 100 queries), and `repeat` (the warm pass: all 100 queries) — each with client/server latency min/max/average/P50/P95/P99 and outcome counts. Only query 0 of pass `first` is truly cold; the remaining 99 first-pass queries and all repeat queries run against the warmed namespace, so the cold signal appears in the `cold` bucket and in the `first` bucket's max/P99 rather than its P50.

To limit one invocation's runtime, select one group instead:

```bash
# Replace A with B, C, or D for later invocations.
PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation dense \
  --multitenant-manifest "$DENSE_MANIFEST" \
  --multitenant-group A \
  --k 100
```

All group invocations share the same dense JSONL checkpoint. A later `--multitenant-group all` invocation skips completed namespaces and processes only the remainder.

## BM25 run

BM25 needs a fresh prefix and manifest if its first query is to be treated as cold. A dense query may already have changed the namespace cache state.

```bash
export BM25_PREFIX=tp_mt_bm25_small_20260916
export BM25_MANIFEST="$MULTITENANT_RUN_DIR/bm25-small.json"

PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation setup \
  --multitenant-profile small \
  --multitenant-manifest "$BM25_MANIFEST" \
  --multitenant-prepared-data "$MULTITENANT_DATA" \
  --multitenant-queries-file "$QUERIES_FILE" \
  --multitenant-run-prefix "$BM25_PREFIX"

PYTHONPATH="$VDBBENCH_WORKDIR" "$VDBBENCH_BIN" turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" \
  --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation bm25 \
  --multitenant-manifest "$BM25_MANIFEST" \
  --multitenant-group all \
  --k 100
```

Expected BM25 search summary values are `status=complete`, `profile=small`, and `search_field=content`.

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

## Artifacts

For `dense-small.json`, setup creates:

```text
dense-small.json
dense-small.checkpoints.jsonl
dense-small.queries.json
```

Dense search creates:

```text
dense-small.dense.search.jsonl
dense-small.dense.all.summary.json
```

A group-specific run writes a summary such as `dense-small.dense.a.summary.json` while sharing `dense-small.dense.search.jsonl`.

The summary separates `cold` (query 0 of pass `first`, one sample per namespace), `first` (the full cold pass), and `repeat` (the warm pass) results for every namespace-size group. Each bucket contains outcomes plus client, server-total, and query-execution min/max/average/P50/P95/P99.

## Cold-wait note

Turbopuffer has no cache-eviction API (`hint_cache_warm` and namespace pinning only warm). A namespace is `cold` only after the service evicts it from cache while idle, so wait after setup before searching — otherwise the "cold" query is already `hot` (as the first pilot observed). Every event records the backend-reported `cache_temperature`/`cache_hit_ratio`; verify the `cold` bucket's query-0 events report `cold` before trusting the cold latencies, and extend the idle wait if they are `hot`.

## Resume and failure behavior

Repeat the exact setup command to resume setup. The manifest must match the profile, fields, queries file, prepared file, and prefix. Completed namespaces are skipped.

Repeat the exact search command to resume search. A recorded `started` event without a terminal event becomes `indeterminate` and is never rerun. If a first-pass query failed or became indeterminate, its repeat is marked `skipped`. Other namespaces and other queries continue, and an incomplete invocation still writes its summary before reporting failure.

Measured queries have no benchmark-level or SDK-level retries. The case never deletes namespaces automatically; use a new unique prefix for another cold run and clean up old namespaces separately after reviewing the artifacts.
