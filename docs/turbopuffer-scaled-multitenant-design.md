# Turbopuffer multi-tenant cold-start experiment

**Rationale:** [Turbopuffer multi-tenant test](https://zilliverse.feishu.cn/wiki/Kdc3wiyAyihSmMkSIhhcHKMrnih)

**Original reference:** [Milvus multi-tenant design](https://zilliverse.feishu.cn/wiki/RA6zwNZ23i5A0hkendecfcMdnHg)

## Objective

Measure how namespace size affects turbopuffer cold-start and subsequent-query latency. Repeat each small namespace size enough times to observe the latency distribution and whether queries to other namespaces influence it.

This is a focused turbopuffer experiment. It does not reproduce the original Milvus fleet distribution, fixed-QPS case, maximum-QPS case, six query variants, or the 500M-row collection.

## Namespace setup

Keep the original per-namespace sizes and use equal sample counts for A-C. Equal counts compare latency by namespace size without recreating the original fleet population:

| Profile | A namespaces | B namespaces | C namespaces | D namespaces | Total namespaces | Total rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 20 | 20 | 20 | 1 | 61 | 5.38M |
| Medium | 100 | 100 | 100 | 1 | 301 | 6.9M |
| Large | 300 | 300 | 300 | 1 | 901 | 10.7M |

Medium is the default. Small is a functional and variance-estimation pilot. Large is reserved for cases where Medium's confidence intervals are too wide. P99 remains descriptive because even Large has only about three observations in its top percentile.

Use one unique run prefix. Namespace names retain the same row-size suffixes and stable numeric widths:

- `multi_tenant_1000_0001` onward for A
- `multi_tenant_3000_0001` onward for B
- `multi_tenant_15000_001` onward for C
- `multi_tenant_5m_0000001`

Refuse an existing run-prefix collision. Never delete existing namespaces automatically.

## Source data and insertion contract

Use a one-time standalone preparation script to read from `s3://file-transfering-bucket/widetablebenchmark/1b-clean/` in stable object-key and row order. The script downloads only enough source Parquets to produce one prepared 5M-row Parquet file. Credentials use the runtime AWS credential chain and must never be logged, saved in the output, or committed. VDBBench reads only the prepared local file and does not access private S3 during setup or measurement.

The inspected `/home/ubuntu/vdbbench-data-inspect/wide_table_0000.parquet` contains 500,000 rows and uses ZSTD compression. Project only these source columns:

| Source column | Turbopuffer representation |
| --- | --- |
| `emb_768` | 768-dimensional float vector, cosine distance |
| `content` | string with full-text/BM25 indexing |
| `i32_region` | int |
| `f64_price` | float |
| `bool_active` | bool |
| `vc_uuid` | string |
| `vc_tag` | nullable string |
| `vc_desc` | string |
| `bluesky_json` | JSON text stored as string |
| `arr_str_labels` | string array |
| `$meta` | preserve the original JSON string as `meta_json` |

The source has no `pk`. The preparation script assigns deterministic integer IDs from `0` through `4,999,999` in stable source order. Exclude the other ten wide-table columns.

Validate the projected schema, 768-dimensional finite vectors, JSON-object fields, and row counts for only the source objects needed to produce 5M rows. Fail on schema drift. Write ZSTD-compressed row groups so the prepared file remains streamable with bounded memory.

The inspected `$meta` column is a required Arrow string. All 500,000 rows parse as JSON objects; their values use five stable keys (`dyn_extra_a`, `dyn_extra_b`, `dyn_extra_c`, `dyn_source`, and `dyn_version`) with string values. Turbopuffer attribute names cannot start with `$`, and its documented schema has no general JSON-object type. Preserve the source string unchanged in a non-filterable `meta_json` string attribute and record the `$meta` → `meta_json` mapping in the manifest. Do not expand the keys, because expansion would change one source field into five nullable fields. [Turbopuffer attributes](https://turbopuffer.com/docs/write#attributes).

Each setup run creates exactly one namespace of `--multitenant-namespace-rows` rows (default 15000), sliced from row zero of the prepared 5M-row file, so a single setup run needs at most `rows ≤ 5M`. Run setup once per size (e.g. 1K, 10K, 15K, 5M), each with a fresh prefix and manifest. The namespace is named `{run_prefix}_{rows}_0001` (size embedded in the name). Write a static manifest (version 7) containing the run prefix, rows-per-namespace, namespace name and prepared-file range, schema version, the shared queries file and count, and a deterministic search order. Record `started` and `completed` events in an append-only checkpoint file, and copy the shared query set into an atomic `<manifest>.queries.json` sidecar.

Prepare the file on the remote client after configuring its standard AWS credential chain:

```bash
AWS_SHARED_CREDENTIALS_FILE=/home/ubuntu/.aws/vdbbench-turbopuffer-multitenant \
  /home/ubuntu/VectorDBBench/.venv/bin/python scripts/prepare_turbopuffer_multitenant_data.py \
  --download-dir /home/ubuntu/vdbbench-data-inspect/turbopuffer-source \
  --output /home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet
```

Successful output is a JSON summary with `output_rows` equal to `5000000`. The command refuses to replace an existing output file.

## One benchmark case with three operations

Register one case type: `TurboPufferMultiTenantColdStart`. Separate invocations select `setup`, `dense`, or `bm25`. Hybrid search and explicit sparse-vector `SparseKNN` are outside the first version.

The setup operation loads the namespaces without issuing search queries. It also records one shared out-of-sample query set (100 dense vectors plus their BM25 text, extracted from a wide-table Parquet that was NOT used to build the prepared insert file) into the manifest and a `<manifest>.queries.json` sidecar. Dense and BM25 operations use the completed setup manifest and query every namespace in deterministic manifest order:

1. Bind a turbopuffer client to one namespace from the manifest.
2. Run all 100 shared queries in pass `first`, each with turbopuffer's `disable_cache` query flag — every first-pass query is a genuinely cold, uncached read.
3. Run all 100 shared queries again in pass `repeat` without the flag — the warm pass over the same 100 queries.
4. Continue to the next namespace with no concurrent requests.

No tenant argument is required because the runner selects one namespace before issuing each `CustomizedRequest`. The `first` bucket contains every first-pass query (cold) and the `repeat` bucket contains every repeat query (warm), so each namespace contributes a full cold and warm latency distribution over identical queries.

Use a deterministic permutation of namespaces, stored in the manifest, so groups are interleaved and repeated runs are comparable. Do not insert an artificial one-second delay; the experiment is serial because only one request is in flight.

For `dense`, use cosine ANN, `topK=100`, no filter, and IDs only by default. Query vectors come from the shared out-of-sample query set rather than from the namespace's own rows, so no query vector exists inside any namespace.

For `bm25`, issue customized BM25 requests with the shared query set's text and the configured full-text field. The initial field is `content`. Use `topK=100`, no filter, and IDs only by default.

The setup manifest stores the authoritative dense and BM25 field names. `--multitenant-dense-field` and `--multitenant-bm25-field` apply only to setup. Search invocations read those names from the manifest. `--multitenant-output-fields` optionally requests comma-separated declared attributes such as `vc_uuid,vc_tag`; returned values are validated and discarded, while only counts, timing, and cache metadata are persisted.

Run dense and BM25 as separate benchmark invocations with separate setup manifests (the runner rejects a manifest already measured by the other mode). The `disable_cache` flag forces cold regardless of prior query activity, so no idle wait is needed between setup and search.

Before every measured query, append a `started` event to the mode-specific JSONL artifact. Append `completed` or `error` after the call. Record client latency, `cache_hit_ratio`, `cache_temperature`, `server_total_ms`, `query_execution_ms`, namespace name, query mode, pass, query index, result count, and error type. Do not persist query vectors, query text, returned IDs, returned attribute values, or exception messages. Do not retry a measured query in either VDBBench or the SDK. [Turbopuffer query response](https://turbopuffer.com/docs/query).

On resume, a `started` event without a terminal event becomes `indeterminate` and is never rerun. A repeat query with no `started` event after a completed first query is safe to run. A failed or indeterminate first query causes its repeat to be marked `skipped`. Query failures do not stop collection for later namespaces, but the invocation exits as incomplete after writing the summary.

The compact summary reports outcomes plus client, server-total, and query-execution min/max/average/P50/P95/P99, `cache_temperature` counts, and `cache_hit_ratio` stats for the `first` (cold pass, `disable_cache: true`) and `repeat` (warm pass) buckets, plus top-level `cache_temperature`/`cache_hit_ratio` aggregates, `rows_per_namespace`, and `total_rows`. With one namespace per setup run, each run's summary is already per-size; combine summaries across runs for the size comparison. Every event's cache cold/warmness is classified strictly from turbopuffer's reported `cache_temperature`/`cache_hit_ratio` (task policy in AGENTS.md); latency metrics are recorded but never used to label cache state. The summary also points to the raw JSONL event artifact, which all invocations share so resume never reclassifies a completed query.

Turbopuffer honors an undocumented per-query `disable_cache` request-body flag (verified 2026-09-18 for this account; not in the public SDKs or docs). Flagged queries report `cache_temperature="cold"`/`cache_hit_ratio=0.0` and pay the uncached-read cost; un-flagged queries run warm (`hot`/1.0). At equal namespace size the flagged cold latency matches a fresh `branch_from` clone's first query (511 vs 525 ms @ 15K rows), so the flag replaces both the idle-eviction wait (whose samples reported `hot`/1.0 and could not be labeled cold under the policy) and the branch_from approach (issue #2, closed). The internal `_debug/purge_cache`/`_debug/warm_cache` endpoints return 200 for this account but purge has no observable effect — do not rely on them. Cold/warm classification follows the task policy in AGENTS.md: turbopuffer's `cache_temperature`/`cache_hit_ratio` are the SOLE indicators of cache cold/warmness; `server_total_ms`, `query_execution_ms`, and client latency are recorded metrics only and must not label cache state. [Warm-cache API](https://turbopuffer.com/docs/warm-cache).

### CLI

Run setup once for a namespace prefix:

```bash
PYTHONPATH=/home/ubuntu/VectorDBBench-stage1-test \
  /home/ubuntu/VectorDBBench/.venv/bin/vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation setup \
  --multitenant-manifest /home/ubuntu/vdbbench-data-inspect/dense-setup.json \
  --multitenant-prepared-data /home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet \
  --multitenant-namespace-rows 15000 \
  --multitenant-queries-file /home/ubuntu/vdbbench-data-inspect/turbopuffer-queries/queries.json \
  --multitenant-run-prefix multi_tenant_dense
```

Measure every namespace:

```bash
PYTHONPATH=/home/ubuntu/VectorDBBench-stage1-test \
  /home/ubuntu/VectorDBBench/.venv/bin/vectordbbench turbopuffer \
  --api-key "$TURBOPUFFER_API_KEY" --region "$TURBOPUFFER_REGION" \
  --case-type TurboPufferMultiTenantColdStart \
  --multitenant-operation dense \
  --multitenant-manifest /home/ubuntu/vdbbench-data-inspect/dense-setup.json \
  --k 100
```

BM25 requires a separately prepared namespace prefix and manifest (the runner enforces separate manifests per mode). Replace `dense` with `bm25` and optionally add `--multitenant-output-fields vc_uuid,vc_tag`.

See the [run README](turbopuffer-multitenant/README.md) for complete commands, artifact names, resume behavior, and result interpretation.

## Supporting implementation status before a live test

### Wide source and schema

- Implemented a one-time standalone preparation script that uses the AWS credential chain, downloads enough source Parquets for 5M rows, projects and validates the eleven selected fields, assigns deterministic IDs, renames `$meta`, and writes one prepared Parquet file.
- Implemented a bounded VDBBench streaming iterator for that prepared file.
- Reuse the prepared file from row zero for each namespace group according to the selected profile.
- Implemented deterministic row-to-namespace slicing for the A/B/C/D counts.
- Save one shared out-of-sample query set (100 dense vectors plus their BM25 text, extracted from an unused wide-table Parquet) in an atomic query artifact referenced by the manifest.

### Backend-neutral customized-data bridge

- Implemented the customized-row contract, optional `VectorDB` methods, `CustomizedRequest`, and structured `SearchResult` described below.

The simplest alternative is a turbopuffer-specific wide-table adapter called directly by this case. It would be shorter, but it would duplicate client creation, error handling, schema writing, and query response parsing. The proposed optional `insert_customized_rows()` and `search_customized_queries()` capabilities are justified because they bridge the actual VDBBench data-model gap while preserving all existing case interfaces.

#### Concrete API proposal

Put the backend-neutral value objects in one small module, `backend/customized.py`:

```python
@dataclass(frozen=True)
class FieldSchema:
    data_type: Literal["int", "float", "bool", "string", "string[]", "vector"]
    dimensions: int | None = None
    metric: Literal["cosine"] | None = None
    full_text_search: bool = False
    filterable: bool | None = None
    nullable: bool = True

@dataclass(frozen=True)
class CustomizedRow:
    id: str | int
    fields: Mapping[str, Any]

@dataclass(frozen=True)
class CustomizedRequest:
    mode: Literal["dense", "bm25"]
    field: str
    value: Sequence[float] | str
    top_k: int = 100
    include_fields: tuple[str, ...] = ()

@dataclass(frozen=True)
class SearchPerformance:
    cache_hit_ratio: float | None = None
    cache_temperature: str | None = None
    server_total_ms: float | None = None
    query_execution_ms: float | None = None

@dataclass(frozen=True)
class SearchResult:
    ids: list[str | int]
    fields: Mapping[str, Sequence[Any]]
    performance: SearchPerformance
```

Each `CustomizedRow` represents one stored row. The insertion call receives a bounded sequence of rows plus one shared schema, avoiding a redundant schema copy in every row. It validates that every row field is declared, vector dimensions match, and no field collides with `id`. Schema validation belongs at this boundary so the dataset runner and database client cannot silently drop or reshape fields.

Add three optional methods to `VectorDB`. They are concrete methods whose defaults report unsupported capability, rather than abstract methods that would force every existing backend to implement them:

```python
@classmethod
def supports_customized_api(cls) -> bool:
    return False

def insert_customized_rows(
    self,
    rows: Sequence[CustomizedRow],
    schema: Mapping[str, FieldSchema],
) -> tuple[int, Exception | None]:
    raise NotImplementedError

def search_customized_queries(
    self,
    requests: Sequence[CustomizedRequest],
) -> list[SearchResult]:
    raise NotImplementedError
```

The new case checks `supports_customized_api()` before setup. Existing `insert_embeddings()`, `search_embedding()`, `insert_documents()`, and `search_documents()` contracts remain valid, so current cases and clients do not change behavior.

For turbopuffer, `insert_customized_rows()` converts a bounded row sequence into `upsert_columns` and translates the declared `FieldSchema` values into the turbopuffer schema. The translation is the only backend-specific part of insertion. Namespace selection stays outside the rows: the orchestrator creates a client bound to one manifest namespace and passes only that namespace's rows.

`search_customized_queries()` maps each `CustomizedRequest`: `dense` becomes ANN on the requested vector field and `bm25` becomes BM25 on the requested string field. It converts each SDK response into a `SearchResult`, preserving returned IDs, selected fields, and performance metadata without exposing turbopuffer response objects to runners.

The existing turbopuffer search methods become compatibility wrappers around the same internal query implementation:

```python
def search_documents(self, query, k=100, payload_profile=IDS_ONLY, **kwargs):
    field = kwargs.get("field_name", self._text_field)
    request = CustomizedRequest("bm25", field, query, k)
    return self.search_customized_queries([request])[0].ids
```

`search_embedding()` follows the same pattern for dense ANN. This gives ordinary FTS callers a configurable `field_name` while retaining their current list-of-IDs result. For a wide namespace, supplying `field_name="content"` selects the declared BM25 field even though the client is not using the legacy FTS-only configuration. A call with no field keeps the current FTS configuration check. The multi-tenant cold case calls `search_customized_queries([request])` because it also needs timing and cache metadata; its BM25 request uses the exact same implementation as `search_documents()`.

The wide case can use the existing `TurboPufferIndexConfig` with cosine distance. It does not need a second FTS case configuration because the insertion schema declares `content` as full-text searchable and each `CustomizedRequest` identifies its target field. This removes the current vector-versus-FTS configuration split only for the optional customized-data path.

The insertion pipeline streams bounded lists of `CustomizedRow`. The namespace slicer consumes rows in deterministic source order, closes a write batch at its configured row limit without crossing a namespace boundary, binds a turbopuffer client to that namespace, and calls `insert_customized_rows()`. The setup checkpoint advances only after every expected row has been acknowledged and the namespace query fixture has been saved atomically.

The search pipeline reconstructs one `CustomizedRequest` from each namespace's saved fixture, calls `search_customized_queries([request])` once for the first sample and once for the repeat sample, and immediately appends each result to the JSONL result artifact. Aggregation consumes that artifact, so resume and reporting do not require keeping all namespace results in memory.

Do not create a generic backend schema language beyond the six field kinds needed by the inspected file. Do not expose arbitrary turbopuffer query dictionaries. If another backend later implements this capability, it maps these same narrow contracts or reports the capability as unsupported.

### Turbopuffer implementation

- Implement `insert_customized_rows()` with turbopuffer column writes, declaring cosine distance for `emb_768`, full-text indexing for `content`, and the scalar/array representations in the insertion contract.
- Map the source `$meta` string to a non-filterable `meta_json` string attribute and record that field-name mapping in the manifest.
- Preserve `bluesky_json` as JSON text unless the insertion contract is deliberately revised.
- Implement structured dense queries against `emb_768` and BM25 queries against the configured field, initially `content`.
- Support arbitrary `include_attributes` from the declared schema even though the first cold-latency run uses IDs only.
- Return turbopuffer performance metadata to the runner rather than reducing the SDK response immediately to IDs.
- Require every namespace to have a completed setup checkpoint and the shared query set before allowing measurement. No readiness polling is part of this experiment.

### Cold-latency orchestration and results

- Implemented a dedicated serial first/repeat runner around `search_customized_queries()` so the turbopuffer response metadata is retained.
- Disabled both benchmark-level and SDK-level retries for this case's measured queries.
- Persist each query transition before advancing and preserve interrupted queries as indeterminate.
- Aggregate each namespace group and pass separately, including outcomes and client/server latency distributions.
- Store the deterministic namespace order and completed-query state so resume never reclassifies an already-started namespace query as cold.

### Verification gates

- Add deterministic unit tests for wide Parquet projection, namespace slicing, schema conversion, dense/BM25 dispatch, response metadata, aggregation, failure recording, and resume behavior.
- Run focused VDBBench tests on the designated remote client according to `AGENTS.md`.
- Run a disposable pilot with a few tiny namespaces to verify the actual turbopuffer schema, field returns, BM25 query behavior, query metadata, and first/repeat classification.
- Inspect the pilot artifact and confirm that no credentials or full row payloads appear in logs or results before loading Medium or Large.

No adaptive stopping, fixed-QPS scheduler, max-QPS ramp, hybrid/RRF query, explicit sparse-vector field, eviction emulation, D/E replicas, or aggregate 1 TB guard is part of this implementation.
