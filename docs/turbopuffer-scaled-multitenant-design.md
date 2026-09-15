# Turbopuffer multi-tenant cold-start experiment (revised draft)

**Rationale:** [Turbopuffer multi-tenant test](https://zilliverse.feishu.cn/wiki/Kdc3wiyAyihSmMkSIhhcHKMrnih)

**Original reference:** [Milvus multi-tenant design](https://zilliverse.feishu.cn/wiki/RA6zwNZ23i5A0hkendecfcMdnHg)

## Objective

Measure how namespace size affects turbopuffer cold-start and subsequent-query latency. Repeat each small namespace size enough times to observe the latency distribution and whether queries to other namespaces influence it.

This is a focused turbopuffer experiment. It does not reproduce the original Milvus fleet distribution, fixed-QPS case, maximum-QPS case, six query variants, or the 500M-row collection.

## Namespace setup

Keep the original per-namespace sizes and reduce only the namespace counts:

| Group | Rows per namespace | Namespace count | Total rows |
| --- | ---: | ---: | ---: |
| A | 1,000 | 3,000 | 3M |
| B | 3,000 | 1,000 | 3M |
| C | 15,000 | 200 | 3M |
| D | 5M | 1 | 5M |
| **Total** |  | **4,201** | **14M** |

At the Milvus document's approximate 4.8 KB per row, this is about 67.2 GB. The selected columns in the inspected Parquet sample occupy about 3.96 KB of uncompressed column storage per row, or about 55.4 GB for 14M rows. Both are far below the 1 TB limit, so runtime size-based stopping and small/medium/large scale profiles are unnecessary.

Use one unique run prefix and these suffix ranges:

- `multi_tenant_1000_0001` through `multi_tenant_1000_3000`
- `multi_tenant_3000_0001` through `multi_tenant_3000_1000`
- `multi_tenant_15000_001` through `multi_tenant_15000_200`
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

During setup, reuse the prepared file from the beginning for each group: A, B, and C each consume the first 3M rows, while D consumes all 5M rows. This creates the required 14M inserted rows from 5M prepared rows and gives A-C a common data population for size comparisons. Write a static manifest containing the run prefix, namespace names, group, expected row counts, prepared-file ranges, schema version, fixture paths, and deterministic search order. Record `started` and `completed` events in an append-only checkpoint file, and write each namespace's dense vector and BM25 text fixture to its own atomic JSON file.

Prepare the file on the remote client after configuring its standard AWS credential chain:

```bash
AWS_SHARED_CREDENTIALS_FILE=/home/ubuntu/.aws/vdbbench-turbopuffer-multitenant \
  /home/ubuntu/VectorDBBench/.venv/bin/python scripts/prepare_turbopuffer_multitenant_data.py \
  --download-dir /home/ubuntu/vdbbench-data-inspect/turbopuffer-source \
  --output /home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet
```

Successful output is a JSON summary with `output_rows` equal to `5000000`. The command refuses to replace an existing output file.

## One benchmark case

Register one case type: `TurboPufferMultiTenantColdStart`. Each invocation selects one query mode: `dense` or `bm25`. Hybrid search and explicit sparse-vector `SparseKNN` are outside the first version.

The setup stage loads the 4,201 namespaces without issuing search queries. The search stage launches the existing cold/warm measurement logic against each namespace sequentially. Each per-namespace run uses exactly one query:

1. Bind a turbopuffer client to one namespace from the manifest.
2. Run one first query and retain it as that namespace's cold sample.
3. Repeat the same query immediately and retain it as that namespace's after-first-query sample.
4. Continue to the next namespace with no concurrent requests.

No tenant argument is required on `search_embedding()` or `search_documents()` because each run is bound to a single namespace. Aggregate the individual samples by the namespace's row-count group so A/B/C provide cold and repeat-query latency distributions. Group D has one namespace and therefore produces an individual observation rather than a percentile distribution.

Use a deterministic permutation of namespaces, stored in the manifest, so groups are interleaved and repeated runs are comparable. Do not insert an artificial one-second delay; the experiment is serial because only one request is in flight.

For `dense`, use cosine ANN, `topK=100`, no filter, and IDs only. Select one query vector deterministically from that namespace's rows.

For `bm25`, invoke `search_documents()` with a deterministic text query and a configured full-text field. The initial field is `content`; the API must not hardcode the existing `text` field so another declared string field can be selected later. Use `topK=100`, no filter, and IDs only.

Run dense and BM25 as separate benchmark invocations. A query of either type can warm namespace data, so both modes cannot claim a cold pass against the same already-queried prefix. Give each mode a fresh namespace prefix, or explicitly label the second mode as an after-prior-query experiment.

Record client latency, `cache_hit_ratio`, `cache_temperature`, `server_total_ms`, `query_execution_ms`, namespace group, namespace name, query mode, pass, ordinal position, result count, and errors. Do not retry failed queries inside the measured pass. [Turbopuffer query response](https://turbopuffer.com/docs/query).

The repeat observation means "queried once before" rather than guaranteed cache residency. Turbopuffer has no verified public cache-eviction operation, and its warm-cache hint does not confirm that warming completed. Report the returned cache state rather than relabeling observations based on assumption. [Warm-cache API](https://turbopuffer.com/docs/warm-cache).

Rerunning a true cold pass requires a new namespace prefix. Search-only reruns against the same prefix are repeat-query measurements.

## Supporting implementation required before a live test

### Wide source and schema

- Add a one-time standalone preparation script that uses the AWS credential chain, downloads enough source Parquets for 5M rows, projects and validates the eleven selected fields, assigns deterministic IDs, renames `$meta`, and writes one prepared Parquet file.
- Add a VDBBench streaming iterator for that prepared file. It must preserve the projected Arrow types and must not materialize the 5M rows in memory.
- Reuse the prepared file for each namespace group: read 3M rows for each of A-C and all 5M rows for D.
- Add deterministic row-to-namespace slicing for the A/B/C/D counts. The existing `row_id % tenant_count` distribution cannot produce different namespace sizes.
- Save one deterministic dense query vector and one deterministic BM25 query string for every namespace in a small query-fixture artifact referenced by the manifest.

### Backend-neutral customized-data bridge

- Add a customized-row contract containing an ID and named fields, with one shared schema declaration per insertion call. Keep the existing vector and FTS insertion methods unchanged for current cases.
- Add an optional `insert_customized_rows()` capability to `VectorDB`, with a default unsupported implementation so existing backends are unaffected.
- Add an optional structured query result containing IDs plus backend timing/cache metadata. Current search methods return IDs and discard the turbopuffer response metadata needed by this experiment.
- Add a `CustomizedRequest` containing query mode, target field, vector or text value, `topK`, and included fields. It should express dense and BM25 requests without exposing a raw backend request through the case interface.

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

The insertion pipeline streams bounded lists of `CustomizedRow`. The namespace slicer consumes rows in deterministic source order, closes a write batch at its configured byte or row limit without crossing a namespace boundary, binds a turbopuffer client to that namespace, and calls `insert_customized_rows()`. The manifest advances only after the inserted count and namespace metadata match the expected count.

The search pipeline reconstructs one `CustomizedRequest` from each namespace's saved fixture, calls `search_customized_queries([request])` once for the cold sample and once for the repeat sample, and immediately appends each result to the JSONL result artifact. Aggregation consumes that artifact, so resume and reporting do not require keeping all 4,201 namespace results in memory.

Do not create a generic backend schema language beyond the six field kinds needed by the inspected file. Do not expose arbitrary turbopuffer query dictionaries. If another backend later implements this capability, it maps these same narrow contracts or reports the capability as unsupported.

### Turbopuffer implementation

- Implement `insert_customized_rows()` with turbopuffer column writes, declaring cosine distance for `emb_768`, full-text indexing for `content`, and the scalar/array representations in the insertion contract.
- Map the source `$meta` string to a non-filterable `meta_json` string attribute and record that field-name mapping in the manifest.
- Preserve `bluesky_json` as JSON text unless the insertion contract is deliberately revised.
- Implement structured dense queries against `emb_768` and BM25 queries against the configured field, initially `content`.
- Support arbitrary `include_attributes` from the declared schema even though the first cold-latency run uses IDs only.
- Return turbopuffer performance metadata to the runner rather than reducing the SDK response immediately to IDs.
- Add namespace row-count/readiness checks and record them in the manifest before allowing a cold-latency run.

### Cold-latency orchestration and results

- Reuse the existing cold/warm runner's two-pass timing and percentile helpers, but call it once per namespace with `query_count=1`.
- Add query dispatch through `search_customized_queries()`: dense mode follows the same implementation as `search_embedding()`, while BM25 mode follows the same implementation as `search_documents(field_name="content")`. The structured entry point preserves response metadata that the existing ID-only methods discard.
- Disable benchmark-level retries for measured queries. Record a failed first or repeat query as a sample with its error category.
- Persist every per-namespace sample before advancing so a long 4,201-namespace run can be resumed without losing completed observations.
- Aggregate by `row_count × query_mode × pass`, reporting sample count, failures, average, P50, P95, and P99. Report D as a single observation.
- Keep first-query and repeat-query results separate; do not combine them into one latency distribution.
- Store the namespace order and completed-query state so resume never accidentally reclassifies an already-queried namespace as cold.

### Verification gates

- Add deterministic unit tests for wide Parquet projection, namespace slicing, schema conversion, dense/BM25 dispatch, response metadata, aggregation, failure recording, and resume behavior.
- Run focused VDBBench tests on the designated remote client according to `AGENTS.md`.
- Run a disposable pilot with a few tiny namespaces to verify the actual turbopuffer schema, field returns, BM25 index readiness, query metadata, and cold/repeat classification.
- Inspect the pilot artifact and confirm that no credentials or full row payloads appear in logs or results before loading the 14M-row setup.

No scale flag, fixed-QPS scheduler, max-QPS ramp, hybrid/RRF query, explicit sparse-vector field, eviction emulation, D/E replicas, or aggregate 1 TB guard is part of this implementation.
