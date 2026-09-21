# Finding: turbopuffer's undocumented `disable_cache` query parameter

**Date:** 2026-09-18 · **Source:** `turbopuffer/tpuf-benchmark` (main @ `6a1d2cf`, the tool behind
https://turbopuffer.github.io/tpuf-benchmark/) · **SDKs checked:** turbopuffer 2.6.0 (Python),
turbopuffer-go v1.12.2

## Summary

turbopuffer's query endpoint accepts an **undocumented `disable_cache` request-body flag**, and there
are internal `_debug/warm_cache` / `_debug/purge_cache` endpoints. They are **not** in the public SDKs
or docs; they are used by turbopuffer's own benchmark tool (`tpuf-benchmark`), whose nightlies data
embeds `"disable_cache": true` in the query templates.

## Evidence

- All three cold benchmark definitions include `"disable_cache": true` in the query template:
  - **ANN** — `benchmarks/website/vector-10m-cold.toml`: `[workload.query.ann]`, `rank_by: ["vector", "ANN", …]`
    (the nightlies' `vectors_10m_cold`). Note: there is no literal `ann_search` anywhere — the workload
    key is `ann` and the `rank_by` operator is `"ANN"`.
  - **kNN** — `benchmarks/vector-knn-1m-cold.toml`: `rank_by: ["vector", "kNN", …]`
  - **BM25** — `benchmarks/website/fulltext-10m-cold.toml`: `rank_by: ["text", "BM25", …]`
- tpuf-benchmark POSTs the template body **verbatim** to `POST /v2/namespaces/{ns}/query`
  (`pkg/bench/namespace.go`, `Namespace.Query`), bypassing the typed SDK params.
- Exposed cache-control surface in tpuf-benchmark (module `github.com/turbopuffer/tpuf-benchmark`,
  importable Go library + CLI):
  - `Namespace.Query(ctx, maxRetries, tmpl)` — any body, incl. `"disable_cache": true`; returns
    `QueryPerformance` (`cache_temperature`, `cache_hit_ratio`).
  - `Namespace.WarmCache(ctx)` → GET `_debug/warm_cache`; `Namespace.PurgeCache(ctx)` → GET
    `_debug/purge_cache` ("ensures the namespace is in a cold state").
  - CLI: `tpufbench run --warm-cache | --purge-cache <definition.toml>`.
- **Not exposed**: Python SDK 2.6.0 `NamespaceQueryParams` TypedDict, Go SDK typed params, public docs.
  The `_debug/*` calls silently no-op on HTTP 404 (code treats 404 as success), so they may be gated to
  internal accounts — unverified for our API key.

## Relevance to the multitenant cold-start case

- Current design relies on natural idle eviction for cold samples (cold-wait note in
  `docs/turbopuffer-multitenant/README.md`); `cache_temperature`/`cache_hit_ratio` remain the SOLE
  cold/warm labels per AGENTS.md.
- If `disable_cache: true` is honored for our key, it yields **deterministic per-query cold samples** —
  no 60-min idle wait, no `branch_from` cost — and is a potential simpler alternative to the branch-from
  hack (issue #2).
- Classification policy unchanged: `disable_cache` is a mechanism candidate, not a classifier.

## Proposed verification (fold into the stage-4 pilot)

1. Send one query with `"disable_cache": true` against a tiny namespace (Python SDK `extra_body`, or
   a compiled `tpufbench` / `pkg/bench` probe). Accepted, or rejected as an unknown field?
2. If accepted: record reported `cache_temperature`/`cache_hit_ratio` on that query and an immediate repeat.
3. Try `_debug/purge_cache` and `_debug/warm_cache` with our key; watch for silent 404 no-ops.
4. Compare against the branch_from result (issue #2): cold latency, reproducibility, cost.

## Verification (2026-09-18) — honored for our key ✅

One-shot probe on the remote client (`aws-us-west-2`, SDK 2.6.0, disposable 1K-row namespace
`tp_mt_dc_probe_20260918_a1k`), using the SDK's `extra_body` on `ns.query`:

| Query | flag | cache_temperature | cache_hit_ratio | server_total_ms |
|---|---|---|---|---|
| ANN q1 | `disable_cache: true` | `cold` | 0.0 | 80 |
| ANN q2 (immediate repeat) | — | `hot` | 1.0 | 17 |
| ANN q3 | `disable_cache: true` | `cold` | 0.0 | 67 |
| BM25 | `disable_cache: true` | `cold` | 0.0 | 77 |

- Accepted (no unknown-field error); deterministic `cold`/`0.0` with the uncached-read latency
  (67–80 ms); repeated flagged queries stay cold; repeat without the flag is `hot`/1.0 (17 ms).
- `_debug/purge_cache` and `_debug/warm_cache` return 200 `{"status":"OK"}` for our key, but
  **purge_cache had no observable effect** (next un-flagged query still `hot`/1.0 @ 12 ms) — do not
  rely on it as a cold-forcer; `disable_cache` makes it unnecessary.
- **Recommendation**: send `disable_cache: true` on the entire `first` pass (cold pass) and no flag on
  `repeat` (warm pass) — supersedes the 60-min idle-eviction wait and the branch-from hack (issue #2).

## Size-controlled comparison (2026-09-18) — disable_cache ≈ branch_from at equal size

Same 15K-row namespace (`tp_mt_cmp_probe_20260918_c15k`, 46 MB vectors):

| Scenario | server_total_ms | cache_temperature | cache_hit_ratio |
|---|---|---|---|
| warm baseline (no flag) | 20 | `hot` | 1.0 |
| `disable_cache=true` | **511** | `cold` | 0.0 |
| `disable_cache=true` (repeat) | 511 | `cold` | 0.0 |
| branch_from first query | **525** | `cold` | 0.0 |
| branch repeat | 21 | `hot` | 1.0 |

- At equal size the two mechanisms agree (511 vs 525 ms): `disable_cache` is the same uncached read,
  not a partial bypass. Cold latency scales with data size (3 MB → ~80 ms, 46 MB → ~511 ms).
- The pilots' "idle-evicted cold" (58–96 ms, reported `hot`/1.0) is a milder partial-retention state
  that the cache fields never labeled cold; `disable_cache` yields the true cold read AND labels it.

## Open questions

- Is `disable_cache` a supported customer-facing API or internal-only? (Works today; unverified stability.)
- Exact cold magnitude/stability at the 5M-row namespace (~15 GB) — one confirmation probe before Medium.
- Interaction with `hint_cache_warm` and namespace pinning.
