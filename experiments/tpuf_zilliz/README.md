# Turbopuffer vs Zilliz Cloud — Reproducibility Bundle

Standalone scripts + VDBBench invocations referenced by
`turbopuffer_vs_zilliz_final_report.md`
(at `github.com/jamesgao-jpg/turbopuffer_test_results`, root folder).

## Branch pin

Everything in this folder was written against this VDBBench fork:

- Repo: `github.com/jamesgao-jpg/VectorDBBench.git`
- Branch: `main` (fast-forwarded from the feature branch `tp_vs_zilliz_0415`)
- Commit: **`dd6f786`** ("turbopuffer + milvus: RETURN_VECTOR env var toggle for recall-field test")

All env-var toggles on that commit default to off, so unchanged
VDBBench invocations reproduce every published number byte-for-byte.

## Folder layout

Each subfolder corresponds to one section of the customer-facing report.

| Folder | Report § | Contents |
|--------|----------|----------|
| `insertion/` | § 1 (Write / Index Construction) | `batch100_bp_on_reattach.py`, `batch_size_sweep.py`, `wait_for_index_drain.py` |
| `search/` | § 2 (Search) | `serial_stability_test.py`, `topk_saturation_test.py` |
| `cost/` | § 3 (Single-tenant Cost) | `cost_calc.py` |
| `multi_tenant/` | § 4 (Multi-tenant Cost) | tenant splitter + per-backend loaders + search harnesses |
| `delete_consistency/` | § 5 (TB delete bug) | `delete_consistency_test_cohere.py`, `delete_consistency_test.py` |
| `cache_probe/` | § 2 supporting | `cache_eviction_probe_test.py` |
| `shared/` | utilities | `plot_recall.py` |

## Env vars touched by VDBBench invocations

| Env var | Default | Purpose |
|---------|---------|---------|
| `NUM_PER_BATCH` | case-specific | bulk batch size |
| `DISABLE_BACKPRESSURE` | off | TB `disable_backpressure=True` on writes |
| `STAGE_ROWS` | 0 | per-worker stage log every N rows |
| `TPUF_SKIP_CACHE_WARM` | off | suppress TB `hint_cache_warm` in `optimize()` — required for true-cold search |
| `RETURN_VECTOR` | off | `include_attributes=["vector"]` (TB) / `output_fields=["vector"]` (Milvus) |

Each script's header documents any env overrides specific to it (e.g.
`POLL_ITERATIONS`, `BATCH_SIZES`, `DURATION_SEC`). Defaults always
match the numbers in the report.

## Dataset

- **LAION 100 M** — `s3://assets.zilliz.com/benchmark/laion_large_100m/`
  (anonymous, `us-west-2`, ~200 GB across 100 train parquet files)
- **Cohere 10 M** — `s3://assets.zilliz.com/benchmark/cohere_large_10m/`
  (anonymous, `us-west-2`, ~30 GB across 10 shuffled train files)

Local cache path expected by all scripts:
`/tmp/vectordb_bench/dataset/{laion,cohere}/{laion_large_100m,cohere_large_10m}/`

VDBBench auto-downloads from S3 on first `--case-type` reference; the
standalone scripts assume the files are already present.

## Reproducibility rules

1. Re-runs must invoke the committed script unchanged. Any SDK-breakage
   patch needs explicit sign-off and is noted in the per-section
   report.
2. Parameter sweeps are additive (env var with prior default), never
   edit-in-place — so the unchanged invocation still reproduces
   byte-for-byte.
3. Consistency level is stated per section. TB defaults to Strong;
   Zilliz (ZT / ZC) uses Session (set by VDBBench at collection
   creation).

## Full test reports

Every script's experiment has a corresponding full-methodology report
at the root of `turbopuffer_test_results`:

| Folder in `turbopuffer_test_results` | Report |
|--------------------------------------|--------|
| `insertion/` | `insertion_test.md` + `batch100_bp_on_results.json` |
| `turbopuffer_different_batch_speed/` | `turbopuffer_different_batch_speed_test.md` + JSONs + PNG |
| `search/` | `search_test.md` |
| `filter_search/` | `filter_search_test.md` |
| `topk_saturation/` | `topk_saturation_test.md` |
| `recall_field_query/` | `recall_field_query_test.md` |
| `serial_stability/` | `serial_stability_test.md` + JSON + PNG |
| `cache_eviction_probe/` | `cache_eviction_probe_test.md` + JSON + PNG |
| `delete_consistency/` | `delete_consistency_test.md` + 2 JSONs |
| `multi_tenant_test/` | `README.md` + `insertion/` `search/` `cost/` subfolders |
| `cost_qps_pareto/` | `cost_qps_pareto_test.md`, `full_cost_test.md` + `cost_calc.py` |

## Empirical billing reconciliation

Per actual TB dashboard reading on 2026-04-20:
- 811,456 queries across our test campaign → billed **47,389.27 TB**
- **Matches the fp32-basis calculation** (811,456 × 58.37 GB effective = 47,362 TB) within 0.06 %
- Confirms TB queries bill on **physical upload precision (fp32)**,
  not the quantized-index size (fp16)
- Per-query cost at LAION 100 M scale: **$5.84 × 10⁻⁵**
- Cost models in `cost/cost_calc.py` and the report use this rate.
