# Session Progress

**Date:** 2026-09-15
**Working Directory:** /Users/james.gao/Desktop/VectorDBBench-turbopuffer

## Completed Work

### 1. Update local main
Fetched `zilliztech/VectorDBBench` and fast-forwarded local `main` to `401a6b78da7275a81067399483c01881f5727112`.
- Files: `AGENTS.md` (read)

### 2. Check out main
Preserved 74 tracked and untracked changes from `fts_impl_only` in `stash@{0}`, then switched this checkout to clean, up-to-date `main`.
- Files: `AGENTS.md` (read)

### 3. Review the multi-tenant experiment and VDBBench paths
Read the Feishu Milvus design, VDBBench's multitenant and turbopuffer code, and turbopuffer's API documentation. Identified the size-cap conflict and backend schema/cache differences, then proposed three scaled profiles and a dedicated case design.
- Files: `vectordb_bench/backend/cases.py` (read), `vectordb_bench/backend/task_runner.py` (read), `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py` (read), `vectordb_bench/backend/runner/serial_runner.py` (read)

### 4. Record the draft design
Saved the proposed profiles, schema translation, workload phases, VDBBench extension points, and unresolved inputs for later refinement.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (created), `SESSION_PROGRESS.md` (created)

### 5. Record the test-data source and attempt format inspection
Recorded the user-designated S3 prefix in the draft without credentials. Read-only listing from the remote client failed: the supplied key pair was rejected by AWS, and anonymous listing was denied. No source objects or rows were inspected.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated)

### 6. Verify replacement access and inspect the Parquet source
Verified the replacement credentials for read-only S3 access without persisting them. The `widetablebenchmark/1b-clean/` prefix has 2,000 objects totaling 3.887 TB stored. Inspected the remote `wide_table_0000.parquet` footer and 1,024 rows, then recorded the selected-column insertion contract and exclusions in the design. The sample file has 500,000 rows, 21 top-level fields, and no `pk` column.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated)

### 7. Split the proposed search workload into three cases
Rechecked the Milvus plan's serial baseline, fixed-QPS, and maximum-QPS sections. Revised the draft to use three selectable VDBBench case types with one shared data/setup manifest, disjoint cold-query namespace pools, and a separate runtime budget for each invocation. Recorded cold-first measurements as the accepted priority and clarified that backend representation and cache differences are reporting limits rather than approval gates.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated)

### 8. Rework the scale profiles around tenant distribution
Changed the draft profiles to stratify A–C at their original 50/30/20 namespace-count ratio and original rows per namespace, while including D/E as rare explicit strata. The proposed small/medium/large profiles have 302/5,002/25,004 namespaces and nominal raw sizes of 7.8/129.6/696 GB. Large retains 5M rows per D namespace. Documented that D/E cold statistics cannot cover all six variants from one representative setup under the cap.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated)

### 9. Replace the scaled reproduction with a focused cold-start experiment
Read the revised turbopuffer rationale and simplified the draft to one setup and one serial case. The setup now has 3,000 × 1k-row, 1,000 × 3k-row, 200 × 15k-row, and 1 × 5M-row namespaces: 4,201 namespaces and 14M rows, estimated at roughly 55–67 GB. The measured workflow is one first-query pass followed by the same query pass again. Removed the scale profiles, fixed/max-QPS cases, six-variant matrix, 500M-row tier, cold-pool allocation, and byte-cap machinery.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (rewritten), `SESSION_PROGRESS.md` (updated)

### 10. Record the pre-test implementation checklist
Reviewed the current cold-latency, custom Parquet, insert, payload, full-text, and turbopuffer paths. Refined the case to run one cold and one repeat query against each namespace sequentially, aggregate by namespace size, and support separate dense and BM25 invocations. Hybrid and explicit sparse-vector search are out of scope. Recorded the wide-row data bridge, private-S3 reader, turbopuffer schema/query work, per-namespace result persistence, and verification gates required before a live test.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated), `SESSION_PROGRESS.md` (updated)

### 11. Specify the backend-neutral bridge
Defined a minimal typed contract for customized rows, field schemas, dense/BM25 customized requests, structured search results, and performance metadata. Proposed optional `VectorDB.insert_customized_rows()` and `search_customized_queries()` methods with unsupported defaults, plus turbopuffer compatibility wrappers so existing vector and FTS cases retain their current APIs. `CustomizedRow` represents one row and `CustomizedRequest` represents one query; the plural methods accept bounded sequences. Recorded the namespace-bound streaming insertion flow and JSONL-backed cold/repeat query flow.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated), `SESSION_PROGRESS.md` (updated)

### 12. Resolve the `$meta` representation
Inspected all 500,000 `$meta` values in the local Parquet sample on the remote client. The column is a required Arrow string, every value is a valid JSON object, and the objects contain five stable string-valued dynamic keys. Turbopuffer forbids attribute names beginning with `$` and does not document a general JSON-object field type, so the design now preserves the source value as a non-filterable `meta_json` string and records the field-name mapping in the manifest.
- Files: `docs/turbopuffer-scaled-multitenant-design.md` (updated), `SESSION_PROGRESS.md` (updated)

### 13. Record the four stages and implement stage 1
Recorded the four-stage end-to-end implementation plan in the repository-local `AGENTS.md`. Implemented `FieldSchema`, `CustomizedRow`, `CustomizedRequest`, `SearchPerformance`, and `SearchResult`; added optional customized-data methods to `VectorDB`; and implemented schema conversion, wide-row insertion, dense/BM25 dispatch, selected-field returns, and query performance extraction for turbopuffer. Existing dense and FTS methods share the customized-query implementation. Added focused coverage for the complete selected schema, dense and BM25 requests, performance metadata, configurable BM25 field selection, and schema-drift rejection.

Verified on the designated remote client in a clean detached worktree at the latest `fork/fts_impl_only` commit `ae054dfd781645a425efb5233f0355fcbc1fa838`: 39 focused and compatibility tests passed, and Ruff check/format passed for all changed Python files. The broader `test_cloud_insert_case.py` module could not collect because the shared remote virtual environment lacks its optional `pinecone` dependency; the new customized insertion test passed. No live namespace was created because the remote client has no configured turbopuffer profile or `TURBOPUFFER_API_KEY`/region environment variables.
- Files: `AGENTS.md` (updated, repository-local and ignored), `vectordb_bench/backend/customized.py` (created), `vectordb_bench/backend/clients/api.py` (updated), `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py` (updated), `tests/test_turbopuffer_customized.py` (created), `docs/turbopuffer-scaled-multitenant-design.md` (updated), `SESSION_PROGRESS.md` (updated)

### 14. Isolate the task in matching worktrees
Assigned the task name `turbopuffer_multitenant` to local `/Users/james.gao/Desktop/VectorDBBench-turbopuffer` and remote `/home/ubuntu/VectorDBBench-stage1-test` worktrees in the repository-local `AGENTS.md`. Created `codex/turbopuffer_multitenant` in both repositories from freshly fetched `origin/main` commit `401a6b78da7275a81067399483c01881f5727112`, applied the task files to both, and removed the duplicate task files and hunks from the shared local checkout. The shared local and remote checkouts retain their unrelated user changes.

Reverified the exact latest-main remote worktree: 55 focused and compatibility tests passed, and Ruff check/format passed for all changed Python files.
- Files: `AGENTS.md` (updated, repository-local and ignored), all stage-1 implementation and design files transferred without content changes

### 15. Implement the one-time 5M-row Parquet preparation
Added a standalone script that uses the AWS credential chain, downloads source Parquets in stable object-key order until their footers cover 5M rows, validates the selected schema, finite 768-dimensional vectors, and JSON-object fields, then writes exactly 5M rows to one ZSTD-compressed Parquet file. It assigns deterministic integer IDs and renames `$meta` to `meta_json`. The script streams bounded record batches and uses partial files so failed downloads or writes do not appear complete.

Verified on the assigned remote worktree with synthetic Parquet objects: it downloaded only the files needed to cover the target, trimmed the final input to the exact row count, preserved the output schema and order, renamed `$meta`, generated deterministic IDs, and rejected incorrect vector dimensions. Two focused tests passed. The CLI help command passed, and the script accepted the schema of the inspected 500,000-row source sample.

Ran the preparation against private S3 using a dedicated remote credential file owned by `ubuntu` with mode `0600`. Ten source Parquets totaling 5M rows produced `/home/ubuntu/vdbbench-data-inspect/turbopuffer_multitenant_5m.parquet`. Independent verification confirmed 5,000,000 rows, 310 row groups, deterministic IDs `0..4,999,999`, the exact prepared schema, ZSTD compression for every column chunk, a 12,255,547,788-byte file, no leftover partial output, and no credential pattern in the job log.
- Files: `scripts/prepare_turbopuffer_multitenant_data.py` (created), `tests/test_prepare_turbopuffer_multitenant_data.py` (created), `docs/turbopuffer-scaled-multitenant-design.md` (updated)

### 16. Implement the VDBBench namespace setup loader
Added optional namespace selection and existence checks to the database contract and implemented them for turbopuffer using one initialized SDK client. Added the exact A/B/C/D group definitions and a bounded prepared-Parquet reader that restarts from row zero per group, never crosses a namespace boundary in one insert call, and converts each Arrow slice into the Stage 1 `CustomizedRow` contract.

The setup runner refuses untracked existing namespaces, records `started` before the first write, uses deterministic IDs for safe whole-batch upserts after partial failures, and records `completed` only after the namespace reaches its exact expected row count and its query fixture is atomically saved. A static manifest records all 4,201 names, source ranges, schema, fixture paths, and a deterministic interleaved search order; an append-only JSONL file records resumable setup checkpoints.

Verified on the assigned remote worktree: 10 focused preparation/setup/client tests passed. A footer-only check against the real prepared file produced exactly 4,201 namespace specifications totaling 14,000,000 planned inserts, from `multi_tenant_1000_0001` through `multi_tenant_5m_0000001`. No live turbopuffer namespace was created.
- Files: `vectordb_bench/backend/turbopuffer_multitenant.py` (created), `vectordb_bench/backend/clients/api.py` (updated), `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py` (updated), `tests/test_turbopuffer_multitenant_setup.py` (created), `tests/test_turbopuffer_customized.py` (updated), `AGENTS.md` (updated, repository-local and ignored)

### 17. Implement the Stage 3 benchmark case
Registered `TurboPufferMultiTenantColdStart` and exposed independent `setup`, `dense`, and `bm25` operations through the turbopuffer CLI. Setup always creates A/B/C/D, while search can run all distributions in deterministic interleaved order or select one group. The setup manifest is version 2 and stores the dense and BM25 field contract. Search optionally fetches declared output fields, validates them, and discards their values.

Added a dedicated sequential search runner that records `started` before every first/repeat query and writes a terminal event afterward. It does not retry measured queries at either the benchmark or turbopuffer SDK layer. Interrupted requests become indeterminate; a repeat that has not started after a completed first request remains safe to resume. Query errors are recorded without payloads or exception messages, later namespaces continue, and the invocation reports incomplete after preserving its artifacts.

Mode-specific JSONL artifacts contain per-namespace outcomes, client timing, turbopuffer performance metadata, and result counts. Compact summaries group first and repeat observations by A/B/C/D and report outcomes plus client, server-total, and query-execution average/P50/P95/P99. Dense and BM25 reject reuse of the same setup manifest so both cannot silently claim cold measurements against namespaces already queried by the other mode.

Verified on the assigned remote worktree: 19 focused Stage 1-3 client, setup, CLI/config, search, resume, aggregation, and serialization tests passed. A CLI help check and dry-run also exposed the new options and produced a task with no generic VDBBench stages. No live turbopuffer request was issued.
- Files: `vectordb_bench/backend/turbopuffer_multitenant.py` (updated), `vectordb_bench/backend/cases.py` (updated), `vectordb_bench/backend/assembler.py` (updated), `vectordb_bench/backend/task_runner.py` (updated), `vectordb_bench/backend/clients/turbopuffer/cli.py` (updated), `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py` (updated), `vectordb_bench/cli/cli.py` (updated), `vectordb_bench/models.py` (updated), `tests/test_turbopuffer_multitenant_case.py` (created), `tests/test_turbopuffer_multitenant_setup.py` (updated), `tests/test_turbopuffer_customized.py` (updated), `docs/turbopuffer-scaled-multitenant-design.md` (updated), `AGENTS.md` (updated, repository-local and ignored)

### 18. Reduce namespace counts with setup profiles
Replaced the 4,201-namespace distribution with equal A/B/C sample counts selected by one setup-only `--multitenant-profile` option. Small creates 20 namespaces per A/B/C group plus D (61 namespaces and 5.38M rows), Medium creates 100 per group plus D (301 namespaces and 6.9M rows), and Large creates 300 per group plus D (901 namespaces and 10.7M rows). Medium is the default. Namespace row sizes, schema, fixtures, sequential first/repeat behavior, search group selection, and separate cold prefixes for dense and BM25 remain unchanged.

The manifest is version 3 and records the resolved profile. Setup and search summaries now report the profile and actual total rows. Search remains manifest-driven, so users do not repeat the profile during dense or BM25 operations. The case's generic dataset descriptor now identifies the fixed 5M prepared source rather than reporting a hardcoded logical population.

Added a standalone run README covering prerequisites, Small pilot commands, dense and BM25 setup/search, group-specific runs, returned attributes, artifact paths, resume rules, expected outputs, and percentile interpretation. Updated the design to make equal per-size sampling and the three profiles authoritative.

Verified on the assigned remote worktree: 20 focused tests passed. CLI help exposed `--multitenant-profile [small|medium|large]`, and a setup dry-run carried `profile=small` into the case configuration without issuing a live request. A footer-only check against the real prepared Parquet resolved Small/Medium/Large to 61/301/901 namespaces and 5.38M/6.9M/10.7M inserted rows; Large's maximum source span is 5M rows, matching the prepared file.
- Files: `vectordb_bench/backend/turbopuffer_multitenant.py` (updated), `vectordb_bench/backend/cases.py` (updated), `vectordb_bench/backend/task_runner.py` (updated), `vectordb_bench/backend/clients/turbopuffer/cli.py` (updated), `vectordb_bench/cli/cli.py` (updated), `tests/test_turbopuffer_multitenant_setup.py` (updated), `tests/test_turbopuffer_multitenant_case.py` (updated), `docs/turbopuffer-scaled-multitenant-design.md` (updated), `docs/turbopuffer-multitenant/README.md` (created), `AGENTS.md` (updated, repository-local and ignored)

### 19. Add a setup flag to skip the 5M-row namespace
Added the setup-only `--multitenant-exclude-5m` flag so the 5M-row D namespace can be skipped for cheaper pilots. A/B/C counts are unchanged; the profile reports as `small-no-5m`, `medium-no-5m`, or `large-no-5m` (Small becomes 60 namespaces / 380K rows). The manifest version is now 4 and records the resolved groups; setup resumes reject mismatched manifests and search reads namespaces from the manifest, so dense/BM25 commands need no change. The flag is recorded on the case config, validated to setup-only, and documented in the run README.

Also recorded the turbopuffer API token in git-ignored `turbopuffer_api_key.txt` files on the local and remote worktrees (mode 0600) and pointed the task `AGENTS.md` at them, unblocking the stage-4 pilot pending region confirmation.

Verified on the assigned remote worktree: 21 focused tests passed, ruff check introduced no new violations (33 pre-existing baseline unchanged), ruff format is clean for the new lines (4 pre-existing drift files left untouched), CLI help exposes the flag, and a setup dry-run carried `exclude_5m=True`. The real prepared Parquet resolves Small-no-5m to 60 namespaces and 380,000 rows. Changes are uncommitted on top of `d6c0f56`.
- Files: `vectordb_bench/backend/turbopuffer_multitenant.py` (updated), `vectordb_bench/backend/clients/turbopuffer/cli.py` (updated), `vectordb_bench/cli/cli.py` (updated), `vectordb_bench/backend/cases.py` (updated), `vectordb_bench/backend/task_runner.py` (updated), `tests/test_turbopuffer_multitenant_setup.py` (updated), `tests/test_turbopuffer_multitenant_case.py` (updated), `docs/turbopuffer-multitenant/README.md` (updated), `turbopuffer_api_key.txt` (created, ignored), `AGENTS.md` (updated, repository-local and ignored)

### 20. Run the stage-4 disposable Small pilot (live)
Ran the live pilot on the remote client against real turbopuffer in `aws-us-west-2` with the Small profile and `--multitenant-exclude-5m` (60 namespaces, 380,000 rows, no D).

Setup completed cleanly: `profile=small-no-5m`, `inserted_rows=380000`, `completed_namespaces=60/60`, 120 checkpoint events, 60 fixtures, manifest version 4 (groups A/B/C). Dense search over all groups completed: 60/60 namespaces × first+repeat passes, 0 errors, 240 completed events. Latency p50 (first/repeat ms): A 20.9/19.1, B 20.9/19.4, C 18.4/16.0; every event carries turbopuffer performance metadata (cache_temperature, cache_hit_ratio, server_total_ms, query_execution_ms). Security scan found no API token and no full-row payloads in the log, manifest, checkpoints, fixtures, or search artifacts; the API key is masked in logs. First-query cache_temperature was already "hot" (setup leaves namespaces warm), so cold-vs-warm separation should be interpreted with the backend-reported cache state.

Artifacts under `/home/ubuntu/vdbbench-data-inspect/turbopuffer-runs/`: `dense-small-no5m-pilot.json` (manifest), `.checkpoints.jsonl`, `.fixtures/` (60), `dense-small-no5m-pilot.dense.search.jsonl`, `dense-small-no5m-pilot.dense.all.summary.json`. Live namespaces `tp_mt_small_no5m_pilot_20260917_*` (60) remain in turbopuffer pending user cleanup decision; the design forbids automatic namespace deletion.
- Files: `SESSION_PROGRESS.md` (updated), `AGENTS.md` (updated, repository-local and ignored)

### 21. Switch to a shared 100-query out-of-sample set with cold/warm passes
Re-accessed the private S3 bucket (credentials still valid; 2,000 objects) and confirmed turbopuffer has no namespace-eviction API (`hint_cache_warm` and pinning only warm). Downloaded `wide_table_0010.parquet` (the first file beyond the 5M rows used for inserts) and added `scripts/extract_turbopuffer_multitenant_queries.py` to extract its first 100 `emb_768` vectors plus `content` strings into a deterministic JSON query file (`/home/ubuntu/vdbbench-data-inspect/turbopuffer-queries/queries.json`), so no query vector exists inside any namespace.

Reworked the case to use the shared query set: manifest v5 records `queries_file`/`query_count` and setup copies it to a `<manifest>.queries.json` sidecar via the new setup-only `--multitenant-queries-file` option. The search runner now queries every namespace with all 100 queries in pass `first` (query 0 is the cold sample, 1-99 the warm-up ramp) then all 100 again in pass `repeat`; events carry `query_index`, resume is per (namespace, pass, query), and per-group summaries report `cold` (query 0 of pass `first`) and `warm` (all of pass `repeat`) buckets with client/server latency percentiles. Per-namespace query fixtures were removed. Updated cases.py, both CLIs, task_runner.py, tests, README, and the design doc.

Verified on the assigned remote worktree: 28 focused tests passed (setup, search, resume, CLI config, extraction), ruff introduced no new violations in shared code (module diff is empty vs HEAD), the new script matches the sibling prepare script's accepted baseline (7 EM102 + 1 T201), and a setup dry-run carried `queries_file` end to end.

Calibrated cold eviction on the old pilot namespaces (idle ~1.5-2h): every namespace pays a one-time 58-96 ms `server_total_ms` on its first query versus 10-20 ms warm (4-6x cold/warm ratio), but `cache_temperature`/`cache_hit_ratio` report `hot`/`1.0` even on that load query, so server latency is the cold discriminator, not the reported cache state. The original pilot's "cold" numbers (~18-21 ms) were therefore warm measurements.
- Files: `scripts/extract_turbopuffer_multitenant_queries.py` (created), `tests/test_extract_turbopuffer_multitenant_queries.py` (created), `vectordb_bench/backend/turbopuffer_multitenant.py` (updated), `vectordb_bench/backend/clients/turbopuffer/cli.py` (updated), `vectordb_bench/cli/cli.py` (updated), `vectordb_bench/backend/cases.py` (updated), `vectordb_bench/backend/task_runner.py` (updated), `tests/test_turbopuffer_multitenant_setup.py` (updated), `tests/test_turbopuffer_multitenant_case.py` (updated), `docs/turbopuffer-multitenant/README.md` (updated), `docs/turbopuffer-scaled-multitenant-design.md` (updated), `SESSION_PROGRESS.md` (updated)

### 22. Run the cold/warm 100-query pilot (live, with eviction wait)
Ran the fresh Small no-5m pilot on the remote client: 60 namespaces, 380K rows, prefix `tp_mt_small_no5m_100q_20260917`, 100-query shared out-of-sample set, dense mode. Setup completed at 06:01:59 (`profile=small-no-5m`, 60/60 namespaces, `queries=100`), then a 60-minute idle wait for cache eviction, then the search ran 24,000 measured queries (100 queries × 2 passes × 60 namespaces) with zero errors in ~4 minutes.

Results (all 60 namespaces completed): three summary buckets per group — `cold` (query 0 of pass `first`, 20/group — the only genuinely cold query per namespace), `first` (full cold pass, 2,000/group), `repeat` (warm pass, 2,000/group) — each with client/server min/max/avg/p50/p95/p99:
- Cold (query 0) client p50 / server avg: A 71.1 / 70.0 ms, B 78.6 / 76.0 ms, C 80.5 / 81.1 ms; client range 43-156 ms.
- First pass client p50 / p99: A 17.1 / 54.3 ms, B 17.8 / 82.0 ms, C 14.1 / 61.2 ms — diluted by 99 warm samples per namespace; the cold signal appears in max/p99.
- Repeat (warm) pass client p50 / p99: A 16.7 / 41.7 ms, B 17.6 / 51.5 ms, C 14.1 / 35.0 ms.
- Truly-cold vs warm ratio ~4.6-5.7x (cold p50 vs repeat p50); cold scales mildly with namespace size (70 -> 76 -> 81 ms server avg), warm is flat.
- The 60-minute wait was sufficient for eviction (shorter than the >=1.5h observed earlier); cold samples carry the one-time load cost.
- Every event reported `cache_temperature=hot`/`cache_hit_ratio=1.0`; under the AGENTS.md policy those fields are the sole cold/warmness indicator (latency metrics are recorded but not used to classify cache state).

Security scan clean: no API token and no full-row payload values in the manifest, queries sidecar, checkpoints, search JSONL, summary, or logs. Artifacts: `dense-small-no5m-100q.json`, `.checkpoints.jsonl`, `.queries.json`, `.dense.search.jsonl` (24,001 lines), `.dense.all.summary.json`.
- Files: `SESSION_PROGRESS.md` (updated), `AGENTS.md` (updated, repository-local and ignored)

Direct fresh-namespace test (2026-09-17): inserting a brand-new namespace (1K and 15K rows) and querying it immediately yields a WARM first query — `server_total_ms` 18-20 ms vs 8-22 ms warm baseline and 39-153 ms for idle-evicted cold. Writing warms the namespace; the 60-minute idle wait after setup is therefore essential for genuine cold measurements. Test namespaces `tp_mt_fresh_cold_test_20260917_a1k` / `_c15k` were created.

### 23. Refactor to single-size namespaces with a disable_cache first pass
Probed turbopuffer's undocumented `disable_cache` query flag (issue #3; branch_from issue #2 closed as superseded): honored for our key, deterministic `cache_temperature="cold"`/`cache_hit_ratio=0.0` on every flagged query, ANN and BM25, at 1K/15K/100K rows (67-80 / 511 / 410-414 ms server), immediate un-flagged repeat is `hot`/1.0; `_debug/purge_cache`/`warm_cache` return 200 but purge has no observable effect. At equal size the flagged cold matches a fresh branch's first query (511 vs 525 ms @ 15K), so the flag supersedes both the idle-eviction wait (whose samples reported `hot`/1.0 and were unclassifiable as cold under the policy) and the branch_from hack.

Refactored the case per user decisions: each setup run creates exactly ONE namespace of `--multitenant-namespace-rows` rows (int, default 15000; `--multitenant-namespace-count` dropped after the user clarified one namespace per size — run setup once per size, e.g. 1K/10K/15K/5M, with fresh prefixes); namespaces named `{run_prefix}_{rows}_0001`; A/B/C/D groups, `--multitenant-group`, `--multitenant-profile`, `--multitenant-exclude-5m`, and the multi-namespace count removed; warmup machinery untouched (the cold-start case never ran it — stages stay empty). Pass `first` now sends `disable_cache: true` (genuinely cold), pass `repeat` omits it (warm); summaries have top-level `first`/`repeat` buckets plus `rows_per_namespace`/`total_rows`; manifest v7 (no `groups`/`profile`/`namespace_count`).

Verified on the assigned remote worktree: 39 focused tests passed (setup, case, CLI config, extraction; two rewritten suites), ruff 33 vs HEAD 40 on the touched files (no new violations; removed code cleaned up 7), a CLI dry-run carried `namespace_rows=15000` into the task with no group/profile/exclude-5m/namespace-count. `docs/turbopuffer-multitenant/README.md` rewritten for the new design. Changes uncommitted on top of `df0a295`.
- Files: `vectordb_bench/backend/turbopuffer_multitenant.py` (updated), `vectordb_bench/backend/customized.py` (updated), `vectordb_bench/backend/clients/turbopuffer/turbopuffer.py` (updated), `vectordb_bench/backend/clients/turbopuffer/cli.py` (updated), `vectordb_bench/cli/cli.py` (updated), `vectordb_bench/backend/cases.py` (updated), `vectordb_bench/backend/task_runner.py` (updated), `tests/test_turbopuffer_multitenant_setup.py` (rewritten), `tests/test_turbopuffer_multitenant_case.py` (rewritten), `tests/test_turbopuffer_customized.py` (updated), `docs/turbopuffer-multitenant/README.md` (rewritten), `docs/turbopuffer-multitenant/disable_cache-finding.md` (updated), `docs/turbopuffer-scaled-multitenant-design.md` (updated), `SESSION_PROGRESS.md` (updated)

### 24. Live multi-size dense run (1K/10K/15K × 100 cold + 100 hot)
Ran the new one-namespace-per-size design live on the remote client: three setups (one namespace each, prefixes `tp_mt_dense_{size}_20260920`) and three dense searches (k=100). All 600 measured queries completed with zero errors; every first-pass query reported `cold`/0.0 and every repeat `hot`/1.0 (100/100 per bucket per size) — the first run fully policy-compliant without any idle wait.

Client p50 / server p50 (ms): cold 61.2/58.0 (1K), 80.4/77.0 (10K), 275.8/272.0 (15K); hot 14.5/11.0, 17.6/15.0, 12.4/9.0. Cold scales with size (15K jumps — data-transfer-bound at 46 MB); warm is flat (in-memory index). Cold/warm client-p50 ratios ~4.2x / 4.6x / 22x. Results posted to task tracker issue #1; artifacts `turbopuffer-runs/dense-{1000,10000,15000}.dense.summary.json`. Namespaces left for cleanup.
- Files: `SESSION_PROGRESS.md` (updated)

### 25. 5M-row namespace: insert (28.32 GB) + dense cold/hot run
Inserted one 5M-row namespace (`tp_mt_dense_5m_20260920_5000000_0001`) with `--insert-batch-size 1000 --disable-backpressure`. First attempt failed: turbopuffer throttled writes with 429 "indexing backlog" once the backlog built (~660K rows in, setup exhausted retries). The documented fix (`disable_backpressure: true` in the upsert) worked: full 5M insert finished in ~70 min, `approx_logical_bytes` = **28.32 GB** (~5.66 KB/row, confirming the projection). Indexed to `up-to-date` before measuring.

Dense results (100 cold + 100 hot): cold server p50 787 ms / p99 1320 / max 1831, 100/100 `cold`/0.0; repeat server p50 187.5 ms but cache fields mixed — `cold`(71) + `warm`(29), hit_ratio 0.83. Key finding: at 5M the repeat pass does NOT fully warm (28 GB exceeds cache capacity), so per-event cache fields must classify the warm side; the cold/warm p50 ratio drops to ~4.2x vs 15K's ~22x. Posted to issue #1. The job wrapper hung on the CLI pipe after completion (remote work saved) and was killed. Namespaces left for cleanup.
- Files: `SESSION_PROGRESS.md` (updated)

### 26. Warm reattempt correction at 5M (2026-09-20)
Re-ran 100 warm queries (no flag) on the 5M namespace ~40 min after the repeat pass: 100/100 `hot`/1.0, server p50 10 ms / p99 20 / max 32. The earlier repeat pass (`cold` 71 / `warm` 29, 187 ms p50) was a warm-up transient — the `disable_cache` first pass does not populate the cache, so the immediate repeat started cold and warmed progressively. Corrected 5M cold/warm ratio ~79x (787 ms vs 10 ms), consistent with small sizes. **Case implication**: repeat pass needs a warm-up phase (discard first warm pass or add a warm-up ramp) at large scale; small sizes unaffected. Correction posted to issue #1.
- Files: `SESSION_PROGRESS.md` (updated)

### 27. Payload sweep + same-manifest payload re-runs (2026-09-21)
Payload sweep at 15K (P1-P4) showed returning up to 307 KB/query has no measurable latency cost (fresh P0 control 92.5 ms cold p50 vs P1-P4 86-98 ms; warm 15-18 vs 16) — the earlier 272 ms 15K cold was run-to-run variance. Sweep at 1K/10K also completed (8 namespaces). At 5M, the per-config fresh-namespace requirement (output_fields baked into the search checkpoint header) would have meant 4 x 28 GB re-inserts; the 5M part was killed (partial tp_mt_payload_5000000_p1_20260921_5000000_0001, 751K rows, left for cleanup) and the design was fixed instead: the search header now treats output_fields as a re-runnable variant — same manifest re-searches with a new payload (fresh event file + full re-run, no re-insert), with per-payload summaries ({manifest}.{mode}.{fields}.summary.json). 40 focused tests pass. Payload runs at 1K/10K summaries saved; 5M payloads can now run on the existing 5M namespace with no insert.
- Files: `vectordb_bench/backend/turbopuffer_multitenant.py` (updated), `tests/test_turbopuffer_multitenant_case.py` (updated), `docs/turbopuffer-multitenant/README.md` (updated), `SESSION_PROGRESS.md` (updated)

### 28. 5M payload sweep on the existing namespace (2026-09-21)
Ran P0-P4 on the existing 5M namespace via the same manifest (no re-insert; the output_fields decoupling from stage 27 worked). Per-payload summaries preserved: dense-5m.dense{.fields}.summary.json. Server p50 cold: P0 787, P1 982, P2 1271, P3 1174, P4 982 ms; warm P1-P4 all 100/100 hot at 11-13 ms (P0's own repeat still shows the 5M warm-up transient: 187 ms, cold 71 / warm 29). Cold with payloads is +25-60% over baseline at 5M but not monotonic in payload size — run-to-run variance dominates; warm is unaffected. Cold/warm ratios at 5M with payloads ~89-107x. Posted to issue #4. Findings in `docs/turbopuffer-multitenant/disable_cache-finding.md` unchanged; SESSION_PROGRESS current.

### 29. BM25 can share manifests with dense (2026-09-21)
Removed the obsolete "dense and BM25 require separate setup manifests" guard: under disable_cache the cold pass is forced regardless of prior mode, so bm25 runs on the same manifest/namespace as dense with no re-insert. Replaced the fresh-manifest-per-mode test with both-modes-share-one-manifest (dense then bm25 on the same manifest; bm25 first-pass disable_cache True, repeat False). README BM25 section rewritten. 40 focused tests pass. Committed 866fcc7 + pushed to fork.

## Current Status
The design and implementation are recorded and mostly committed on `codex/turbopuffer_multitenant` (pushed to the fork, no PR yet): single-size namespaces + `disable_cache` first pass (manifest v7), payload variants re-runnable on the same manifest, and dense/BM25 share manifests. Measured live: dense size ladder 1K/10K/15K/5M cold vs warm, payload matrix at all sizes (free at <=15K; elevated-but-variance-bound at 5M), 5M storage 28.32 GB confirmed. Remaining: BM25 runs, optional cold-payload repeats at 5M, cleanup, and PR. Stages 1-3 plus the exclude-5m flag are committed through `1225e19c165f667059aa3bab471db01cee351229` (5 commits ahead of `origin/main`, not pushed). The shared 100-query cold/warm design (manifest v5) is implemented and remotely verified (28 tests), and the fresh Small no-5m pilot PASSED — but its "cold" samples reported `hot`/1.0 and could not be labeled cold under the AGENTS.md policy. The disable_cache probe (issue #3) then proved turbopuffer honors an undocumented `disable_cache` query flag for our key (deterministic `cold`/0.0, ANN + BM25, 1K/15K/100K rows), and the refactor to single-size namespaces + `disable_cache` first pass (manifest v6) is implemented and remotely verified (39 tests, ruff 34 vs HEAD 40, CLI dry-run carries `namespace_rows`/`namespace_count`). The refactor and pilot records remain uncommitted. No Medium or Large load has been run, and the 5M-row cold magnitude is unmeasured.

## Open Issues
- Decide whether to delete the disposable pilot and probe namespaces (`tp_mt_small_no5m_pilot_20260917_*`, `tp_mt_small_no5m_100q_20260917_*`, `tp_mt_dc_probe_20260918_a1k`, `tp_mt_cmp_probe_20260918_c15k*`, `tp_mt_scale_probe_20260918_c100k`) from turbopuffer (the design forbids automatic deletion).
- Confirm the disable_cache cold magnitude at 5M rows (round-trip-dominated scaling makes extrapolation unreliable) before a large run.
- Confirm with the wider run that the projected wide schema and deterministic IDs remain accepted at large scale.
- Cold/warm classification follows the AGENTS.md policy: turbopuffer's `cache_temperature`/`cache_hit_ratio` are the SOLE indicators; latency metrics are recorded but not used to label cache state. `disable_cache` is an undocumented API — stability unverified; the branch_from numbers (issue #2, closed) remain the fallback record.
