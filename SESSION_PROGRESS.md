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

## Current Status
The design and four implementation stages are recorded. Stages 1-3 are committed through `7caa730305145078df05d237c043df879c46c96c`. The profile refinement and run README are implemented and remotely verified but uncommitted. The 5M-row prepared artifact remains available on the remote client. No live turbopuffer namespace has been created.

## Open Issues
- Configure a turbopuffer API key and region on the remote client before the stage-4 live pilot; keep them out of files, logs, and results.
- Confirm with the stage-4 live pilot that the projected wide schema and deterministic IDs are accepted by turbopuffer.
- The second pass is an after-one-query observation, not guaranteed cache residency; use the backend-reported cache state when interpreting it.
