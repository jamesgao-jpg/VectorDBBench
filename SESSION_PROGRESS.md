# Session Progress

**Date:** 2026-09-14
**Working Directory:** /Users/james.gao/Desktop/VectorDBBench

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

## Current Status
The design and four implementation stages are recorded. Stage 1's customized-data contracts and turbopuffer implementation are present in matching local and remote `codex/turbopuffer_multitenant` worktrees based on latest `origin/main`. The source/data pipeline and benchmark case are not implemented, and no live turbopuffer namespace has been created.

## Open Issues
- Configure a turbopuffer API key and region on the remote client before the stage-4 live pilot; keep them out of files, logs, and results.
- Confirm with the stage-4 live pilot that the projected wide schema and deterministic IDs are accepted by turbopuffer.
- Validate the schema and row counts of only the S3 objects consumed by the 14M-row setup.
- The second pass is an after-one-query observation, not guaranteed cache residency; use the backend-reported cache state when interpreting it.
