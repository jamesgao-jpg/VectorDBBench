# Turbopuffer vs Zilliz — Reproducibility Bundle

Companion code for `turbopuffer_vs_zilliz_final_report.md`
(`github.com/jamesgao-jpg/turbopuffer_test_results`).

- **VDBBench branch / commit:** `tp_vs_zilliz_0415` @ `a177dbf`
- **Driver host:** `m6i.2xlarge` in `aws-us-west-2`
- **Dataset cache:** `/tmp/vectordb_bench/dataset/{laion,cohere}/...` (auto-downloaded from `s3://assets.zilliz.com/benchmark/`)

| Folder | Report § |
|--------|----------|
| `insertion/` | § 1 |
| `cost/`      | § 3 |
| `multi_tenant/` | § 4 |
| `delete_consistency/` | § 5 |

All env-var toggles on this VDBBench commit default to off.
Full canonical commands per section: see `reproducibility_guide.md`
in the results repo.
