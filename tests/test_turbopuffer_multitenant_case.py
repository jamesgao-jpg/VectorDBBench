import json
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
from click import UsageError

from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import CaseLabel, CaseType, TurboPufferMultiTenantColdStartCase
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import EmptyDBCaseConfig
from vectordb_bench.backend.customized import SearchPerformance, SearchResult
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.turbopuffer_multitenant import (
    MANIFEST_VERSION,
    SEARCH_RESULT_VERSION,
    MultiTenantSearchIncomplete,
    MultiTenantSearchRunner,
    customized_schema,
)
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, TaskConfig, TestResult


TEST_QUERIES = [
    {"index": 0, "dense": [1.0] * 768, "bm25": "query-0"},
    {"index": 1, "dense": [2.0] * 768, "bm25": "query-1"},
]


def _write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "setup.json"
    namespaces = [
        {"name": "run_1_01", "rows": 1, "source_start": 0, "source_end": 1},
    ]
    (tmp_path / "setup.queries.json").write_text(
        json.dumps(
            {
                "version": 1,
                "count": len(TEST_QUERIES),
                "dense_field": "emb_768",
                "bm25_field": "content",
                "queries": TEST_QUERIES,
            },
            separators=(",", ":"),
        )
    )
    manifest.write_text(
        json.dumps(
            {
                "version": MANIFEST_VERSION,
                "rows_per_namespace": 1,
                "schema": {name: asdict(field) for name, field in customized_schema("emb_768", "content").items()},
                "search_fields": {"dense": "emb_768", "bm25": "content"},
                "queries_file": "setup.queries.json",
                "query_count": len(TEST_QUERIES),
                "namespaces": namespaces,
                "search_order": ["run_1_01"],
                "checkpoint_file": "setup.checkpoints.jsonl",
            },
            separators=(",", ":"),
        )
    )
    (tmp_path / "setup.checkpoints.jsonl").write_text(
        "".join(
            json.dumps({"namespace": entry["name"], "state": "completed"}) + "\n" for entry in namespaces
        )
    )
    return manifest


class _SearchDB:
    def __init__(self, fail_namespaces: set[str] | None = None):
        self.active_namespace = None
        self.calls = []
        self.fail_namespaces = fail_namespaces or set()

    @classmethod
    def supports_customized_api(cls) -> bool:
        return True

    @classmethod
    def supports_namespace_selection(cls) -> bool:
        return True

    @contextmanager
    def init(self):
        yield

    def select_namespace(self, namespace: str) -> None:
        self.active_namespace = namespace

    def search_customized_queries(self, requests):  # noqa: ANN001
        request = requests[0]
        self.calls.append((self.active_namespace, request))
        if self.active_namespace in self.fail_namespaces:
            raise RuntimeError("injected failure")
        return [
            SearchResult(
                ids=[1],
                fields={name: [f"returned-{name}"] for name in request.include_fields},
                performance=SearchPerformance(
                    cache_hit_ratio=0.25,
                    cache_temperature="cold",
                    server_total_ms=12,
                    query_execution_ms=9,
                ),
            )
        ]


def test_multitenant_search_runs_first_and_repeat_in_manifest_order(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    db = _SearchDB()

    summary = MultiTenantSearchRunner(
        db,
        manifest,
        "dense",
        output_fields=("vc_uuid",),
        top_k=10,
    ).run()

    assert [namespace for namespace, _ in db.calls] == ["run_1_01", "run_1_01", "run_1_01", "run_1_01"]
    assert all(call.top_k == 10 and call.include_fields == ("vc_uuid",) for _, call in db.calls)
    assert summary["status"] == "complete"
    assert summary["rows_per_namespace"] == 1
    assert summary["query_count"] == 2
    assert summary["total_rows"] == 1
    assert "groups" not in summary
    assert summary["first"]["count"] == 2
    assert summary["repeat"]["count"] == 2
    assert summary["first"]["server_total_ms"]["average_ms"] == 12
    assert summary["first"]["query_execution_ms"]["average_ms"] == 9
    assert "min_ms" in summary["first"]
    assert "max_ms" in summary["first"]
    assert "p99_ms" in summary["repeat"]
    assert summary["first"]["cache_temperature"] == {"cold": 2}
    assert summary["first"]["cache_hit_ratio"]["average_ms"] == 0.25
    assert summary["cache_temperature"] == {"cold": 4}
    assert summary["cache_hit_ratio"]["count"] == 4
    event_text = Path(summary["event_path"]).read_text()
    assert "query-0" not in event_text
    assert '"cache_hit_ratio":0.25' in event_text


def test_multitenant_search_uses_disable_cache_only_on_first_pass(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    db = _SearchDB()

    MultiTenantSearchRunner(db, manifest, "dense").run()

    assert [(call.disable_cache, call.mode) for _, call in db.calls] == [
        (True, "dense"),
        (True, "dense"),
        (False, "dense"),
        (False, "dense"),
    ]


def test_multitenant_search_reruns_same_manifest_with_new_payload(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    db = _SearchDB()

    first = MultiTenantSearchRunner(db, manifest, "dense", output_fields=("vc_uuid",), top_k=10).run()
    assert [namespace for namespace, _ in db.calls].count("run_1_01") == 4
    assert first["status"] == "complete"
    first_summary = Path(first["summary_path"])
    assert first_summary.is_file()

    second = MultiTenantSearchRunner(db, manifest, "dense", output_fields=("content",), top_k=10).run()
    assert second["status"] == "complete"
    assert [namespace for namespace, _ in db.calls].count("run_1_01") == 8  # all queries re-ran
    assert Path(second["summary_path"]).is_file()
    assert second["summary_path"] != first["summary_path"]
    assert first_summary.is_file()  # prior payload summary preserved
    assert second["first"]["cache_temperature"] == {"cold": 2}


def test_multitenant_search_does_not_retry_and_preserves_failure_summary(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    db = _SearchDB({"run_1_01"})
    runner = MultiTenantSearchRunner(db, manifest, "dense")

    with pytest.raises(MultiTenantSearchIncomplete) as failure:
        runner.run()

    assert [namespace for namespace, _ in db.calls].count("run_1_01") == 2
    assert failure.value.summary["status"] == "incomplete"
    assert failure.value.summary["first"]["outcomes"]["error"] == 2
    assert failure.value.summary["repeat"]["outcomes"]["skipped"] == 2
    assert Path(failure.value.summary["summary_path"]).is_file()


def test_multitenant_search_marks_interrupted_query_indeterminate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    db = _SearchDB()
    runner = MultiTenantSearchRunner(db, manifest, "dense")
    header = {
        "event": "header",
        "version": SEARCH_RESULT_VERSION,
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "mode": "dense",
        "search_field": "emb_768",
        "output_fields": [],
        "top_k": 100,
        "query_count": 2,
    }
    runner.event_path.write_text(
        json.dumps(header, separators=(",", ":"))
        + "\n"
        + json.dumps({"event": "started", "namespace": "run_1_01", "pass": "first", "query_index": 0})
        + "\n"
    )

    with pytest.raises(MultiTenantSearchIncomplete) as failure:
        runner.run()

    assert [namespace for namespace, _ in db.calls] == ["run_1_01", "run_1_01"]
    assert failure.value.summary["first"]["outcomes"] == {"indeterminate": 1, "completed": 1}
    assert failure.value.summary["repeat"]["outcomes"] == {"skipped": 1, "completed": 1}


def test_multitenant_search_resumes_repeat_that_never_started(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    db = _SearchDB()
    runner = MultiTenantSearchRunner(db, manifest, "dense")
    runner.event_path.write_text(
        json.dumps(runner._header(), separators=(",", ":"))
        + "\n"
        + json.dumps(
            {
                "event": "completed",
                "namespace": "run_1_01",
                "pass": "first",
                "query_index": 0,
                "client_latency_ms": 1,
                "result_count": 1,
                "performance": {},
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "completed",
                "namespace": "run_1_01",
                "pass": "first",
                "query_index": 1,
                "client_latency_ms": 1,
                "result_count": 1,
                "performance": {},
            }
        )
        + "\n"
    )

    summary = runner.run()

    # run_1_01's first pass was already completed in the checkpoint file, so only its
    # repeat queries execute here.
    assert [(namespace, request.mode) for namespace, request in db.calls] == [
        ("run_1_01", "dense"),
        ("run_1_01", "dense"),
    ]
    assert all(request.disable_cache is False for _, request in db.calls)
    assert summary["status"] == "complete"


def test_multitenant_search_both_modes_share_one_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    dense = MultiTenantSearchRunner(_SearchDB(), manifest, "dense").run()
    assert dense["status"] == "complete"

    db = _SearchDB()
    bm25 = MultiTenantSearchRunner(db, manifest, "bm25").run()
    assert bm25["status"] == "complete"
    assert [(call.mode, call.disable_cache) for _, call in db.calls] == [
        ("bm25", True),
        ("bm25", True),
        ("bm25", False),
        ("bm25", False),
    ]


def test_multitenant_case_and_cli_config() -> None:
    custom_case = get_custom_case_config(
        {
            "case_type": "TurboPufferMultiTenantColdStart",
            "multitenant_operation": "dense",
            "multitenant_manifest": "/tmp/setup.json",
            "multitenant_output_fields": ["vc_uuid", "vc_tag"],
        }
    )
    case = CaseConfig(case_id=CaseType.TurboPufferMultiTenantColdStart, custom_case=custom_case).case

    assert isinstance(case, TurboPufferMultiTenantColdStartCase)
    assert case.label == CaseLabel.TurboPufferMultiTenantColdStart
    assert case.operation == "dense"
    assert case.output_fields == ("vc_uuid", "vc_tag")
    assert case.dataset.data.name == "TurbopufferMultiTenantSource"
    assert case.dataset.data.size == 5_000_000


def test_multitenant_setup_cli_requires_data_and_prefix() -> None:
    custom_case = get_custom_case_config(
        {
            "case_type": "TurboPufferMultiTenantColdStart",
            "multitenant_operation": "setup",
            "multitenant_manifest": "/tmp/setup.json",
            "multitenant_output_fields": [],
            "multitenant_prepared_data": None,
            "multitenant_run_prefix": None,
            "multitenant_dense_field": "emb_768",
            "multitenant_bm25_field": "content",
            "multitenant_namespace_rows": 15_000,
        }
    )

    assert custom_case["namespace_rows"] == 15_000

    with pytest.raises(ValueError, match="setup requires prepared_data and run_prefix"):
        CaseConfig(case_id=CaseType.TurboPufferMultiTenantColdStart, custom_case=custom_case).case

    with pytest.raises(UsageError, match="operation and --multitenant-manifest"):
        get_custom_case_config(
            {
                "case_type": "TurboPufferMultiTenantColdStart",
                "multitenant_operation": None,
                "multitenant_manifest": None,
            }
        )


def test_multitenant_setup_cli_carries_namespace_rows() -> None:
    custom_case = get_custom_case_config(
        {
            "case_type": "TurboPufferMultiTenantColdStart",
            "multitenant_operation": "setup",
            "multitenant_manifest": "/tmp/setup.json",
            "multitenant_output_fields": [],
            "multitenant_prepared_data": "/tmp/prepared.parquet",
            "multitenant_run_prefix": "run",
            "multitenant_dense_field": "emb_768",
            "multitenant_bm25_field": "content",
            "multitenant_namespace_rows": 2,
            "multitenant_queries_file": "/tmp/queries.json",
        }
    )

    assert custom_case["namespace_rows"] == 2
    assert custom_case["queries_file"] == "/tmp/queries.json"
    case = CaseConfig(case_id=CaseType.TurboPufferMultiTenantColdStart, custom_case=custom_case).case
    assert isinstance(case, TurboPufferMultiTenantColdStartCase)
    assert case.namespace_rows == 2
    assert case.queries_file == "/tmp/queries.json"

    with pytest.raises(ValueError, match="applies only to the setup operation"):
        CaseConfig(
            case_id=CaseType.TurboPufferMultiTenantColdStart,
            custom_case={
                "operation": "dense",
                "manifest_path": "/tmp/setup.json",
                "namespace_rows": 2,
            },
        ).case

    with pytest.raises(ValueError, match="setup requires queries_file"):
        CaseConfig(
            case_id=CaseType.TurboPufferMultiTenantColdStart,
            custom_case={
                "operation": "setup",
                "manifest_path": "/tmp/setup.json",
                "prepared_data": "/tmp/prepared.parquet",
                "run_prefix": "run",
                "namespace_rows": 2,
            },
        ).case


def test_multitenant_result_serializes_compact_summary() -> None:
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.TurboPufferMultiTenantColdStart,
            custom_case={
                "operation": "dense",
                "manifest_path": "/tmp/setup.json",
            },
        ),
        stages=[],
    )
    result = TestResult(
        run_id="run",
        task_label="multitenant",
        results=[
            CaseResult(
                task_config=task,
                metrics=Metric(additional_parameters={"turbopuffer_multitenant": {"status": "complete"}}),
            )
        ],
    ).model_dump_for_output()

    assert result["results"][0]["metrics"] == {
        "inserted_count": 0,
        "turbopuffer_multitenant": {"status": "complete"},
    }

    with pytest.raises(ValueError, match="supports only the TurboPuffer backend"):
        Assembler.assemble("run", task, DatasetSource.S3)
