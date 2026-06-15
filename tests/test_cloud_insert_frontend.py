import json
from pathlib import Path

import pytest

from vectordb_bench.backend.result_collector import ResultCollector
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.cloudleaderboard.insert import (
    CloudInsertParseError,
    cloud_insert_records,
    load_cloud_insert_rows,
)
from vectordb_bench.frontend.pages.cloud_insert import (
    PAGE_CASE_CAPTION,
    PAGE_CASE_TITLE,
    PAGE_HEADER_CAPTION,
    PAGE_HEADER_TITLE,
)


def _write_cloud_insert_result(
    root: Path,
    relative_path: str,
    *,
    case_id: int = 600,
    db: str = "ZillizCloud",
    db_label: str = "zillz_cloud_cap_12cu_cloud_insert_laion100m_bs10k",
    custom_case: dict | None = None,
    metrics: dict | None = None,
    load_concurrency: int = 0,
) -> Path:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "run-1",
        "task_label": "cloud-insert-fixture",
        "results": [
            {
                "label": ":)",
                "task_config": {
                    "db": db,
                    "db_config": {
                        "db_label": db_label,
                        "version": "fixture",
                    },
                    "case_config": {
                        "case_id": case_id,
                        "custom_case": custom_case
                        or {
                            "batch_size": 10000,
                            "duration": None,
                            "dataset_with_size_type": "LAION 100M",
                        },
                    },
                    "load_concurrency": load_concurrency,
                },
                "metrics": metrics
                or {
                    "inserted_count": 100_000_000,
                    "insert_rows_per_second": 8788.4947,
                    "insert_completion_seconds": 11378.5129,
                    "searchable_after_insert_seconds": 0.0,
                    "indexed_after_searchable_seconds": 113.7851,
                    "additional_parameters": {"fully_searchable": True},
                },
            }
        ],
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_cloud_insert_loader_extracts_insert_metrics_and_path_dimensions(tmp_path):
    _write_cloud_insert_result(
        tmp_path,
        "zillz_cloud_cap_12cu/batch_10k/result_fixture.json",
    )

    rows = load_cloud_insert_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.product_key == "zillz_cloud_cap_12cu"
    assert row.product_name == "Zilliz Cloud Capacity 12CU"
    assert row.mode_key == "default"
    assert row.mode_display == "Default"
    assert row.batch_key == "batch_10k"
    assert row.batch_size == 10000
    assert row.dataset == "LAION 100M"
    assert row.duration is None
    assert row.load_concurrency == 0
    assert row.inserted_count == 100_000_000
    assert row.insert_rows_per_second == 8788.4947
    assert row.insert_completion_seconds == 11378.5129
    assert row.searchable_after_insert_seconds == 0.0
    assert row.indexed_after_searchable_seconds == 113.7851
    assert row.total_readiness_seconds == 11492.298


def test_cloud_insert_loader_preserves_turbopuffer_backpressure_mode(tmp_path):
    _write_cloud_insert_result(
        tmp_path,
        "turbopuffer/bp_off/batch_5k/result_fixture.json",
        db="TurboPuffer",
        db_label="turbopuffer_bp_off_cloud_insert_laion100m_bs5k",
        custom_case={
            "batch_size": 5000,
            "duration": None,
            "dataset_with_size_type": "LAION 100M",
        },
    )

    row = load_cloud_insert_rows(tmp_path)[0]

    assert row.product_key == "turbopuffer"
    assert row.product_name == "Turbopuffer (Backpressure Off)"
    assert row.mode_key == "bp_off"
    assert row.mode_display == "Backpressure Off"
    assert row.batch_key == "batch_5k"
    assert row.batch_size == 5000


def test_cloud_insert_loader_displays_turbopuffer_backpressure_modes_as_products(tmp_path):
    _write_cloud_insert_result(
        tmp_path,
        "turbopuffer/bp_off/batch_5k/result_bp_off.json",
        db="TurboPuffer",
        db_label="turbopuffer_bp_off_cloud_insert_laion100m_bs5k",
        custom_case={
            "batch_size": 5000,
            "duration": None,
            "dataset_with_size_type": "LAION 100M",
        },
    )
    _write_cloud_insert_result(
        tmp_path,
        "turbopuffer/bp_on/batch_5k/result_bp_on.json",
        db="TurboPuffer",
        db_label="turbopuffer_bp_on_cloud_insert_laion100m_bs5k",
        custom_case={
            "batch_size": 5000,
            "duration": None,
            "dataset_with_size_type": "LAION 100M",
        },
    )

    records = cloud_insert_records(load_cloud_insert_rows(tmp_path))

    assert [record["Product"] for record in records] == [
        "Turbopuffer (Backpressure Off)",
        "Turbopuffer (Backpressure On)",
    ]


def test_cloud_insert_records_are_table_ready(tmp_path):
    _write_cloud_insert_result(
        tmp_path,
        "pinecone_serverless/batch_1k/result_fixture.json",
        db="Pinecone",
        db_label="pinecone_cloud_insert_laion100m_bs1k",
        custom_case={
            "batch_size": 1000,
            "duration": None,
            "dataset_with_size_type": "LAION 100M",
        },
        metrics={
            "inserted_count": 100_000_000,
            "insert_rows_per_second": 248.6872,
            "insert_completion_seconds": 402124.2213,
            "searchable_after_insert_seconds": 0.0,
            "indexed_after_searchable_seconds": 0.0422,
            "additional_parameters": {},
        },
    )

    records = cloud_insert_records(load_cloud_insert_rows(tmp_path))

    assert records == [
        {
            "Product": "Pinecone Serverless",
            "Mode": "Default",
            "Dataset": "LAION 100M",
            "Batch Size": 1000,
            "Load Concurrency": 0,
            "Inserted Count": 100_000_000,
            "Insert Rows/s": 248.6872,
            "Insert Completion (s)": 402124.2213,
            "Searchable Delay (s)": 0.0,
            "Indexed Delay (s)": 0.0422,
            "Total Readiness (s)": 402124.2635,
        }
    ]


def test_cloud_insert_loader_rejects_wrong_case_id(tmp_path):
    _write_cloud_insert_result(
        tmp_path,
        "pinecone_serverless/batch_1k/result_fixture.json",
        case_id=500,
    )

    with pytest.raises(CloudInsertParseError, match="case_id"):
        load_cloud_insert_rows(tmp_path)


def test_legacy_result_collector_ignores_cloud_insert_raw_files(tmp_path):
    cloud_file = (
        tmp_path
        / "cloudleaderboard"
        / "cloud_insert"
        / "raw_results"
        / "pinecone_serverless"
        / "batch_1k"
        / "result_invalid.json"
    )
    cloud_file.parent.mkdir(parents=True)
    cloud_file.write_text("{not valid json", encoding="utf-8")

    assert ResultCollector.collect(tmp_path) == []


def test_top_nav_links_to_cloud_insert_page():
    class FakeStreamlit:
        html = ""

        def markdown(self, html, unsafe_allow_html=False):
            self.html = html

    fake = FakeStreamlit()

    NavToPages(fake)

    assert "/cloud_insert" in fake.html
    assert "Cloud Insert" in fake.html


def test_cloud_insert_sidebar_copy_names_the_case():
    assert PAGE_CASE_TITLE == "Cloud Insert"
    assert PAGE_CASE_CAPTION == "Insert readiness case"
    assert PAGE_HEADER_TITLE == "Cloud Insert"
    assert PAGE_HEADER_CAPTION == "Hosted vector database insert readiness results."
