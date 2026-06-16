import json
from pathlib import Path

import pytest

from vectordb_bench.backend.result_collector import ResultCollector
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.cloudleaderboard.cold_latency import (
    CloudColdLatencyParseError,
    cloud_cold_latency_records,
    load_cloud_cold_latency_rows,
)
from vectordb_bench.frontend.pages.cloud_cold_latency import (
    PAGE_CASE_CAPTION,
    PAGE_CASE_TITLE,
    PAGE_HEADER_CAPTION,
    PAGE_HEADER_TITLE,
    chart_records_by_mode,
    chart_records_for_selection,
)


def _write_cloud_cold_latency_result(
    root: Path,
    relative_path: str,
    *,
    case_id: int = 700,
    db: str = "Pinecone",
    db_label: str = "pinecone_cloud_cold_latency_laion100m_int_filter_0_9",
    custom_case: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "run-1",
        "task_label": "cloud-cold-latency-fixture",
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
                            "payload_profile": "ids_only",
                            "query_count": 1000,
                            "filter_rate": 0.9,
                        },
                    },
                },
                "metrics": metrics
                or {
                    "payload_profile": "ids_only",
                    "payload_estimated_bytes_per_query": 2000,
                    "load_duration": 11.1,
                    "insert_duration": 22.2,
                    "optimize_duration": 33.3,
                    "cold_latency": {
                        "cold_stats": {
                            "first_query_latency": 0.9106,
                            "p99_latency": 1.1424,
                            "p95_latency": 0.6552,
                            "avg_latency": 0.237,
                        },
                        "warm_stats": {
                            "first_query_latency": 0.115,
                            "p99_latency": 1.1493,
                            "p95_latency": 0.6602,
                            "avg_latency": 0.2473,
                        },
                        "cold_warm_ratio": {
                            "first_query_latency_ratio": 7.9183,
                            "p99_latency_ratio": 0.994,
                            "p95_latency_ratio": 0.9924,
                            "avg_latency_ratio": 0.9584,
                        },
                    },
                },
            }
        ],
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_cloud_cold_latency_loader_flattens_nested_latency_metrics(tmp_path):
    _write_cloud_cold_latency_result(
        tmp_path,
        "pinecone_serverless/int_filter_0.9/result_fixture.json",
    )

    rows = load_cloud_cold_latency_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.product_key == "pinecone_serverless"
    assert row.product_name == "Pinecone Serverless"
    assert row.mode_key == "int_filter_0.9"
    assert row.mode_display == "Int Filter 0.9"
    assert row.filter_rate == 0.9
    assert row.status == "Accepted"
    assert row.payload_profile == "ids_only"
    assert row.query_count == 1000
    assert row.payload_bytes_per_query == 2000
    assert row.cold_first_query_seconds == 0.9106
    assert row.cold_p99_seconds == 1.1424
    assert row.cold_p95_seconds == 0.6552
    assert row.cold_avg_seconds == 0.237
    assert row.warm_first_query_seconds == 0.115
    assert row.warm_p99_seconds == 1.1493
    assert row.warm_p95_seconds == 0.6602
    assert row.warm_avg_seconds == 0.2473
    assert row.first_query_ratio == 7.9183
    assert row.p99_ratio == 0.994
    assert row.p95_ratio == 0.9924
    assert row.avg_ratio == 0.9584
    assert row.load_duration == 11.1
    assert row.insert_duration == 22.2
    assert row.optimize_duration == 33.3


def test_cloud_cold_latency_loader_marks_tiered_rows_as_rebench_needed(tmp_path):
    _write_cloud_cold_latency_result(
        tmp_path,
        "zilliz_cloud_tiered_4cu/unfiltered/result_fixture.json",
        db="ZillizCloud",
        db_label="zilliz_cloud_tiered_4cu_cloud_cold_latency_laion100m_unfiltered",
        custom_case={
            "payload_profile": "ids_only",
            "query_count": 1000,
        },
    )

    row = load_cloud_cold_latency_rows(tmp_path)[0]

    assert row.product_name == "Zilliz Cloud Tiered 4CU"
    assert row.mode_display == "Unfiltered"
    assert row.filter_rate is None
    assert row.status == "Rebench needed"


def test_cloud_cold_latency_loader_computes_missing_ratios_safely(tmp_path):
    _write_cloud_cold_latency_result(
        tmp_path,
        "turbopuffer_pinned/unfiltered/result_fixture.json",
        db="TurboPuffer",
        db_label="turbopuffer_pinned_2rep_cloud_cold_latency_laion100m_unfiltered",
        custom_case={
            "payload_profile": "ids_only",
            "query_count": 1000,
        },
        metrics={
            "payload_profile": "ids_only",
            "payload_estimated_bytes_per_query": 2000,
            "load_duration": 1.0,
            "insert_duration": 2.0,
            "optimize_duration": 3.0,
            "cold_latency": {
                "cold_stats": {
                    "first_query_latency": 2.0,
                    "p99_latency": 4.0,
                    "p95_latency": 3.0,
                    "avg_latency": 1.5,
                },
                "warm_stats": {
                    "first_query_latency": 1.0,
                    "p99_latency": 2.0,
                    "p95_latency": 0.0,
                    "avg_latency": 1.0,
                },
            },
        },
    )

    row = load_cloud_cold_latency_rows(tmp_path)[0]

    assert row.product_name == "Turbopuffer Pinned 2 Replicas"
    assert row.first_query_ratio == 2.0
    assert row.p99_ratio == 2.0
    assert row.p95_ratio is None
    assert row.avg_ratio == 1.5


def test_cloud_cold_latency_records_are_table_ready(tmp_path):
    _write_cloud_cold_latency_result(
        tmp_path,
        "pinecone_serverless/int_filter_0.9/result_fixture.json",
    )

    records = cloud_cold_latency_records(load_cloud_cold_latency_rows(tmp_path))

    assert records == [
        {
            "Product": "Pinecone Serverless",
            "Mode": "Int Filter 0.9",
            "Status": "Accepted",
            "Payload": "IDs Only",
            "Query Count": 1000,
            "Payload Bytes/Query": 2000,
            "First Cold Query (s)": 0.9106,
            "Cold P99 (s)": 1.1424,
            "Cold P95 (s)": 0.6552,
            "Cold Avg (s)": 0.237,
            "Warm First Query (s)": 0.115,
            "Warm P99 (s)": 1.1493,
            "Warm P95 (s)": 0.6602,
            "Warm Avg (s)": 0.2473,
            "First Query Ratio": 7.9183,
            "P99 Ratio": 0.994,
            "P95 Ratio": 0.9924,
            "Avg Ratio": 0.9584,
            "Load Duration (s)": 11.1,
            "Insert Duration (s)": 22.2,
            "Optimize Duration (s)": 33.3,
        }
    ]


def test_cloud_cold_latency_loader_rejects_wrong_case_id(tmp_path):
    _write_cloud_cold_latency_result(
        tmp_path,
        "pinecone_serverless/unfiltered/result_fixture.json",
        case_id=500,
    )

    with pytest.raises(CloudColdLatencyParseError, match="case_id"):
        load_cloud_cold_latency_rows(tmp_path)


def test_legacy_result_collector_ignores_cloud_cold_latency_raw_files(tmp_path):
    cloud_file = (
        tmp_path
        / "cloudleaderboard"
        / "cloud_cold_latency"
        / "raw_results"
        / "pinecone_serverless"
        / "unfiltered"
        / "result_invalid.json"
    )
    cloud_file.parent.mkdir(parents=True)
    cloud_file.write_text("{not valid json", encoding="utf-8")

    assert ResultCollector.collect(tmp_path) == []


def test_top_nav_links_to_cloud_cold_latency_page():
    class FakeStreamlit:
        html = ""

        def markdown(self, html, unsafe_allow_html=False):
            self.html = html

    fake = FakeStreamlit()

    NavToPages(fake)

    assert "/cloud_cold_latency" in fake.html
    assert "Cloud Cold Latency" in fake.html


def test_cloud_cold_latency_sidebar_copy_names_the_case():
    assert PAGE_CASE_TITLE == "Cloud Cold Latency"
    assert PAGE_CASE_CAPTION == "Cold/warm latency case"
    assert PAGE_HEADER_TITLE == "Cloud Cold Latency"
    assert PAGE_HEADER_CAPTION == "Hosted vector database cold/warm query latency results."


def test_cloud_cold_latency_chart_records_group_by_mode():
    records = [
        {"Mode": "Unfiltered", "Product": "Pinecone Serverless"},
        {"Mode": "Int Filter 0.9", "Product": "Pinecone Serverless"},
        {"Mode": "Unfiltered", "Product": "Turbopuffer"},
    ]

    grouped = chart_records_by_mode(records)

    assert list(grouped.keys()) == ["Unfiltered", "Int Filter 0.9"]
    assert grouped["Unfiltered"] == [
        {"Mode": "Unfiltered", "Product": "Pinecone Serverless"},
        {"Mode": "Unfiltered", "Product": "Turbopuffer"},
    ]


def test_cloud_cold_latency_chart_records_for_selection_show_one_payload():
    records = [
        {"Mode": "Unfiltered", "Payload": "IDs Only", "Product": "Pinecone Serverless"},
        {"Mode": "Unfiltered", "Payload": "Vector", "Product": "Pinecone Serverless"},
        {"Mode": "Int Filter 0.9", "Payload": "IDs Only", "Product": "Pinecone Serverless"},
    ]

    selected = chart_records_for_selection(
        records,
        mode="Unfiltered",
        payload="IDs Only",
    )

    assert selected == [
        {"Mode": "Unfiltered", "Payload": "IDs Only", "Product": "Pinecone Serverless"},
    ]
