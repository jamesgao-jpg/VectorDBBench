import json
from pathlib import Path

import pytest

from vectordb_bench.backend.result_collector import ResultCollector
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.pages.cloudleaderboard import (
    PAGE_CASE_CAPTION,
    PAGE_HEADER_CAPTION,
    PAGE_HEADER_TITLE,
    PAGE_CASE_TITLE,
    chart_records_by_search_mode,
    chart_records_for_selection,
)
from vectordb_bench.frontend.components.cloudleaderboard.payload_search import (
    PayloadSearchParseError,
    aggregate_payload_search_rows,
    payload_search_records,
    load_payload_search_rows,
)


def _write_payload_result(
    root: Path,
    relative_path: str,
    *,
    case_id: int = 500,
    custom_case: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "run-1",
        "task_label": "payload-search-fixture",
        "results": [
            {
                "label": ":)",
                "task_config": {
                    "db": "ZillizCloud",
                    "db_config": {
                        "db_label": "zilliz_cloud_tiered_4cu_int_filter_1p_vector",
                        "version": "fixture",
                    },
                    "case_config": {
                        "case_id": case_id,
                        "custom_case": custom_case
                        or {
                            "payload_profile": "vector",
                            "filter_rate": 0.99,
                        },
                    },
                },
                "metrics": metrics
                or {
                    "qps": 123.4,
                    "recall": 0.95,
                    "ndcg": 0.96,
                    "serial_latency_p95": 0.11,
                    "serial_latency_p99": 0.22,
                    "conc_num_list": [60, 80],
                    "conc_qps_list": [111.1, 123.4],
                    "conc_latency_avg_list": [0.40, 0.50],
                    "conc_latency_p95_list": [0.60, 0.70],
                    "conc_latency_p99_list": [0.80, 0.90],
                    "payload_estimated_bytes_per_query": 309200,
                },
            }
        ],
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_payload_search_loader_derives_matrix_dimensions_from_path(tmp_path):
    _write_payload_result(
        tmp_path,
        "zilliz_cloud_tiered_4cu/int_filter/1p/vector/concurrent_qps/result_fixture.json",
    )

    rows = load_payload_search_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.product_key == "zilliz_cloud_tiered_4cu"
    assert row.product_name == "Zilliz Cloud Tiered 4CU"
    assert row.search_mode == "int_filter"
    assert row.filter_key == "1p"
    assert row.filter_display == "99%"
    assert row.filter_value == 0.99
    assert row.payload_profile == "vector"
    assert row.phase == "concurrent_qps"
    assert row.best_concurrency == 80
    assert row.best_latency_p95 == 0.70
    assert row.best_latency_p99 == 0.90
    assert row.payload_bytes_per_query == 309200


def test_payload_search_aggregation_combines_serial_and_concurrent_artifacts(tmp_path):
    _write_payload_result(
        tmp_path,
        "zilliz_cloud_tiered_4cu/int_filter/1p/vector/serial_recall/result_serial.json",
        metrics={
            "qps": 0,
            "recall": 0.951,
            "ndcg": 0.961,
            "serial_latency_p95": 0.12,
            "serial_latency_p99": 0.34,
            "payload_estimated_bytes_per_query": 309200,
        },
    )
    _write_payload_result(
        tmp_path,
        "zilliz_cloud_tiered_4cu/int_filter/1p/vector/concurrent_qps/result_concurrent.json",
        metrics={
            "qps": 200.0,
            "recall": 0,
            "ndcg": 0,
            "conc_num_list": [60, 80],
            "conc_qps_list": [150.0, 200.0],
            "conc_latency_avg_list": [0.10, 0.20],
            "conc_latency_p95_list": [0.30, 0.40],
            "conc_latency_p99_list": [0.50, 0.60],
            "payload_estimated_bytes_per_query": 309200,
        },
    )

    aggregated = aggregate_payload_search_rows(load_payload_search_rows(tmp_path))

    assert len(aggregated) == 1
    row = aggregated[0]
    assert row.recall == 0.951
    assert row.ndcg == 0.961
    assert row.serial_latency_p95 == 0.12
    assert row.serial_latency_p99 == 0.34
    assert row.qps == 200.0
    assert row.best_concurrency == 80
    assert row.has_serial_recall is True
    assert row.has_concurrent_qps is True


def test_payload_search_records_are_table_ready(tmp_path):
    _write_payload_result(
        tmp_path,
        "zilliz_cloud_tiered_4cu/int_filter/1p/vector/concurrent_qps/result_fixture.json",
    )
    rows = aggregate_payload_search_rows(load_payload_search_rows(tmp_path))

    records = payload_search_records(rows)

    assert records == [
        {
            "Product": "Zilliz Cloud Tiered 4CU",
            "Search Mode": "Integer Filter",
            "Filter": "99%",
            "Payload": "Vector",
            "Max QPS": 123.4,
            "Best Concurrency": 80,
            "Recall": 0.95,
            "NDCG": 0.96,
            "P95 Latency (s)": 0.70,
            "P99 Latency (s)": 0.90,
            "Payload Bytes/Query": 309200,
            "Serial Recall": "missing",
            "Concurrent QPS": "available",
        }
    ]


def test_payload_search_loader_rejects_wrong_case_id(tmp_path):
    _write_payload_result(
        tmp_path,
        "zilliz_cloud_tiered_4cu/int_filter/1p/vector/concurrent_qps/result_fixture.json",
        case_id=800,
    )

    with pytest.raises(PayloadSearchParseError, match="case_id"):
        load_payload_search_rows(tmp_path)


def test_legacy_result_collector_ignores_cloudleaderboard_raw_files(tmp_path):
    cloud_file = (
        tmp_path
        / "cloudleaderboard"
        / "cloud_payload_search"
        / "raw_results"
        / "zilliz_cloud_tiered_4cu"
        / "int_filter"
        / "1p"
        / "vector"
        / "concurrent_qps"
        / "result_invalid.json"
    )
    cloud_file.parent.mkdir(parents=True)
    cloud_file.write_text("{not valid json", encoding="utf-8")

    assert ResultCollector.collect(tmp_path) == []


def test_top_nav_links_to_cloud_payload_search_page():
    class FakeStreamlit:
        html = ""

        def markdown(self, html, unsafe_allow_html=False):
            self.html = html

    fake = FakeStreamlit()

    NavToPages(fake)

    assert "/cloudleaderboard" in fake.html
    assert "Cloud Payload Search" in fake.html
    assert "Cloud Leaderboard" not in fake.html


def test_chart_records_are_split_by_search_mode_to_avoid_plotly_facet_labels():
    records = [
        {"Search Mode": "Integer Filter", "Product": "A"},
        {"Search Mode": "Scalar Label Filter", "Product": "B"},
        {"Search Mode": "Integer Filter", "Product": "C"},
    ]

    grouped = chart_records_by_search_mode(records)

    assert list(grouped.keys()) == ["Integer Filter", "Scalar Label Filter"]
    assert grouped["Integer Filter"] == [
        {"Search Mode": "Integer Filter", "Product": "A"},
        {"Search Mode": "Integer Filter", "Product": "C"},
    ]


def test_chart_records_for_selection_show_one_filter_rate_and_payload():
    records = [
        {"Search Mode": "Integer Filter", "Filter": "50%", "Payload": "IDs Only", "Product": "A"},
        {"Search Mode": "Integer Filter", "Filter": "50%", "Payload": "Vector", "Product": "A"},
        {"Search Mode": "Integer Filter", "Filter": "90%", "Payload": "IDs Only", "Product": "A"},
        {"Search Mode": "Scalar Label Filter", "Filter": "50%", "Payload": "IDs Only", "Product": "B"},
    ]

    selected = chart_records_for_selection(
        records,
        search_mode="Integer Filter",
        filter_display="50%",
        payload="IDs Only",
    )

    assert selected == [
        {"Search Mode": "Integer Filter", "Filter": "50%", "Payload": "IDs Only", "Product": "A"}
    ]


def test_cloud_payload_search_sidebar_copy_names_the_case_not_whole_leaderboard():
    assert PAGE_CASE_TITLE == "Cloud Payload Search"
    assert PAGE_CASE_CAPTION == "Payload search case"
    assert PAGE_HEADER_TITLE == "Cloud Payload Search"
    assert PAGE_HEADER_CAPTION == "Hosted vector database payload search results."
