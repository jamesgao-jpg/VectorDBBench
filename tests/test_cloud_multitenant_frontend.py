import json
from pathlib import Path

import pytest

from vectordb_bench.backend.result_collector import ResultCollector
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.cloudleaderboard.multi_tenant_search import (
    MultiTenantSearchParseError,
    load_multi_tenant_search_rows,
    multi_tenant_search_records,
)
from vectordb_bench.frontend.pages.cloud_multi_tenant_search import (
    PAGE_CASE_CAPTION,
    PAGE_CASE_TITLE,
    PAGE_HEADER_CAPTION,
    PAGE_HEADER_TITLE,
    chart_records_by_concurrency_signature,
    chart_records_for_selection,
)


def _write_multi_tenant_result(
    root: Path,
    relative_path: str,
    *,
    case_id: int = 800,
    custom_case: dict | None = None,
    metrics: dict | None = None,
    k: int = 50,
) -> Path:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "run-1",
        "task_label": "multi-tenant-fixture",
        "results": [
            {
                "label": ":)",
                "task_config": {
                    "db": "ZillizCloud",
                    "db_config": {
                        "db_label": "zilliz_cloud_capacity_2cu",
                        "version": "fixture",
                    },
                    "case_config": {
                        "case_id": case_id,
                        "k": k,
                        "custom_case": custom_case
                        or {
                            "dataset_with_size_type": "Large Cohere (768dim, 10M)",
                            "tenant_count": 1000,
                            "tenant_prefix": "tenant_",
                            "tenant_id_width": 4,
                            "payload_profile": "ids_only",
                            "filter_rate": 0.5,
                        },
                    },
                },
                "metrics": metrics
                or {
                    "qps": 551.2294,
                    "conc_num_list": [4],
                    "conc_qps_list": [551.2294],
                    "conc_latency_avg_list": [0.0073],
                    "conc_latency_p95_list": [0.0110],
                    "conc_latency_p99_list": [0.0325],
                    "payload_estimated_bytes_per_query": 1000,
                },
            }
        ],
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_multi_tenant_loader_derives_dimensions_and_tenant_metadata(tmp_path):
    _write_multi_tenant_result(
        tmp_path,
        "pinecone_serverless/int_filter/50p/ids_only/concurrent_qps/result_fixture.json",
    )

    rows = load_multi_tenant_search_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.product_key == "pinecone_serverless"
    assert row.product_name == "Pinecone Serverless"
    assert row.search_mode == "int_filter"
    assert row.filter_key == "50p"
    assert row.filter_display == "50%"
    assert row.filter_value == 0.5
    assert row.payload_profile == "ids_only"
    assert row.phase == "concurrent_qps"
    assert row.dataset == "Large Cohere (768dim, 10M)"
    assert row.tenant_count == 1000
    assert row.tenant_id_format == "tenant_0000..tenant_0999"
    assert row.top_k == 50
    assert row.concurrency_signature == "c4"
    assert row.best_concurrency == 4
    assert row.best_latency_p95 == 0.0110
    assert row.best_latency_p99 == 0.0325
    assert row.payload_bytes_per_query == 1000


def test_multi_tenant_loader_computes_best_concurrency_from_qps_list(tmp_path):
    _write_multi_tenant_result(
        tmp_path,
        "zilliz_cloud_capacity_2cu/scalar_label_filter/1p/scalar_label/concurrent_qps/result_fixture.json",
        custom_case={
            "dataset_with_size_type": "Large Cohere (768dim, 10M)",
            "tenant_count": 1000,
            "tenant_prefix": "tenant_",
            "tenant_id_width": 4,
            "payload_profile": "scalar_label",
            "label_percentage": 0.01,
        },
        metrics={
            "qps": 222.0,
            "conc_num_list": [60, 80],
            "conc_qps_list": [222.0, 221.0],
            "conc_latency_avg_list": [0.10, 0.20],
            "conc_latency_p95_list": [0.30, 0.40],
            "conc_latency_p99_list": [0.50, 0.60],
            "payload_estimated_bytes_per_query": 1800,
        },
    )

    row = load_multi_tenant_search_rows(tmp_path)[0]

    assert row.product_name == "Zilliz Cloud Capacity 2CU"
    assert row.filter_display == "1%"
    assert row.concurrency_signature == "c60,c80"
    assert row.best_concurrency == 60
    assert row.best_latency_p95 == 0.30
    assert row.best_latency_p99 == 0.50


def test_multi_tenant_records_are_table_ready_and_show_concurrency(tmp_path):
    _write_multi_tenant_result(
        tmp_path,
        "zilliz_cloud_capacity_2cu/int_filter/50p/scalar_label/concurrent_qps/result_fixture.json",
        custom_case={
            "dataset_with_size_type": "Large Cohere (768dim, 10M)",
            "tenant_count": 1000,
            "tenant_prefix": "tenant_",
            "tenant_id_width": 4,
            "payload_profile": "scalar_label",
            "filter_rate": 0.5,
        },
        metrics={
            "qps": 892.0028,
            "conc_num_list": [60, 80],
            "conc_qps_list": [886.0709, 892.0028],
            "conc_latency_avg_list": [0.080, 0.090],
            "conc_latency_p95_list": [0.0928, 0.1095],
            "conc_latency_p99_list": [0.0960, 0.1134],
            "payload_estimated_bytes_per_query": 1800,
        },
    )

    records = multi_tenant_search_records(load_multi_tenant_search_rows(tmp_path))

    assert records == [
        {
            "Product": "Zilliz Cloud Capacity 2CU",
            "Dataset": "Large Cohere (768dim, 10M)",
            "Search Mode": "Integer Filter",
            "Filter": "50%",
            "Payload": "Scalar Label",
            "Tenant Count": 1000,
            "Tenant IDs": "tenant_0000..tenant_0999",
            "Top K": 50,
            "Concurrency": "c60,c80",
            "Max QPS": 892.0028,
            "Best Concurrency": 80,
            "P95 Latency (s)": 0.1095,
            "P99 Latency (s)": 0.1134,
            "Payload Bytes/Query": 1800,
        }
    ]


def test_multi_tenant_loader_rejects_wrong_case_id(tmp_path):
    _write_multi_tenant_result(
        tmp_path,
        "pinecone_serverless/int_filter/50p/ids_only/concurrent_qps/result_fixture.json",
        case_id=500,
    )

    with pytest.raises(MultiTenantSearchParseError, match="case_id"):
        load_multi_tenant_search_rows(tmp_path)


def test_legacy_result_collector_ignores_multi_tenant_raw_files(tmp_path):
    cloud_file = (
        tmp_path
        / "cloudleaderboard"
        / "cloud_multi_tenant_search"
        / "raw_results"
        / "pinecone_serverless"
        / "int_filter"
        / "50p"
        / "ids_only"
        / "concurrent_qps"
        / "result_invalid.json"
    )
    cloud_file.parent.mkdir(parents=True)
    cloud_file.write_text("{not valid json", encoding="utf-8")

    assert ResultCollector.collect(tmp_path) == []


def test_top_nav_links_to_cloud_multi_tenant_search_page():
    class FakeStreamlit:
        html = ""

        def markdown(self, html, unsafe_allow_html=False):
            self.html = html

    fake = FakeStreamlit()

    NavToPages(fake)

    assert "/cloud_multi_tenant_search" in fake.html
    assert "Cloud Multi-Tenant Search" in fake.html


def test_multi_tenant_chart_records_are_split_by_concurrency_signature():
    records = [
        {"Concurrency": "c4", "Product": "Pinecone Serverless"},
        {"Concurrency": "c60,c80", "Product": "Zilliz Cloud Capacity 2CU"},
        {"Concurrency": "c4", "Product": "Pinecone Serverless"},
    ]

    grouped = chart_records_by_concurrency_signature(records)

    assert list(grouped.keys()) == ["c4", "c60,c80"]
    assert grouped["c4"] == [
        {"Concurrency": "c4", "Product": "Pinecone Serverless"},
        {"Concurrency": "c4", "Product": "Pinecone Serverless"},
    ]


def test_multi_tenant_chart_records_for_selection_show_one_filter_rate_and_payload():
    records = [
        {"Concurrency": "c60,c80", "Search Mode": "Integer Filter", "Filter": "50%", "Payload": "IDs Only"},
        {"Concurrency": "c60,c80", "Search Mode": "Integer Filter", "Filter": "50%", "Payload": "Vector"},
        {"Concurrency": "c4", "Search Mode": "Integer Filter", "Filter": "50%", "Payload": "IDs Only"},
        {"Concurrency": "c60,c80", "Search Mode": "Scalar Label Filter", "Filter": "50%", "Payload": "IDs Only"},
    ]

    selected = chart_records_for_selection(
        records,
        concurrency_signature="c60,c80",
        search_mode="Integer Filter",
        filter_display="50%",
        payload="IDs Only",
    )

    assert selected == [
        {"Concurrency": "c60,c80", "Search Mode": "Integer Filter", "Filter": "50%", "Payload": "IDs Only"}
    ]


def test_cloud_multi_tenant_search_sidebar_copy_names_the_case_not_whole_leaderboard():
    assert PAGE_CASE_TITLE == "Cloud Multi-Tenant Search"
    assert PAGE_CASE_CAPTION == "Multi-tenant search case"
    assert PAGE_HEADER_TITLE == "Cloud Multi-Tenant Search"
    assert PAGE_HEADER_CAPTION == "Hosted vector database multi-tenant search results."
