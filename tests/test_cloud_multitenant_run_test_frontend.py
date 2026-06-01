from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UI_CASE_CLUSTERS,
    generate_cloud_multi_tenant_search_cases,
)


def test_cloud_multi_tenant_search_cases_cover_payloads_and_filter_modes():
    cases = generate_cloud_multi_tenant_search_cases()

    assert len(cases) == 42
    assert {case.case_id for case in cases} == {CaseType.CloudMultiTenantSearchCase}
    assert {case.k for case in cases} == {50}
    assert {case.custom_case["payload_profile"] for case in cases} == {"ids_only", "scalar_label", "vector"}
    assert {case.custom_case["dataset_with_size_type"] for case in cases} == {DatasetWithSizeType.CohereLarge.value}
    assert {case.custom_case["tenant_count"] for case in cases} == {1000}
    assert {case.custom_case["tenant_prefix"] for case in cases} == {"tenant_"}
    assert {case.custom_case["tenant_id_width"] for case in cases} == {4}
    assert sum("filter_rate" not in case.custom_case and "label_percentage" not in case.custom_case for case in cases) == 3
    assert sum("filter_rate" in case.custom_case for case in cases) == 12
    assert sum("label_percentage" in case.custom_case for case in cases) == 27
    assert all(
        not ("filter_rate" in case.custom_case and "label_percentage" in case.custom_case)
        for case in cases
    )


def test_run_test_page_registers_cloud_multi_tenant_search_cluster():
    cluster = next(
        cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Multi-Tenant Search"
    )

    assert [item.label for item in cluster.uiCaseItems] == [
        "Cloud Multi-Tenant Search - Unfiltered",
        "Cloud Multi-Tenant Search - Integer Filter",
        "Cloud Multi-Tenant Search - Scalar Label Filter",
    ]
    assert all(
        case.case_id == CaseType.CloudMultiTenantSearchCase
        for item in cluster.uiCaseItems
        for case in item.cases
    )


def test_cloud_multi_tenant_search_selection_generates_task_configs():
    cluster = next(
        cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Multi-Tenant Search"
    )
    selected_cases = cluster.uiCaseItems[0].get_cases()

    tasks = generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        selected_cases,
        {DB.Test: {case: {} for case in selected_cases}},
    )

    assert len(tasks) == 3
    assert {task.case_config.case_id for task in tasks} == {CaseType.CloudMultiTenantSearchCase}
    assert {task.case_config.k for task in tasks} == {50}
    assert {task.case_config.custom_case["payload_profile"] for task in tasks} == {
        "ids_only",
        "scalar_label",
        "vector",
    }
    assert {task.case_config.custom_case["tenant_count"] for task in tasks} == {1000}
