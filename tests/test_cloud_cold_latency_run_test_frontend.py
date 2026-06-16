from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UI_CASE_CLUSTERS,
    generate_cloud_cold_latency_cases,
)


def test_cloud_cold_latency_cases_cover_current_modes():
    cases = generate_cloud_cold_latency_cases()

    assert len(cases) == 2
    assert {case.case_id for case in cases} == {CaseType.CloudColdLatencyCase}
    assert {case.custom_case["payload_profile"] for case in cases} == {"ids_only"}
    assert {case.custom_case["query_count"] for case in cases} == {1000}
    assert sum("filter_rate" not in case.custom_case for case in cases) == 1
    assert sum(case.custom_case.get("filter_rate") == 0.9 for case in cases) == 1
    assert all("label_percentage" not in case.custom_case for case in cases)
    assert all("dataset_with_size_type" not in case.custom_case for case in cases)


def test_run_test_page_registers_cloud_cold_latency_cluster():
    cluster = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Cold Latency")

    assert [item.label for item in cluster.uiCaseItems] == [
        "Cloud Cold Latency - LAION 100M - Unfiltered",
        "Cloud Cold Latency - LAION 100M - Int Filter 0.9",
    ]
    assert all(
        case.case_id == CaseType.CloudColdLatencyCase
        for item in cluster.uiCaseItems
        for case in item.cases
    )


def test_cloud_cold_latency_unfiltered_selection_generates_task_configs():
    cluster = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Cold Latency")
    selected_cases = cluster.uiCaseItems[0].get_cases()

    tasks = generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        selected_cases,
        {DB.Test: {case: {} for case in selected_cases}},
    )

    assert len(tasks) == 1
    assert {task.case_config.case_id for task in tasks} == {CaseType.CloudColdLatencyCase}
    assert {task.case_config.custom_case["payload_profile"] for task in tasks} == {"ids_only"}
    assert {task.case_config.custom_case["query_count"] for task in tasks} == {1000}
    assert all("filter_rate" not in task.case_config.custom_case for task in tasks)


def test_cloud_cold_latency_int_filter_selection_generates_filter_task():
    cluster = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Cold Latency")
    selected_cases = cluster.uiCaseItems[1].get_cases()

    tasks = generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        selected_cases,
        {DB.Test: {case: {} for case in selected_cases}},
    )

    assert len(tasks) == 1
    assert {task.case_config.case_id for task in tasks} == {CaseType.CloudColdLatencyCase}
    assert {task.case_config.custom_case["payload_profile"] for task in tasks} == {"ids_only"}
    assert {task.case_config.custom_case["query_count"] for task in tasks} == {1000}
    assert {task.case_config.custom_case["filter_rate"] for task in tasks} == {0.9}
