from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UI_CASE_CLUSTERS,
    generate_cloud_payload_search_cases,
)


def test_cloud_payload_search_cases_cover_payloads_and_filter_modes():
    cases = generate_cloud_payload_search_cases()

    assert len(cases) == 42
    assert {case.case_id for case in cases} == {CaseType.CloudPayloadSearchCase}
    assert {case.custom_case["payload_profile"] for case in cases} == {"ids_only", "scalar_label", "vector"}
    assert sum("filter_rate" not in case.custom_case and "label_percentage" not in case.custom_case for case in cases) == 3
    assert sum("filter_rate" in case.custom_case for case in cases) == 12
    assert sum("label_percentage" in case.custom_case for case in cases) == 27
    assert all(
        not ("filter_rate" in case.custom_case and "label_percentage" in case.custom_case)
        for case in cases
    )


def test_run_test_page_registers_cloud_payload_search_cluster():
    cluster = next(
        cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Payload Search"
    )

    assert [item.label for item in cluster.uiCaseItems] == [
        "Cloud Payload Search - Unfiltered - IDs Only",
        "Cloud Payload Search - Unfiltered - Scalar Label",
        "Cloud Payload Search - Unfiltered - Vector",
        "Cloud Payload Search - Integer Filter - IDs Only",
        "Cloud Payload Search - Integer Filter - Scalar Label",
        "Cloud Payload Search - Integer Filter - Vector",
        "Cloud Payload Search - Scalar Label Filter - IDs Only",
        "Cloud Payload Search - Scalar Label Filter - Scalar Label",
        "Cloud Payload Search - Scalar Label Filter - Vector",
    ]
    assert all(
        case.case_id == CaseType.CloudPayloadSearchCase
        for item in cluster.uiCaseItems
        for case in item.cases
    )


def test_cloud_payload_search_selection_generates_task_configs():
    cluster = next(
        cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Payload Search"
    )
    selected_cases = cluster.uiCaseItems[0].get_cases()

    tasks = generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        selected_cases,
        {DB.Test: {case: {} for case in selected_cases}},
    )

    assert len(tasks) == 1
    assert {task.case_config.case_id for task in tasks} == {CaseType.CloudPayloadSearchCase}
    assert {task.case_config.custom_case["payload_profile"] for task in tasks} == {"ids_only"}


def test_cloud_payload_integer_filter_payload_selection_generates_filter_tasks():
    cluster = next(
        cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Payload Search"
    )
    vector_integer_filter = next(
        item for item in cluster.uiCaseItems if item.label == "Cloud Payload Search - Integer Filter - Vector"
    )
    selected_cases = vector_integer_filter.get_cases()

    tasks = generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        selected_cases,
        {DB.Test: {case: {} for case in selected_cases}},
    )

    assert len(tasks) == 4
    assert {task.case_config.custom_case["payload_profile"] for task in tasks} == {"vector"}
    assert {task.case_config.custom_case["filter_rate"] for task in tasks} == {0.999, 0.99, 0.9, 0.5}
