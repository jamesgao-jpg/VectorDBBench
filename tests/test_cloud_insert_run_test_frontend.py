from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import (
    UI_CASE_CLUSTERS,
    generate_cloud_insert_cases,
)


def test_cloud_insert_cases_cover_laion_batch_sizes():
    cases = generate_cloud_insert_cases()

    assert len(cases) == 3
    assert {case.case_id for case in cases} == {CaseType.CloudInsertCase}
    assert {case.custom_case["batch_size"] for case in cases} == {1000, 5000, 10000}
    assert {case.custom_case["duration"] for case in cases} == {None}
    assert {case.custom_case["dataset_with_size_type"] for case in cases} == {
        DatasetWithSizeType.LAIONLarge.value
    }


def test_run_test_page_registers_cloud_insert_cluster():
    cluster = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Insert")

    assert [item.label for item in cluster.uiCaseItems] == [
        "Cloud Insert - LAION 100M - Batch 1k",
        "Cloud Insert - LAION 100M - Batch 5k",
        "Cloud Insert - LAION 100M - Batch 10k",
    ]
    assert all(
        case.case_id == CaseType.CloudInsertCase
        for item in cluster.uiCaseItems
        for case in item.cases
    )


def test_cloud_insert_selection_generates_task_configs():
    cluster = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "Cloud Insert")
    selected_cases = cluster.uiCaseItems[2].get_cases()

    tasks = generate_tasks(
        [DB.Test],
        {DB.Test: DB.Test.config_cls()},
        selected_cases,
        {DB.Test: {case: {} for case in selected_cases}},
    )

    assert len(tasks) == 1
    assert {task.case_config.case_id for task in tasks} == {CaseType.CloudInsertCase}
    assert {task.case_config.custom_case["batch_size"] for task in tasks} == {10000}
    assert {task.case_config.custom_case["dataset_with_size_type"] for task in tasks} == {
        DatasetWithSizeType.LAIONLarge.value
    }
