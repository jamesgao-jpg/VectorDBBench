import pytest

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.models import ProgressStage, ProgressStatus, TaskStage


def test_fts_pre_run_reports_setup_before_download_without_reordering_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = []
    filter_obj = object()

    class Dataset:
        def prepare(self, source: DatasetSource, filters: object | None = None) -> None:
            operations.append(("prepare", source, filters))

    class Case:
        label = CaseLabel.FullTextSearchPerformance
        is_multitenant = False
        dataset = Dataset()
        filters = filter_obj

    config = type("Config", (), {"stages": [TaskStage.LOAD]})()
    runner = CaseRunner.construct(
        run_id="fts-stage-order",
        ca=Case(),
        config=config,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    updates = []
    runner.set_progress_callback(updates.append)

    def record_init_db(_runner: CaseRunner, drop_old: bool = True) -> None:
        operations.append(("init_db", drop_old))

    monkeypatch.setattr(CaseRunner, "init_db", record_init_db)

    runner._pre_run(drop_old=False)

    assert operations == [
        ("prepare", DatasetSource.S3, filter_obj),
        ("init_db", False),
    ]
    assert [(update.stage, update.status) for update in updates] == [
        (ProgressStage.SETUP, ProgressStatus.RUNNING),
        (ProgressStage.SETUP, ProgressStatus.COMPLETED),
        (ProgressStage.DOWNLOAD, ProgressStatus.RUNNING),
        (ProgressStage.DOWNLOAD, ProgressStatus.COMPLETED),
    ]


@pytest.mark.parametrize(
    ("stages", "expected_stage"),
    [
        ([TaskStage.SEARCH_SERIAL], ProgressStage.SEARCH_SERIAL),
        ([TaskStage.SEARCH_CONCURRENT], ProgressStage.SEARCH_CONCURRENT),
        (
            [TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
            ProgressStage.SEARCH_CONCURRENT,
        ),
    ],
)
def test_search_runner_initialization_failure_uses_next_search_stage(
    monkeypatch: pytest.MonkeyPatch,
    stages: list[TaskStage],
    expected_stage: ProgressStage,
) -> None:
    class Case:
        label = CaseLabel.Performance

    config = type("Config", (), {"stages": stages})()
    runner = CaseRunner.construct(
        run_id="search-init-failure",
        ca=Case(),
        config=config,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )
    updates = []
    runner.set_progress_callback(updates.append)

    def skip_pre_run(_runner: CaseRunner, _drop_old: bool = True) -> None:
        return None

    def fail_search_runner_initialization(_runner: CaseRunner) -> None:
        raise RuntimeError("search runner initialization failed")

    monkeypatch.setattr(CaseRunner, "_pre_run", skip_pre_run)
    monkeypatch.setattr(CaseRunner, "_init_search_runners", fail_search_runner_initialization)

    with pytest.raises(RuntimeError, match="search runner initialization failed"):
        runner.run(drop_old=False)

    assert updates[-2].stage == expected_stage
    assert updates[-2].status == ProgressStatus.RUNNING
    assert updates[-1].stage == expected_stage
    assert updates[-1].status == ProgressStatus.FAILED
