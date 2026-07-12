import logging
import pickle
import time
from types import SimpleNamespace

import pytest
import ujson

from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner import mp_runner as mp_runner_module
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.backend.workload import WorkloadKind
from vectordb_bench.interface import SIGNAL, BenchMarkRunner
from vectordb_bench.models import (
    CaseConfig,
    CaseType,
    DB,
    IndexType,
    ProgressStage,
    ProgressStatus,
    ProgressUpdate,
    TaskConfig,
    TaskStage,
)

log = logging.getLogger(__name__)


def _progress_update(**overrides) -> ProgressUpdate:
    values = {
        "run_id": "run-1",
        "case_index": 0,
        "case_total": 1,
        "stage": ProgressStage.DOWNLOAD,
        "stage_index": 1,
        "stage_total": 7,
        "status": ProgressStatus.RUNNING,
        "message": "Downloading dataset",
        "started_at": 100.0,
        "updated_at": 101.0,
    }
    values.update(overrides)
    return ProgressUpdate(**values)


class _SignalConnection:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    def poll(self):
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        self.closed = True


class _RunningTask:
    def __init__(self):
        self.finished = []

    def set_finished(self, index):
        self.finished.append(index)


class TestBenchRunner:
    def test_get_results(self):
        runner = BenchMarkRunner()

        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_whole(self):
        runner = BenchMarkRunner()

        task_config = TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        runner._sync_running_task()
        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_clean(self):
        runner = BenchMarkRunner()

        task_config = TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        time.sleep(3)
        runner.stop_running()

    def test_performance_case_no_error(self):
        task_config = TaskConfig(
            db=DB.ZillizCloud,
            db_config=DB.ZillizCloud.config(uri="xxx", user="abc", password="1234"),
            db_case_config=DB.ZillizCloud.case_config_cls()(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        t = task_config.copy()
        d = t.json(exclude={"db_config": {"password", "api_key"}})
        log.info(f"{d}")

        loads = ujson.loads(d)
        log.info(f"{loads}")


def test_progress_update_contract_serializes_frozen_fields():
    update = _progress_update(current=256, total=1024, unit="bytes", duration_hint_seconds=30)

    assert update.model_dump(mode="json") == {
        "run_id": "run-1",
        "case_index": 0,
        "case_total": 1,
        "stage": "download",
        "stage_index": 1,
        "stage_total": 7,
        "status": "running",
        "message": "Downloading dataset",
        "current": 256,
        "total": 1024,
        "unit": "bytes",
        "started_at": 100.0,
        "updated_at": 101.0,
        "duration_hint_seconds": 30.0,
    }


def test_benchmark_runner_reduces_progress_signal_to_latest_snapshot():
    runner = BenchMarkRunner()
    running_task = _RunningTask()
    update = _progress_update(current=512, total=1024)
    runner.running_task = running_task
    runner.receive_conn = _SignalConnection(
        [
            (SIGNAL.PROGRESS, update.model_dump()),
            (SIGNAL.WIP, 0),
        ]
    )

    assert runner.get_progress() == update
    assert running_task.finished == [0]


def test_stop_running_marks_progress_cancelled():
    runner = BenchMarkRunner()
    runner.latest_progress = _progress_update()

    runner.stop_running()

    progress = runner.get_progress()
    assert progress is not None
    assert progress.status == ProgressStatus.CANCELLED
    assert progress.message == "Benchmark cancelled"


def _case_runner(dataset, *, run_id="run-1") -> CaseRunner:
    concurrency_config = SimpleNamespace(num_concurrency=[1, 5], concurrency_duration=30)
    config = SimpleNamespace(
        stages=[TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_CONCURRENT, TaskStage.SEARCH_SERIAL],
        db=DB.Milvus,
        db_case_config=SimpleNamespace(),
        case_config=SimpleNamespace(concurrency_search_config=concurrency_config),
    )
    case = SimpleNamespace(
        label=CaseLabel.Performance,
        dataset=dataset,
        filters=non_filter,
        is_multitenant=False,
        with_scalar_labels=False,
    )
    return CaseRunner.model_construct(
        run_id=run_id,
        config=config,
        ca=case,
        status=RunningStatus.PENDING,
        dataset_source=DatasetSource.S3,
    )


def test_case_runner_forwards_dataset_byte_progress(monkeypatch):
    class Dataset:
        def prepare(self, *args, progress_callback=None, **kwargs):
            assert progress_callback is not None
            progress_callback(25, 100, "Downloading train.parquet")
            return True

    runner = _case_runner(Dataset())
    updates = []
    runner.set_progress_callback(updates.append, case_index=1, case_total=3)
    monkeypatch.setattr(CaseRunner, "init_db", lambda self, drop_old=True: None)

    runner._pre_run()

    assert [(update.stage, update.status) for update in updates] == [
        (ProgressStage.SETUP, ProgressStatus.RUNNING),
        (ProgressStage.SETUP, ProgressStatus.COMPLETED),
        (ProgressStage.DOWNLOAD, ProgressStatus.RUNNING),
        (ProgressStage.DOWNLOAD, ProgressStatus.RUNNING),
        (ProgressStage.DOWNLOAD, ProgressStatus.COMPLETED),
    ]
    byte_update = updates[-2]
    assert byte_update.case_index == 1
    assert byte_update.case_total == 3
    assert byte_update.current == 25
    assert byte_update.total == 100
    assert byte_update.unit == "bytes"


def test_case_runner_reports_failure_for_active_stage(monkeypatch):
    runner = _case_runner(SimpleNamespace())
    updates = []
    runner.set_progress_callback(updates.append)

    def fail_pre_run(self, drop_old=True):
        self._emit_progress(ProgressStage.SETUP, ProgressStatus.RUNNING, "Preparing benchmark target")
        raise RuntimeError("target unavailable")

    monkeypatch.setattr(CaseRunner, "_pre_run", fail_pre_run)

    with pytest.raises(RuntimeError, match="target unavailable"):
        runner.run()

    assert updates[-1].stage == ProgressStage.SETUP
    assert updates[-1].status == ProgressStatus.FAILED
    assert updates[-1].started_at == updates[0].started_at


def test_case_runner_drops_progress_callback_from_nested_process_state():
    runner = _case_runner(SimpleNamespace())
    runner.set_progress_callback(lambda update: None)

    state = runner.__getstate__()

    assert state["__pydantic_private__"]["_progress_callback"] is None
    pickle.dumps(runner)


def test_search_runner_reports_level_boundaries_and_drops_callback_from_worker_state(monkeypatch):
    class DB:
        name = "fake"

        def supports_payload_profile(self, payload_profile):
            return True

    class Future:
        def result(self):
            return 10, 1.0, [0.001, 0.002]

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, *args, **kwargs):
            return Future()

    class Condition:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def notify_all(self):
            pass

    class Manager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def Queue(self):
            return SimpleNamespace()

        def Condition(self):
            return Condition()

    updates = []
    runner = MultiProcessingSearchRunner(
        db=DB(),
        test_data=[[0.0]],
        concurrencies=[1, 2],
        duration=1,
        workload_kind=WorkloadKind.VECTOR,
        progress_callback=lambda current, total, concurrency, message: updates.append(
            (current, total, concurrency, message)
        ),
    )
    monkeypatch.setattr(mp_runner_module.mp, "Manager", Manager)
    monkeypatch.setattr(mp_runner_module.concurrent.futures, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(runner, "_wait_for_queue_fill", lambda queue, size: None)

    runner.run()

    assert [(current, total, concurrency) for current, total, concurrency, _ in updates] == [
        (None, None, 1),
        (1, 2, 1),
        (None, None, 2),
        (2, 2, 2),
    ]
    assert updates[0][3].startswith("Running concurrency 1")
    assert updates[-1][3].startswith("Completed concurrency 2")
    assert "progress_callback" not in runner.__getstate__()
