import json
import os
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_milvus import TestMilvusOptimize as OptimizeFixture

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.stage_events import PREFIX, stage
from vectordb_bench.backend.task_runner import CaseRunner


def records(output: str):
    return [json.loads(line[len(PREFIX) :]) for line in output.splitlines() if line.startswith(PREFIX)]


def test_stage_events_are_opt_in_and_report_failure(monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]):
    monkeypatch.delenv("E2E_STAGE_EVENTS", raising=False)
    with stage("insert"):
        pass
    assert not capfd.readouterr().out
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_RUN_ID", "run")
    monkeypatch.setenv("E2E_CASE_ID", "case")
    monkeypatch.setenv("E2E_ATTEMPT", "2")
    with pytest.raises(ValueError, match="failure"), stage("search_concurrent", concurrency=5):
        raise ValueError("failure")
    start, end = records(capfd.readouterr().out)
    assert (start["state"], end["state"]) == ("start", "failed")
    assert start["span_id"] == end["span_id"]
    assert (start["run_id"], start["case_id"], start["attempt"], start["concurrency"]) == ("run", "case", 2, 5)
    assert end["monotonic_ns"] >= start["monotonic_ns"]


def test_stage_events_cross_subprocess_stdout():
    env = {**os.environ, "E2E_STAGE_EVENTS": "1", "E2E_RUN_ID": "child"}
    code = "from vectordb_bench.backend.stage_events import stage\nwith stage('compaction'): pass"
    child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    events = records(child.stdout)
    assert [event["state"] for event in events] == ["start", "end"]
    assert all(event["run_id"] == "child" for event in events)


@pytest.mark.parametrize("failure", [False, True])
def test_index_readiness_reports_each_wait_and_failure(monkeypatch, capfd, failure):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db._main_index_name = "vector_index"
    db.client.describe_index.side_effect = [
        {"pending_index_rows": 10},
        RuntimeError("index read failed") if failure else {"pending_index_rows": 0},
    ]
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.sleep") as sleep:
        if failure:
            with pytest.raises(RuntimeError, match="index read failed"):
                Milvus._wait_for_index(db)
        else:
            Milvus._wait_for_index(db)
        sleep.assert_called_once_with(5)
    start, end = records(capfd.readouterr().out)
    assert start["stage"] == end["stage"] == "index"
    assert start["span_id"] == end["span_id"]
    assert end["state"] == ("failed" if failure else "end")
    assert start["index_name"] == db._main_index_name


def test_compaction_records_before_after_without_changing_timing(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.list_persistent_segments.side_effect = [[1, 2, 3], [4]]
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.perf_counter", side_effect=[10, 13]):
        db._optimize()
    events = records(capfd.readouterr().out)
    assert [event["state"] for event in events] == ["start", "end", "segments"]
    observation = db.optimization_metrics["compaction_segments"][0]
    assert observation["before"]["count"] == 3
    assert observation["after"]["count"] == 1
    assert events[-1]["observation"] == observation
    assert db.optimization_metrics["compaction_duration"] == 3


def test_segment_read_failure_is_not_reported_as_zero(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.list_persistent_segments.side_effect = [RuntimeError("unavailable"), [1]]
    db._optimize()
    observation = db.optimization_metrics["compaction_segments"][0]
    assert observation["before"]["status"] == "unavailable"
    assert observation["before"]["count"] is None
    assert observation["after"]["count"] == 1
    assert records(capfd.readouterr().out)[-1]["state"] == "segments"


def test_concurrency_events_follow_worker_boundaries(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    manager = MagicMock()
    manager.__enter__.return_value.Queue.return_value.qsize.return_value = 100
    future = MagicMock()
    future.result.return_value = (10, 0, {"count": 1, "p99": 0.1, "p95": 0.1, "p50": 0.1, "avg": 0.1})
    executor = MagicMock()
    executor.__enter__.return_value.submit.return_value = future
    runner = MultiProcessingSearchRunner.__new__(MultiProcessingSearchRunner)
    runner.concurrencies, runner.filters, runner.test_data = [1, 4], None, [[1.0]]
    with (
        patch("vectordb_bench.backend.runner.mp_runner.mp.Manager", return_value=manager),
        patch("vectordb_bench.backend.runner.mp_runner.concurrent.futures.ProcessPoolExecutor", return_value=executor),
        patch.object(runner, "get_mp_context", return_value=None),
    ):
        result = runner._run_by_dur(1)
    events = records(capfd.readouterr().out)
    assert [(event["state"], event["concurrency"]) for event in events] == [
        ("start", 1),
        ("end", 1),
        ("start", 4),
        ("end", 4),
    ]
    assert result[2] == [1, 4]


class ProcessEventDatabase:
    def init(self):
        return nullcontext()

    def optimize(self, data_size: int | None = None):
        with stage("compaction"):
            self.optimization_metrics = {"compaction_status": "completed", "compaction_duration": 1.0}


class SerialEventDatabase:
    name = "Milvus"

    def supports_payload_profile(self, _profile):
        return True

    def init(self):
        return nullcontext()

    def prepare_filter(self, _filters):
        pass

    def search_embedding(self, _query, _k):
        return [0]


def test_compaction_events_cross_real_worker_process(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    runner = CaseRunner.model_construct(
        config=SimpleNamespace(db=DB.Milvus),
        ca=SimpleNamespace(dataset=SimpleNamespace(data=SimpleNamespace(size=1)), optimize_timeout=30),
        db=ProcessEventDatabase(),
    )
    runner._optimize()
    events = records(capfd.readouterr().out)
    assert [(event["stage"], event["state"]) for event in events] == [("compaction", "start"), ("compaction", "end")]
    assert runner.optimization_metrics["compaction_duration"] == 1.0


def test_nested_stage_events_link_to_parent_and_restore_context(monkeypatch, capfd):
    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    with stage("index_compaction") as parent:
        with pytest.raises(ValueError), stage("index"):
            raise ValueError("failed")
        with stage("compaction"):
            pass
    with stage("collection_load"):
        pass
    events = records(capfd.readouterr().out)
    children = [event for event in events if event["stage"] in ("index", "compaction")]
    assert all(event["parent_span_id"] == parent for event in children)
    assert all("parent_span_id" not in event for event in events if event["stage"] == "collection_load")


def test_reload_measures_unload_and_load_after_combined_stage(monkeypatch, capfd):
    from pymilvus.client.types import LoadState

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.get_load_state.side_effect = [{"state": LoadState.NotLoad}, {"state": LoadState.Loaded}]
    db._optimize()
    events = records(capfd.readouterr().out)
    parent = next(event for event in events if event["stage"] == "index_compaction" and event["state"] == "start")
    compaction = next(event for event in events if event["stage"] == "compaction" and event["state"] == "start")
    assert compaction["parent_span_id"] == parent["span_id"]
    unload, load = [
        next(event for event in events if event["stage"] == name and event["state"] == "end")
        for name in ("collection_unload", "collection_load")
    ]
    assert unload["reload_id"] == load["reload_id"]
    assert unload["profile"] is False
    db.client.release_collection.assert_called_once()
    db.client.load_collection.assert_called_once()
    db.client.refresh_load.assert_not_called()


@pytest.mark.parametrize("operation", ["release_collection", "load_collection"])
def test_reload_failure_is_not_reported_as_completed(monkeypatch, capfd, operation):
    from pymilvus.client.types import LoadState

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.get_load_state.return_value = {"state": LoadState.NotLoad}
    getattr(db.client, operation).side_effect = RuntimeError("reload failed")
    with pytest.raises(RuntimeError, match="reload failed"):
        db._optimize()
    events = records(capfd.readouterr().out)
    failed = "collection_unload" if operation == "release_collection" else "collection_load"
    assert any(event["stage"] == failed and event["state"] == "failed" for event in events)
    assert not any(event["stage"] == "collection_load" and event["state"] == "end" for event in events)
    assert db.optimization_metrics["compaction_status"] == "completed"


def test_reload_fails_when_collection_disappears():
    from pymilvus.client.types import LoadState

    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    db.client.get_load_state.return_value = {"state": LoadState.NotExist}
    with pytest.raises(RuntimeError, match="disappeared"):
        db._wait_collection_state(LoadState.NotLoad, float("inf"))


def test_reload_state_wait_has_deadline():
    from pymilvus.client.types import LoadState

    db = OptimizeFixture()._milvus(force_merge_enabled=False)
    with patch("vectordb_bench.backend.clients.milvus.milvus.time.monotonic", return_value=10):
        with pytest.raises(TimeoutError, match="before timeout"):
            db._wait_collection_state(LoadState.Loaded, 10)
    db.client.get_load_state.assert_not_called()


def test_serial_details_use_actual_query_mean_and_max(monkeypatch):
    from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner

    db = MagicMock()
    db.name = "Milvus"
    db.supports_payload_profile.return_value = True
    db.init.return_value = nullcontext()
    db.search_embedding.return_value = [0]
    runner = SerialSearchRunner(db=db, test_data=[[1.0], [2.0]], ground_truth=[[0], [0]], k=1)
    runner.e2e_serial_details = True
    values = iter([0.0, 0.2, 1.0, 1.5])
    monkeypatch.setattr("vectordb_bench.backend.runner.serial_runner.time.perf_counter", lambda: next(values))
    result = runner.search((runner.test_data, runner.ground_truth))
    assert result[0] == 1.0
    assert result[-1]["query_count"] == 2
    assert result[-1]["average_latency_seconds"] == pytest.approx(0.35)
    assert result[-1]["maximum_latency_seconds"] == pytest.approx(0.5)


def test_serial_details_cross_real_worker_process():
    from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner

    runner = SerialSearchRunner(db=SerialEventDatabase(), test_data=[[1.0], [2.0]], ground_truth=[[0], [0]], k=1)
    runner.e2e_serial_details = True
    result, _ = runner.run()
    assert result[0] == 1.0
    assert result[-1]["query_count"] == 2
    assert result[-1]["maximum_latency_seconds"] >= result[-1]["average_latency_seconds"] >= 0


def test_existing_serial_result_shape_is_unchanged():
    from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner

    runner = SerialSearchRunner(db=SerialEventDatabase(), test_data=[[1.0]], ground_truth=[[0]], k=1)
    assert len(runner.search((runner.test_data, runner.ground_truth))) == 6


def test_timed_serial_replays_full_cold_sequence_without_recall(monkeypatch):
    from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner

    db = MagicMock()
    db.name = "Milvus"
    db.supports_payload_profile.return_value = True
    db.init.return_value = nullcontext()
    clock = [0.0]
    step = [0.6]
    def search(_query, _k):
        clock[0] += step[0]
        return [0]
    db.search_embedding.side_effect = search
    monkeypatch.setattr("vectordb_bench.backend.runner.serial_runner.time.perf_counter", lambda: clock[0])
    runner = SerialSearchRunner(db=db, test_data=[[1.0], [2.0], [3.0]], ground_truth=[[0]] * 3, k=1)
    cold = runner.timed_search(2)
    cold_queries = [call.args[0] for call in db.search_embedding.call_args_list]
    assert cold["query_count"] == 4
    assert cold["duration_seconds"] >= 2
    assert cold["average_latency_seconds"] == pytest.approx(0.6)
    db.search_embedding.reset_mock()
    step[0] = 0.3
    warm = runner.timed_search(2, cold["query_count"])
    warm_queries = [call.args[0] for call in db.search_embedding.call_args_list]
    assert warm["query_count"] % cold["query_count"] == 0
    assert warm["query_count"] == 8
    assert warm_queries == cold_queries * (warm["query_count"] // cold["query_count"])


def test_timed_serial_crosses_worker_process():
    from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner

    runner = SerialSearchRunner(db=SerialEventDatabase(), test_data=[[1.0], [2.0]], ground_truth=[[0], [0]], k=1)
    cold = runner.run_timed(0.01)
    warm = runner.run_timed(0.01, cold["query_count"])
    assert cold["duration_seconds"] >= 0.01
    assert warm["duration_seconds"] >= 0.01
    assert warm["query_count"] % cold["query_count"] == 0


def test_e2e_search_runs_serial_cold_then_hot_then_concurrency(monkeypatch, capfd):
    from vectordb_bench.metric import Metric
    from vectordb_bench.models import TaskStage

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    search = SimpleNamespace(num_concurrency=[100], concurrency_duration=60)
    serial = SimpleNamespace(measure_recall=True, ground_truth=[[0]], test_data=[[1.0]], k=1, _validate_ground_truth=MagicMock())
    runner = CaseRunner.model_construct(config=SimpleNamespace(db=DB.ZillizCloud,
        stages=[TaskStage.LOAD, TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
        case_config=SimpleNamespace(concurrency_search_config=search, nq=1)), ca=SimpleNamespace(label=None),
        serial_search_runner=serial)
    order = []
    def timed(duration, replay_count):
        order.append(("timed", duration, replay_count))
        return {"maximum_latency_seconds": 0.5, "average_latency_seconds": 0.2, "duration_seconds": 20.0, "query_count": 1}
    serial.run_timed = timed
    def serial_search(_runner):
        order.append("recall")
        return (0.95, 0.9, 0.3, 0.2, 0.1, {})
    def concurrent_search(_runner):
        order.append("concurrent")
        return (20.0, [100], [20.0], [0.1], [0.1], [0.1], [0.1])
    with patch.object(CaseRunner, "_serial_search", serial_search), patch.object(CaseRunner, "_conc_search", concurrent_search):
        metric = Metric()
        runner._run_e2e_search(metric)
    assert order == [("timed", 20, None), ("timed", 20, 1), "recall", "concurrent"]
    assert metric.conc_num_list == [100]
    assert metric.additional_parameters["e2e_recall"]["recall"] == 0.95
    events = records(capfd.readouterr().out)
    assert [(event["stage"], event["state"]) for event in events] == [
        ("search_serial_cold", "start"), ("search_serial_cold", "end"),
        ("search_serial_hot", "start"), ("search_serial_hot", "end"),
        ("search_recall", "start"), ("search_recall", "end")]
    assert metric.additional_parameters["e2e_serial_search"]["hot"]["span_id"] == events[2]["span_id"]
    assert metric.additional_parameters["e2e_recall"]["span_id"] == events[4]["span_id"]


def test_e2e_search_rejects_missing_ground_truth(monkeypatch):
    from vectordb_bench.metric import Metric
    from vectordb_bench.models import TaskStage

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    search = SimpleNamespace(num_concurrency=[100], concurrency_duration=60)
    runner = CaseRunner.model_construct(config=SimpleNamespace(db=DB.ZillizCloud,
        stages=[TaskStage.LOAD, TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
        case_config=SimpleNamespace(concurrency_search_config=search, nq=1)), ca=SimpleNamespace(label=None),
        serial_search_runner=SimpleNamespace(measure_recall=True, ground_truth=None))
    with pytest.raises(ValueError, match="ground truth"):
        runner._run_e2e_search(Metric())


def test_e2e_search_rejects_mismatched_ground_truth_before_timing(monkeypatch, capfd):
    from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
    from vectordb_bench.metric import Metric
    from vectordb_bench.models import TaskStage

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    search = SimpleNamespace(num_concurrency=[100], concurrency_duration=60)
    serial = SerialSearchRunner(db=SerialEventDatabase(), test_data=[[1.0], [2.0]], ground_truth=[[0]], k=1)
    runner = CaseRunner.model_construct(config=SimpleNamespace(db=DB.ZillizCloud,
        stages=[TaskStage.LOAD, TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
        case_config=SimpleNamespace(concurrency_search_config=search, nq=1)), ca=SimpleNamespace(label=None),
        serial_search_runner=serial)
    with pytest.raises(ValueError, match="does not match ground truth"):
        runner._run_e2e_search(Metric())
    assert records(capfd.readouterr().out) == []


def test_performance_case_runs_reload_before_serial_and_concurrent(monkeypatch, capfd):
    from dataclasses import asdict

    from vectordb_bench.models import TaskStage

    monkeypatch.setenv("E2E_STAGE_EVENTS", "1")
    monkeypatch.setenv("E2E_COLLECTION_RELOAD", "1")
    monkeypatch.setenv("E2E_COLD_HOT_SEARCH", "1")
    search = SimpleNamespace(num_concurrency=[100], concurrency_duration=60)
    serial = SimpleNamespace(measure_recall=True, ground_truth=[[0]], test_data=[[1.0]], k=1, _validate_ground_truth=MagicMock())
    cfg = SimpleNamespace(db=DB.ZillizCloud, stages=[TaskStage.LOAD, TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
                          case_config=SimpleNamespace(concurrency_search_config=search, nq=1),
                          insert_batch_size=100, load_concurrency=1)
    runner = CaseRunner.model_construct(config=cfg, ca=SimpleNamespace(label=None, dataset=SimpleNamespace(gt_data=None)),
                                        serial_search_runner=serial)
    order = []
    def load(_runner):
        order.append("insert")
        return 10, 1.0
    def optimize(_runner):
        order.append("reload")
        return 2.0
    def timed(_duration, _replay_count):
        order.append("timed")
        return {"maximum_latency_seconds": .5, "average_latency_seconds": .2, "duration_seconds": 20., "query_count": 1}
    serial.run_timed = timed
    def serial_search(_runner):
        order.append("recall")
        return (1., 1., .3, .2, .1, {})
    def concurrent_search(_runner):
        order.append("concurrent")
        return (20., [100], [20.], [.1], [.1], [.1], [.1])
    with (patch.object(CaseRunner, "_load_data", load), patch.object(CaseRunner, "_optimize", optimize),
          patch.object(CaseRunner, "_init_search_runners"), patch.object(CaseRunner, "_serial_search", serial_search),
          patch.object(CaseRunner, "_conc_search", concurrent_search)):
        result = runner._run_perf_case()
    assert order == ["insert", "reload", "timed", "timed", "recall", "concurrent"]
    assert result.additional_parameters["e2e_serial_search"]["cold"]["query_count"] == 1
    decoded = json.loads(json.dumps(asdict(result)))
    assert decoded["additional_parameters"]["e2e_recall"]["recall"] == 1.0
    assert [event["stage"] for event in records(capfd.readouterr().out) if event["state"] == "start"] == [
        "insert", "search_serial_cold", "search_serial_hot", "search_recall"]


def test_e2e_search_rejects_missing_stages_before_loading(monkeypatch):
    monkeypatch.setenv("E2E_COLD_HOT_SEARCH", "1")
    runner = CaseRunner.model_construct(config=SimpleNamespace(stages=[]), ca=SimpleNamespace(label=None))
    with patch.object(CaseRunner, "_load_data") as load:
        with pytest.raises(ValueError, match="both search stages"):
            runner._run_perf_case()
    load.assert_not_called()
