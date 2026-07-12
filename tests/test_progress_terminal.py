from types import SimpleNamespace

from vectordb_bench import interface as interface_module
from vectordb_bench.frontend.components.run_test import submitTask as submit_task_module
from vectordb_bench.interface import SIGNAL, BenchMarkRunner
from vectordb_bench.metric import Metric
from vectordb_bench.models import ProgressStage, ProgressStatus, ProgressUpdate


def _progress_update(**overrides) -> ProgressUpdate:
    values = {
        "run_id": "run-1",
        "case_index": 0,
        "case_total": 1,
        "stage": ProgressStage.FINALIZE,
        "stage_index": 6,
        "stage_total": 7,
        "status": ProgressStatus.RUNNING,
        "message": "Preparing benchmark result",
        "started_at": 100.0,
        "updated_at": 101.0,
    }
    values.update(overrides)
    return ProgressUpdate(**values)


class _SignalConnection:
    def __init__(self, messages=(), *, eof=False):
        self.messages = list(messages)
        self.eof = eof
        self.closed = False

    def poll(self):
        return self.eof or bool(self.messages)

    def recv(self):
        if self.messages:
            return self.messages.pop(0)
        raise EOFError

    def close(self):
        self.closed = True


class _RunningTask:
    run_id = "run-1"
    case_runners = []


def test_parent_pipe_eof_marks_active_progress_failed(monkeypatch):
    runner = BenchMarkRunner()
    runner.running_task = _RunningTask()
    runner.latest_progress = _progress_update(stage=ProgressStage.INSERT, stage_index=2)
    runner.receive_conn = _SignalConnection(eof=True)
    monkeypatch.setattr(interface_module, "kill_proc_tree", lambda: None)

    assert runner.has_running() is False
    assert runner.latest_error == "Benchmark process ended without a completion signal."
    assert runner.latest_progress.status == ProgressStatus.FAILED
    assert runner.latest_progress.stage == ProgressStage.INSERT


def test_error_signal_preserves_progress_context_and_marks_failed(monkeypatch):
    runner = BenchMarkRunner()
    runner.running_task = _RunningTask()
    runner.latest_progress = _progress_update(stage=ProgressStage.OPTIMIZE, stage_index=3)
    runner.receive_conn = _SignalConnection([(SIGNAL.ERROR, "result write failed")])
    monkeypatch.setattr(interface_module, "kill_proc_tree", lambda: None)

    assert runner.has_running() is False
    assert runner.latest_error == "result write failed"
    assert runner.latest_progress.status == ProgressStatus.FAILED
    assert runner.latest_progress.stage == ProgressStage.OPTIMIZE
    assert runner.latest_progress.message == "result write failed"
    assert runner.latest_progress.started_at == 100.0


def test_stop_preserves_progress_context_and_marks_cancelled():
    runner = BenchMarkRunner()
    runner.latest_progress = _progress_update(stage=ProgressStage.SEARCH_CONCURRENT, stage_index=4)

    runner.stop_running()

    assert runner.latest_progress.status == ProgressStatus.CANCELLED
    assert runner.latest_progress.stage == ProgressStage.SEARCH_CONCURRENT
    assert runner.latest_progress.message == "Benchmark cancelled"
    assert runner.latest_progress.started_at == 100.0


def test_terminal_progress_renders_after_task_finishes(monkeypatch):
    terminal_progress = _progress_update(status=ProgressStatus.COMPLETED, message="Benchmark result saved")
    fake_runner = SimpleNamespace(
        has_running=lambda: False,
        get_progress=lambda: terminal_progress,
        latest_error="",
    )
    rendered = []

    class FakeStreamlit:
        @staticmethod
        def fragment(**_kwargs):
            return lambda func: func

        @staticmethod
        def button(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

        @staticmethod
        def warning(*_args, **_kwargs):
            return None

    monkeypatch.setattr(submit_task_module, "benchmark_runner", fake_runner)
    monkeypatch.setattr(submit_task_module, "st", FakeStreamlit())
    monkeypatch.setattr(submit_task_module, "advancedSettings", lambda _container: (False, False, 10, "1", 1, 1))
    monkeypatch.setattr(
        submit_task_module,
        "render_task_progress",
        lambda container, progress: rendered.append((container, progress)),
    )

    submit_task_module.controlPanel(SimpleNamespace(), [], "demo", True)

    assert rendered == [(submit_task_module.st, terminal_progress)]


def test_serialized_terminal_progress_is_recognized():
    assert submit_task_module._is_terminal_progress({"status": "completed"})


class _RecordingConnection:
    def __init__(self, events):
        self.events = events

    def send(self, payload):
        self.events.append(payload)

    def close(self):
        self.events.append("close")


class _CaseRunner:
    config = SimpleNamespace(stages=[])

    def set_progress_callback(self, callback, **_kwargs):
        self.progress_callback = callback

    def load_reuse_key(self):
        return None

    def run(self, _drop_old):
        return Metric()

    def display(self):
        return "case"


class _AsyncRunningTask:
    run_id = "run-1"
    task_label = "demo"

    def __init__(self):
        self.case_runners = [_CaseRunner()]

    def num_cases(self):
        return len(self.case_runners)


class _CaseResult:
    def __init__(self, metrics, task_config):
        self.metrics = metrics
        self.task_config = task_config
        self.label = "normal"


def _install_result_double(monkeypatch, events, *, flush_error=None):
    class FakeTestResult:
        def __init__(self, **_kwargs):
            pass

        def display(self):
            pass

        def flush(self):
            events.append("flush")
            if flush_error is not None:
                raise flush_error

    monkeypatch.setattr(interface_module, "CaseResult", _CaseResult)
    monkeypatch.setattr(interface_module, "TestResult", FakeTestResult)


def test_final_completion_is_sent_only_after_result_flush(monkeypatch):
    events = []
    _install_result_double(monkeypatch, events)

    BenchMarkRunner()._async_task_v2(_AsyncRunningTask(), _RecordingConnection(events))

    flush_index = events.index("flush")
    completed_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == SIGNAL.PROGRESS and event[1].status == ProgressStatus.COMPLETED
    )
    assert flush_index < completed_index
    assert events[completed_index][1].message == "Benchmark result saved"
    assert events[completed_index + 1] == (SIGNAL.SUCCESS, None)


def test_flush_failure_sends_error_without_final_completion(monkeypatch):
    events = []
    _install_result_double(monkeypatch, events, flush_error=OSError("disk full"))

    BenchMarkRunner()._async_task_v2(_AsyncRunningTask(), _RecordingConnection(events))

    progress_updates = [event[1] for event in events if isinstance(event, tuple) and event[0] == SIGNAL.PROGRESS]
    assert progress_updates[-1].status == ProgressStatus.RUNNING
    assert not any(update.status == ProgressStatus.COMPLETED for update in progress_updates)
    assert events[-2][0] == SIGNAL.ERROR
    assert "disk full" in events[-2][1]
