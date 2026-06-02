from vectordb_bench.frontend.components.run_test import submitTask


class FakeBenchmarkRunner:
    latest_error = ""

    def __init__(self, running=False):
        self.running = running

    def has_running(self):
        return self.running

    def get_current_task_id(self):
        return 0

    def get_tasks_count(self):
        return 1

    def stop_running(self):
        self.running = False


class FakeStreamlit:
    def __init__(self):
        self.fragment_run_every = []
        self.button_calls = []

    def fragment(self, run_every=None):
        self.fragment_run_every.append(run_every)

        def decorator(fn):
            fn()
            return fn

        return decorator

    def progress(self, *args, **kwargs):
        pass

    def columns(self, count):
        return [self for _ in range(count)]

    def button(self, *args, **kwargs):
        self.button_calls.append((args, kwargs))
        return False

    def rerun(self):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _patch_control_panel_dependencies(monkeypatch, running):
    fake_st = FakeStreamlit()
    fake_runner = FakeBenchmarkRunner(running=running)
    monkeypatch.setattr(submitTask, "st", fake_st)
    monkeypatch.setattr(submitTask, "benchmark_runner", fake_runner)
    monkeypatch.setattr(
        submitTask,
        "advancedSettings",
        lambda container: (False, False, 100, "1,5", 30, 0),
    )
    return fake_st


def test_run_test_control_panel_does_not_create_fragment_when_idle(monkeypatch):
    fake_st = _patch_control_panel_dependencies(monkeypatch, running=False)

    submitTask.controlPanel(object(), tasks=[], taskLabel="test", isAllValid=True)

    assert fake_st.fragment_run_every == []


def test_run_test_control_panel_does_not_create_fragment_while_running(monkeypatch):
    fake_st = _patch_control_panel_dependencies(monkeypatch, running=True)

    submitTask.controlPanel(object(), tasks=[], taskLabel="test", isAllValid=True)

    assert fake_st.fragment_run_every == []
    assert any(kwargs.get("key") == "refresh-status-btn" for _, kwargs in fake_st.button_calls)
