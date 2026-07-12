from types import SimpleNamespace

from vectordb_bench.frontend.components.run_test.taskProgress import (
    build_progress_html,
    build_progress_view,
    render_task_progress,
)


def test_build_progress_view_from_dict_with_byte_progress():
    view = build_progress_view(
        {
            "case_index": 0,
            "case_total": 2,
            "stage": "download",
            "stage_index": 1,
            "stage_total": 7,
            "status": "running",
            "message": "Downloading Cohere 1M",
            "current": 1024**3,
            "total": 3 * 1024**3,
            "unit": "bytes",
            "started_at": 100.0,
            "updated_at": 105.0,
        },
        now=110.0,
    )

    assert view.case_label == "Case 1 of 2"
    assert view.stage_position_label == "Stage 2 of 7"
    assert view.stage_label == "Download"
    assert view.detail == "1.00 / 3.00 GiB &middot; 33%"
    assert view.ratio == 1 / 3
    assert [step.state for step in view.steps] == [
        "completed",
        "running",
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
    ]


def test_build_progress_view_from_object_with_duration_hint():
    update = SimpleNamespace(
        case_index=1,
        case_total=3,
        stage="search_concurrent",
        stage_index=4,
        stage_total=7,
        status="running",
        message="Concurrency 30",
        current=None,
        total=None,
        unit=None,
        started_at=50.0,
        updated_at=100.0,
        duration_hint_seconds=30,
    )

    view = build_progress_view(update, now=118.0)

    assert view.case_label == "Case 2 of 3"
    assert view.stage_label == "Concurrent search"
    assert view.detail == "18s / 30s &middot; 60%"
    assert view.ratio == 0.6
    assert view.indeterminate is False


def test_build_progress_view_uses_elapsed_time_for_indeterminate_stage():
    view = build_progress_view(
        {
            "case_index": 0,
            "case_total": 1,
            "stage": "optimize",
            "stage_index": 3,
            "stage_total": 7,
            "status": "running",
            "message": "Building HNSW index",
            "started_at": 100.0,
            "updated_at": 100.0,
        },
        now=358.0,
    )

    assert view.detail == "Elapsed 4m 18s"
    assert view.ratio is None
    assert view.indeterminate is True


def test_terminal_stage_stops_elapsed_clock_at_updated_at():
    view = build_progress_view(
        {
            "case_index": 0,
            "case_total": 1,
            "stage": "finalize",
            "stage_index": 6,
            "stage_total": 7,
            "status": "completed",
            "message": "Result saved",
            "started_at": "2026-07-12T02:00:00Z",
            "updated_at": "2026-07-12T02:00:42Z",
        },
        now=2_000_000_000.0,
    )

    assert view.detail == "Completed in 42s"
    assert view.ratio == 1.0
    assert all(step.state == "completed" for step in view.steps)


def test_failed_stage_marks_only_active_step_as_failed():
    view = build_progress_view(
        {
            "case_index": 0,
            "case_total": 1,
            "stage": "insert",
            "stage_index": 2,
            "stage_total": 7,
            "status": "failed",
            "message": "Insert failed",
            "started_at": 100.0,
            "updated_at": 112.0,
        },
        now=200.0,
    )

    assert view.detail == "Elapsed 12s"
    assert [step.state for step in view.steps[:4]] == ["completed", "completed", "failed", "pending"]
    assert 'class="vdb-progress-shell failed"' in build_progress_html(view)


def test_renderer_escapes_progress_message():
    class FakeContainer:
        def __init__(self):
            self.body = ""
            self.unsafe_allow_html = False

        def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
            self.body = body
            self.unsafe_allow_html = unsafe_allow_html

    update = {
        "case_index": 0,
        "case_total": 1,
        "stage": "insert",
        "stage_index": 2,
        "stage_total": 7,
        "status": "running",
        "message": "Insert <script>alert(1)</script>",
        "started_at": 100.0,
        "updated_at": 100.0,
    }
    container = FakeContainer()

    view = render_task_progress(container, update, now=101.0)

    assert view.stage_label == "Insert"
    assert container.unsafe_allow_html is True
    assert "<script>" not in container.body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in container.body
    assert container.body == build_progress_html(view)
