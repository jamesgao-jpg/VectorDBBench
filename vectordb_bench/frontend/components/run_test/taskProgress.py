from __future__ import annotations

import html
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from vectordb_bench.frontend.config import styles

PROGRESS_STAGES = (
    ("setup", "Setup"),
    ("download", "Download"),
    ("insert", "Insert"),
    ("optimize", "Optimize"),
    ("search_concurrent", "Concurrent search"),
    ("search_serial", "Serial search"),
    ("finalize", "Finalize"),
)

STAGE_LABELS = dict(PROGRESS_STAGES)
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(frozen=True)
class StageStep:
    key: str
    label: str
    state: str


@dataclass(frozen=True)
class ProgressView:
    case_label: str
    stage_position_label: str
    stage_label: str
    status: str
    status_label: str
    message: str
    detail: str
    ratio: float | None
    indeterminate: bool
    steps: tuple[StageStep, ...]


def _read(update: object, field: str, default: Any = None) -> Any:
    if isinstance(update, Mapping):
        return update.get(field, default)
    return getattr(update, field, default)


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower() if raw is not None else ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> float | None:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _elapsed_seconds(update: object, status: str, now: float) -> float:
    started_at = _timestamp(_read(update, "started_at"))
    if started_at is None:
        return 0.0
    end = now
    if status in TERMINAL_STATUSES:
        end = _timestamp(_read(update, "updated_at")) or now
    return max(0.0, end - started_at)


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_number(value: float) -> str:
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _format_byte_pair(current: float, total: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    scale = 1.0
    unit = units[0]
    for candidate in units[1:]:
        if max(current, total) < scale * 1024:
            break
        scale *= 1024
        unit = candidate
    precision = 0 if unit == "B" else 2
    return f"{current / scale:,.{precision}f} / {total / scale:,.{precision}f} {unit}"


def _format_quantity_pair(current: float, total: float, unit: str) -> str:
    if unit.lower() in {"b", "byte", "bytes"}:
        return _format_byte_pair(current, total)
    suffix = f" {unit}" if unit else ""
    return f"{_format_number(current)} / {_format_number(total)}{suffix}"


def _active_stage_index(update: object, stage: str) -> int:
    stage_keys = [key for key, _ in PROGRESS_STAGES]
    if stage in stage_keys:
        return stage_keys.index(stage)
    configured = _number(_read(update, "stage_index"))
    if configured is None:
        return 0
    return min(max(int(configured), 0), len(PROGRESS_STAGES) - 1)


def _stage_steps(active_index: int, status: str) -> tuple[StageStep, ...]:
    steps = []
    for index, (key, label) in enumerate(PROGRESS_STAGES):
        if index < active_index or (index == active_index and status == "completed"):
            state = "completed"
        elif index == active_index:
            state = status if status in {"failed", "cancelled"} else "running"
        else:
            state = "pending"
        steps.append(StageStep(key=key, label=label, state=state))
    return tuple(steps)


def build_progress_view(update: object, now: float | None = None) -> ProgressView:
    now = time.time() if now is None else now
    stage = _enum_value(_read(update, "stage")) or "setup"
    status = _enum_value(_read(update, "status")) or "running"
    active_index = _active_stage_index(update, stage)
    stage_label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
    stage_total = int(_number(_read(update, "stage_total")) or len(PROGRESS_STAGES))

    case_index = int(_number(_read(update, "case_index")) or 0)
    case_total = max(1, int(_number(_read(update, "case_total")) or 1))
    case_position = min(max(case_index + 1, 1), case_total)

    elapsed = _elapsed_seconds(update, status, now)
    current = _number(_read(update, "current"))
    total = _number(_read(update, "total"))
    duration_hint = _number(_read(update, "duration_hint_seconds"))
    unit = str(_read(update, "unit") or "")
    ratio = None
    indeterminate = status == "running"

    if status == "running" and duration_hint is not None and duration_hint > 0:
        interval_started = _timestamp(_read(update, "updated_at")) or now
        interval_elapsed = max(0.0, now - interval_started)
        ratio = min(interval_elapsed / duration_hint, 1.0)
        elapsed_text = _format_duration(min(interval_elapsed, duration_hint))
        detail = f"{elapsed_text} / {_format_duration(duration_hint)} &middot; {ratio:.0%}"
        indeterminate = False
    elif current is not None and total is not None and total > 0:
        ratio = min(max(current / total, 0.0), 1.0)
        detail = f"{_format_quantity_pair(current, total, unit)} &middot; {ratio:.0%}"
        indeterminate = False
    elif status == "completed":
        ratio = 1.0
        detail = f"Completed in {_format_duration(elapsed)}"
        indeterminate = False
    else:
        detail = f"Elapsed {_format_duration(elapsed)}"

    status_labels = {
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "cancelled": "Cancelled",
    }
    return ProgressView(
        case_label=f"Case {case_position} of {case_total}",
        stage_position_label=f"Stage {active_index + 1} of {stage_total}",
        stage_label=stage_label,
        status=status,
        status_label=status_labels.get(status, status.title()),
        message=str(_read(update, "message") or stage_label),
        detail=detail,
        ratio=ratio,
        indeterminate=indeterminate,
        steps=_stage_steps(active_index, status),
    )


def _step_icon(state: str) -> str:
    if state == "completed":
        return "&#10003;"
    if state == "failed":
        return "!"
    if state == "cancelled":
        return "&times;"
    return ""


def build_progress_html(view: ProgressView) -> str:
    steps = "".join(
        f"<div class='vdb-progress-step {step.state}'>"
        f"<span class='vdb-progress-dot'>{_step_icon(step.state)}</span>"
        f"<span class='vdb-progress-step-label'>{html.escape(step.label)}</span>"
        "</div>"
        for step in view.steps
    )
    progress_class = "indeterminate" if view.indeterminate else "determinate"
    status_class = view.status if view.status in TERMINAL_STATUSES | {"running"} else "running"
    width = 28 if view.indeterminate else round((view.ratio or 0) * 100, 1)
    return f"""
<style>
.vdb-progress-shell {{ min-height: {styles.TASK_PROGRESS_MIN_HEIGHT}px; margin: 4px 0 14px; }}
.vdb-progress-meta {{
  display: flex; justify-content: space-between; margin-bottom: 12px;
  color: #526174; font-size: 13px;
}}
.vdb-progress-strip {{ display: flex; min-width: 720px; align-items: flex-start; }}
.vdb-progress-strip-wrap {{ overflow-x: auto; padding-bottom: 4px; }}
.vdb-progress-step {{
  position: relative; flex: 1; min-width: 92px; text-align: center;
  color: #8a94a3; font-size: 12px;
}}
.vdb-progress-step::before {{
  content: ''; position: absolute; top: 9px; left: 0; right: 50%;
  height: 2px; background: #d9dee7;
}}
.vdb-progress-step::after {{
  content: ''; position: absolute; top: 9px; left: 50%; right: 0;
  height: 2px; background: #d9dee7;
}}
.vdb-progress-step:first-child::before, .vdb-progress-step:last-child::after {{ display: none; }}
.vdb-progress-step.completed::before, .vdb-progress-step.completed::after, .vdb-progress-step.running::before,
.vdb-progress-step.failed::before, .vdb-progress-step.cancelled::before {{ background: #16835d; }}
.vdb-progress-dot {{
  position: relative; z-index: 1; display: flex; width: 20px; height: 20px; margin: 0 auto 6px;
  align-items: center; justify-content: center; border: 2px solid #c8ced8; border-radius: 50%;
  background: #fff; color: #fff; font-size: 11px; line-height: 1;
}}
.vdb-progress-step.completed .vdb-progress-dot {{ border-color: #16835d; background: #16835d; }}
.vdb-progress-step.running .vdb-progress-dot {{ border-color: #2563eb; box-shadow: 0 0 0 4px #dbeafe; }}
.vdb-progress-step.failed .vdb-progress-dot {{ border-color: #c2413c; background: #c2413c; }}
.vdb-progress-step.cancelled .vdb-progress-dot {{ border-color: #64748b; background: #64748b; }}
.vdb-progress-step.completed, .vdb-progress-step.running {{ color: #273244; }}
.vdb-progress-active {{
  margin-top: 12px; padding: 11px 13px; border: 1px solid #dde2ea;
  border-radius: 6px; background: #f8fafc;
}}
.vdb-progress-active-head {{ display: flex; align-items: baseline; gap: 9px; min-width: 0; }}
.vdb-progress-active-title {{ color: #202938; font-size: 14px; font-weight: 650; white-space: nowrap; }}
.vdb-progress-message {{
  min-width: 0; flex: 1; overflow: hidden; color: #526174; font-size: 13px;
  text-overflow: ellipsis; white-space: nowrap;
}}
.vdb-progress-status {{ color: #526174; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
.vdb-progress-track {{ height: 6px; margin-top: 9px; overflow: hidden; border-radius: 3px; background: #e3e7ee; }}
.vdb-progress-fill {{ height: 100%; border-radius: 3px; background: #2563eb; }}
.vdb-progress-shell.completed .vdb-progress-fill {{ background: #16835d; }}
.vdb-progress-shell.failed .vdb-progress-fill {{ background: #c2413c; }}
.vdb-progress-shell.cancelled .vdb-progress-fill {{ background: #64748b; }}
.vdb-progress-fill.indeterminate {{ width: 28%; animation: vdb-progress-move 1.4s ease-in-out infinite; }}
.vdb-progress-detail {{ margin-top: 6px; color: #647489; font-size: 12px; }}
@keyframes vdb-progress-move {{ 0% {{ transform: translateX(-110%); }} 100% {{ transform: translateX(360%); }} }}
@media (prefers-reduced-motion: reduce) {{ .vdb-progress-fill.indeterminate {{ animation: none; }} }}
</style>
<div class="vdb-progress-shell {status_class}">
  <div class="vdb-progress-meta">
    <span>{html.escape(view.case_label)}</span><span>{html.escape(view.stage_position_label)}</span>
  </div>
  <div class="vdb-progress-strip-wrap"><div class="vdb-progress-strip">{steps}</div></div>
  <div class="vdb-progress-active">
    <div class="vdb-progress-active-head">
      <span class="vdb-progress-active-title">{html.escape(view.stage_label)}</span>
      <span class="vdb-progress-message">{html.escape(view.message)}</span>
      <span class="vdb-progress-status">{html.escape(view.status_label)}</span>
    </div>
    <div class="vdb-progress-track">
      <div class="vdb-progress-fill {progress_class}" style="width: {width}%;"></div>
    </div>
    <div class="vdb-progress-detail">{view.detail}</div>
  </div>
</div>
"""


def render_task_progress(container: Any, update: object, now: float | None = None) -> ProgressView:
    view = build_progress_view(update, now=now)
    container.markdown(build_progress_html(view), unsafe_allow_html=True)
    return view
