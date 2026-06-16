from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CLOUD_COLD_LATENCY_CASE_ID = 700

PRODUCT_NAMES = {
    "pinecone_serverless": "Pinecone Serverless",
    "turbopuffer": "Turbopuffer",
    "turbopuffer_pinned": "Turbopuffer Pinned 2 Replicas",
    "zilliz_cloud_cap_12cu": "Zilliz Cloud Capacity 12CU",
    "zilliz_cloud_capacity_12cu": "Zilliz Cloud Capacity 12CU",
    "zilliz_cloud_tiered_4cu": "Zilliz Cloud Tiered 4CU",
}

MODE_NAMES = {
    "unfiltered": "Unfiltered",
    "int_filter_0.9": "Int Filter 0.9",
}

REBENCH_NEEDED_PRODUCTS = {"zilliz_cloud_tiered_4cu"}


class CloudColdLatencyParseError(ValueError):
    pass


@dataclass(frozen=True)
class CloudColdLatencyRow:
    product_key: str
    product_name: str
    mode_key: str
    mode_display: str
    status: str
    filter_rate: float | None
    payload_profile: str
    query_count: int
    db: str
    db_label: str
    payload_bytes_per_query: int | None
    cold_first_query_seconds: float
    cold_p99_seconds: float
    cold_p95_seconds: float
    cold_avg_seconds: float
    warm_first_query_seconds: float
    warm_p99_seconds: float
    warm_p95_seconds: float
    warm_avg_seconds: float
    first_query_ratio: float | None
    p99_ratio: float | None
    p95_ratio: float | None
    avg_ratio: float | None
    load_duration: float | None
    insert_duration: float | None
    optimize_duration: float | None
    raw_path: str


def load_cloud_cold_latency_rows(raw_results_dir: Path | str) -> list[CloudColdLatencyRow]:
    root = Path(raw_results_dir)
    if not root.exists():
        return []

    rows = [_load_cloud_cold_latency_file(root, json_file) for json_file in sorted(root.rglob("result_*.json"))]
    rows.sort(
        key=lambda row: (
            _mode_sort_key(row.mode_key),
            row.product_name,
        )
    )
    return rows


def cloud_cold_latency_records(rows: list[CloudColdLatencyRow]) -> list[dict[str, Any]]:
    return [
        {
            "Product": row.product_name,
            "Mode": row.mode_display,
            "Status": row.status,
            "Payload": _payload_display(row.payload_profile),
            "Query Count": row.query_count,
            "Payload Bytes/Query": row.payload_bytes_per_query,
            "First Cold Query (s)": row.cold_first_query_seconds,
            "Cold P99 (s)": row.cold_p99_seconds,
            "Cold P95 (s)": row.cold_p95_seconds,
            "Cold Avg (s)": row.cold_avg_seconds,
            "Warm First Query (s)": row.warm_first_query_seconds,
            "Warm P99 (s)": row.warm_p99_seconds,
            "Warm P95 (s)": row.warm_p95_seconds,
            "Warm Avg (s)": row.warm_avg_seconds,
            "First Query Ratio": row.first_query_ratio,
            "P99 Ratio": row.p99_ratio,
            "P95 Ratio": row.p95_ratio,
            "Avg Ratio": row.avg_ratio,
            "Load Duration (s)": row.load_duration,
            "Insert Duration (s)": row.insert_duration,
            "Optimize Duration (s)": row.optimize_duration,
        }
        for row in rows
    ]


def _load_cloud_cold_latency_file(root: Path, json_file: Path) -> CloudColdLatencyRow:
    product_key, mode_key = _parse_cloud_cold_latency_path(root, json_file)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    result = _single_result(data, json_file)
    task_config = result.get("task_config", {})
    case_config = task_config.get("case_config", {})
    case_id = case_config.get("case_id")
    if case_id != CLOUD_COLD_LATENCY_CASE_ID:
        raise CloudColdLatencyParseError(
            f"{json_file} has case_id={case_id}; expected {CLOUD_COLD_LATENCY_CASE_ID}"
        )

    custom_case = case_config.get("custom_case") or {}
    metrics = result.get("metrics") or {}
    cold_latency = metrics.get("cold_latency") or {}
    if not isinstance(cold_latency, dict):
        raise CloudColdLatencyParseError(f"{json_file} missing metrics.cold_latency")

    cold_stats = cold_latency.get("cold_stats") or {}
    warm_stats = cold_latency.get("warm_stats") or {}
    ratios = cold_latency.get("cold_warm_ratio") or {}
    if not isinstance(cold_stats, dict) or not isinstance(warm_stats, dict) or not isinstance(ratios, dict):
        raise CloudColdLatencyParseError(f"{json_file} has invalid metrics.cold_latency shape")

    cold_first_query = _required_float(cold_stats, "first_query_latency", json_file)
    cold_p99 = _required_float(cold_stats, "p99_latency", json_file)
    cold_p95 = _required_float(cold_stats, "p95_latency", json_file)
    cold_avg = _required_float(cold_stats, "avg_latency", json_file)
    warm_first_query = _required_float(warm_stats, "first_query_latency", json_file)
    warm_p99 = _required_float(warm_stats, "p99_latency", json_file)
    warm_p95 = _required_float(warm_stats, "p95_latency", json_file)
    warm_avg = _required_float(warm_stats, "avg_latency", json_file)

    return CloudColdLatencyRow(
        product_key=product_key,
        product_name=PRODUCT_NAMES.get(product_key, _title_from_key(product_key)),
        mode_key=mode_key,
        mode_display=MODE_NAMES.get(mode_key, _title_from_key(mode_key)),
        status="Rebench needed" if product_key in REBENCH_NEEDED_PRODUCTS else "Accepted",
        filter_rate=_optional_float(custom_case.get("filter_rate")),
        payload_profile=str(custom_case.get("payload_profile") or metrics.get("payload_profile") or ""),
        query_count=int(custom_case.get("query_count") or 0),
        db=str(task_config.get("db") or ""),
        db_label=str((task_config.get("db_config") or {}).get("db_label") or ""),
        payload_bytes_per_query=_optional_int(metrics.get("payload_estimated_bytes_per_query")),
        cold_first_query_seconds=cold_first_query,
        cold_p99_seconds=cold_p99,
        cold_p95_seconds=cold_p95,
        cold_avg_seconds=cold_avg,
        warm_first_query_seconds=warm_first_query,
        warm_p99_seconds=warm_p99,
        warm_p95_seconds=warm_p95,
        warm_avg_seconds=warm_avg,
        first_query_ratio=_ratio(ratios, "first_query_latency_ratio", cold_first_query, warm_first_query),
        p99_ratio=_ratio(ratios, "p99_latency_ratio", cold_p99, warm_p99),
        p95_ratio=_ratio(ratios, "p95_latency_ratio", cold_p95, warm_p95),
        avg_ratio=_ratio(ratios, "avg_latency_ratio", cold_avg, warm_avg),
        load_duration=_optional_float(metrics.get("load_duration")),
        insert_duration=_optional_float(metrics.get("insert_duration")),
        optimize_duration=_optional_float(metrics.get("optimize_duration")),
        raw_path=str(json_file),
    )


def _parse_cloud_cold_latency_path(root: Path, json_file: Path) -> tuple[str, str]:
    try:
        relative = json_file.relative_to(root)
    except ValueError as exc:
        raise CloudColdLatencyParseError(f"{json_file} is not under {root}") from exc

    parts = relative.parts
    if len(parts) < 3:
        raise CloudColdLatencyParseError(f"{json_file} must follow <product>/<mode>/result_*.json")
    product_key, mode_key = parts[:2]
    return product_key, mode_key


def _single_result(data: dict[str, Any], json_file: Path) -> dict[str, Any]:
    results = data.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise CloudColdLatencyParseError(f"{json_file} must contain exactly one result")
    return results[0]


def _required_float(data: dict[str, Any], key: str, json_file: Path) -> float:
    value = data.get(key)
    if value is None:
        raise CloudColdLatencyParseError(f"{json_file} missing required latency field {key}")
    return float(value)


def _ratio(ratios: dict[str, Any], key: str, cold_value: float, warm_value: float) -> float | None:
    if ratios.get(key) is not None:
        return float(ratios[key])
    if warm_value <= 0:
        return None
    return round(cold_value / warm_value, 4)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _payload_display(payload_profile: str) -> str:
    return {
        "ids_only": "IDs Only",
        "scalar_label": "Scalar Label",
        "vector": "Vector",
    }.get(payload_profile, _title_from_key(payload_profile))


def _mode_sort_key(mode_key: str) -> tuple[int, str]:
    if mode_key == "unfiltered":
        return (0, mode_key)
    return (1, mode_key)


def _title_from_key(key: str) -> str:
    return " ".join(part.capitalize() for part in key.split("_"))
