from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CLOUD_PAYLOAD_SEARCH_CASE_ID = 500

PRODUCT_NAMES = {
    "pinecone_serverless": "Pinecone Serverless",
    "turbopuffer_pinned": "Turbopuffer Pinned",
    "turbopuffer_unpinned": "Turbopuffer Unpinned",
    "zilliz_cloud_capacity_12cu": "Zilliz Cloud Capacity 12CU",
    "zilliz_cloud_tiered_4cu": "Zilliz Cloud Tiered 4CU",
}


class PayloadSearchParseError(ValueError):
    pass


@dataclass(frozen=True)
class PayloadSearchRunRow:
    product_key: str
    product_name: str
    search_mode: str
    filter_key: str
    filter_display: str
    filter_value: float | None
    payload_profile: str
    phase: str
    db: str
    db_label: str
    qps: float
    recall: float
    ndcg: float
    serial_latency_p95: float
    serial_latency_p99: float
    concurrency: list[int]
    concurrency_qps: list[float]
    concurrency_latency_avg: list[float]
    concurrency_latency_p95: list[float]
    concurrency_latency_p99: list[float]
    best_concurrency: int | None
    best_latency_avg: float | None
    best_latency_p95: float | None
    best_latency_p99: float | None
    payload_bytes_per_query: int | None
    raw_path: str


@dataclass(frozen=True)
class PayloadSearchDisplayRow:
    product_key: str
    product_name: str
    search_mode: str
    filter_key: str
    filter_display: str
    filter_value: float | None
    payload_profile: str
    db: str
    db_label: str
    qps: float
    recall: float
    ndcg: float
    serial_latency_p95: float
    serial_latency_p99: float
    concurrency: list[int]
    concurrency_qps: list[float]
    concurrency_latency_avg: list[float]
    concurrency_latency_p95: list[float]
    concurrency_latency_p99: list[float]
    best_concurrency: int | None
    best_latency_avg: float | None
    best_latency_p95: float | None
    best_latency_p99: float | None
    payload_bytes_per_query: int | None
    serial_raw_path: str | None
    concurrent_raw_path: str | None
    has_serial_recall: bool
    has_concurrent_qps: bool


def load_payload_search_rows(raw_results_dir: Path | str) -> list[PayloadSearchRunRow]:
    root = Path(raw_results_dir)
    if not root.exists():
        return []

    rows = [_load_payload_search_file(root, json_file) for json_file in sorted(root.rglob("result_*.json"))]
    return rows


def aggregate_payload_search_rows(rows: list[PayloadSearchRunRow]) -> list[PayloadSearchDisplayRow]:
    grouped: dict[tuple[str, str, str, str], list[PayloadSearchRunRow]] = {}
    for row in rows:
        key = (row.product_key, row.search_mode, row.filter_key, row.payload_profile)
        grouped.setdefault(key, []).append(row)

    display_rows = []
    for group_rows in grouped.values():
        serial_row = _best_serial_row(group_rows)
        concurrent_row = _best_concurrent_row(group_rows)
        base_row = concurrent_row or serial_row or group_rows[0]
        quality_row = serial_row or base_row
        throughput_row = concurrent_row or base_row

        display_rows.append(
            PayloadSearchDisplayRow(
                product_key=base_row.product_key,
                product_name=base_row.product_name,
                search_mode=base_row.search_mode,
                filter_key=base_row.filter_key,
                filter_display=base_row.filter_display,
                filter_value=base_row.filter_value,
                payload_profile=base_row.payload_profile,
                db=base_row.db,
                db_label=base_row.db_label,
                qps=throughput_row.qps,
                recall=quality_row.recall,
                ndcg=quality_row.ndcg,
                serial_latency_p95=quality_row.serial_latency_p95,
                serial_latency_p99=quality_row.serial_latency_p99,
                concurrency=throughput_row.concurrency,
                concurrency_qps=throughput_row.concurrency_qps,
                concurrency_latency_avg=throughput_row.concurrency_latency_avg,
                concurrency_latency_p95=throughput_row.concurrency_latency_p95,
                concurrency_latency_p99=throughput_row.concurrency_latency_p99,
                best_concurrency=throughput_row.best_concurrency,
                best_latency_avg=throughput_row.best_latency_avg,
                best_latency_p95=throughput_row.best_latency_p95,
                best_latency_p99=throughput_row.best_latency_p99,
                payload_bytes_per_query=throughput_row.payload_bytes_per_query
                or quality_row.payload_bytes_per_query,
                serial_raw_path=serial_row.raw_path if serial_row else None,
                concurrent_raw_path=concurrent_row.raw_path if concurrent_row else None,
                has_serial_recall=serial_row is not None,
                has_concurrent_qps=concurrent_row is not None,
            )
        )

    display_rows.sort(
        key=lambda row: (
            row.search_mode,
            row.filter_value if row.filter_value is not None else -1,
            row.payload_profile,
            row.product_name,
        )
    )
    return display_rows


def payload_search_records(rows: list[PayloadSearchDisplayRow]) -> list[dict[str, Any]]:
    return [
        {
            "Product": row.product_name,
            "Search Mode": _search_mode_display(row.search_mode),
            "Filter": row.filter_display,
            "Payload": _payload_display(row.payload_profile),
            "Max QPS": row.qps,
            "Best Concurrency": row.best_concurrency,
            "Recall": row.recall,
            "NDCG": row.ndcg,
            "P95 Latency (s)": row.best_latency_p95,
            "P99 Latency (s)": row.best_latency_p99,
            "Payload Bytes/Query": row.payload_bytes_per_query,
            "Serial Recall": "available" if row.has_serial_recall else "missing",
            "Concurrent QPS": "available" if row.has_concurrent_qps else "missing",
        }
        for row in rows
    ]


def _load_payload_search_file(root: Path, json_file: Path) -> PayloadSearchRunRow:
    product_key, search_mode, filter_key, payload_profile, phase = _parse_payload_search_path(root, json_file)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    result = _single_result(data, json_file)
    task_config = result.get("task_config", {})
    case_config = task_config.get("case_config", {})
    case_id = case_config.get("case_id")
    if case_id != CLOUD_PAYLOAD_SEARCH_CASE_ID:
        raise PayloadSearchParseError(
            f"{json_file} has case_id={case_id}; expected {CLOUD_PAYLOAD_SEARCH_CASE_ID}"
        )

    custom_case = case_config.get("custom_case") or {}
    payload_from_json = custom_case.get("payload_profile")
    if payload_from_json and payload_from_json != payload_profile:
        raise PayloadSearchParseError(
            f"{json_file} path payload_profile={payload_profile} differs from JSON payload_profile={payload_from_json}"
        )

    metrics = result.get("metrics") or {}
    concurrency = _as_int_list(metrics.get("conc_num_list", []))
    concurrency_qps = _as_float_list(metrics.get("conc_qps_list", []))
    best_index = _best_qps_index(concurrency_qps)
    filter_value = _filter_value(search_mode, custom_case)

    return PayloadSearchRunRow(
        product_key=product_key,
        product_name=PRODUCT_NAMES.get(product_key, _title_from_key(product_key)),
        search_mode=search_mode,
        filter_key=filter_key,
        filter_display=_filter_display(search_mode, filter_value),
        filter_value=filter_value,
        payload_profile=payload_profile,
        phase=phase,
        db=str(task_config.get("db") or ""),
        db_label=str((task_config.get("db_config") or {}).get("db_label") or ""),
        qps=float(metrics.get("qps") or (max(concurrency_qps) if concurrency_qps else 0)),
        recall=float(metrics.get("recall") or 0),
        ndcg=float(metrics.get("ndcg") or 0),
        serial_latency_p95=float(metrics.get("serial_latency_p95") or 0),
        serial_latency_p99=float(metrics.get("serial_latency_p99") or 0),
        concurrency=concurrency,
        concurrency_qps=concurrency_qps,
        concurrency_latency_avg=_as_float_list(metrics.get("conc_latency_avg_list", [])),
        concurrency_latency_p95=_as_float_list(metrics.get("conc_latency_p95_list", [])),
        concurrency_latency_p99=_as_float_list(metrics.get("conc_latency_p99_list", [])),
        best_concurrency=_value_at(concurrency, best_index),
        best_latency_avg=_value_at(_as_float_list(metrics.get("conc_latency_avg_list", [])), best_index),
        best_latency_p95=_value_at(_as_float_list(metrics.get("conc_latency_p95_list", [])), best_index),
        best_latency_p99=_value_at(_as_float_list(metrics.get("conc_latency_p99_list", [])), best_index),
        payload_bytes_per_query=_optional_int(metrics.get("payload_estimated_bytes_per_query")),
        raw_path=str(json_file),
    )


def _parse_payload_search_path(root: Path, json_file: Path) -> tuple[str, str, str, str, str]:
    try:
        relative = json_file.relative_to(root)
    except ValueError as exc:
        raise PayloadSearchParseError(f"{json_file} is not under {root}") from exc

    parts = relative.parts
    if len(parts) < 6:
        raise PayloadSearchParseError(
            f"{json_file} must follow <product>/<search_mode>/<rate>/<payload_profile>/<phase>/result_*.json"
        )
    product_key, search_mode, filter_key, payload_profile, phase = parts[:5]
    return product_key, search_mode, filter_key, payload_profile, phase


def _single_result(data: dict[str, Any], json_file: Path) -> dict[str, Any]:
    results = data.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise PayloadSearchParseError(f"{json_file} must contain exactly one result")
    return results[0]


def _filter_value(search_mode: str, custom_case: dict[str, Any]) -> float | None:
    if search_mode == "unfiltered":
        return None
    if search_mode == "scalar_label_filter":
        value = custom_case.get("label_percentage")
        return float(value) if value is not None else None
    value = custom_case.get("filter_rate")
    return float(value) if value is not None else None


def _filter_display(search_mode: str, filter_value: float | None) -> str:
    if search_mode == "unfiltered":
        return "Unfiltered"
    if filter_value is None:
        return "Unknown"
    return f"{filter_value * 100:g}%"


def _search_mode_display(search_mode: str) -> str:
    return {
        "unfiltered": "Unfiltered",
        "int_filter": "Integer Filter",
        "scalar_label_filter": "Scalar Label Filter",
    }.get(search_mode, _title_from_key(search_mode))


def _payload_display(payload_profile: str) -> str:
    return {
        "ids_only": "IDs Only",
        "scalar_label": "Scalar Label",
        "vector": "Vector",
    }.get(payload_profile, _title_from_key(payload_profile))


def _best_serial_row(rows: list[PayloadSearchRunRow]) -> PayloadSearchRunRow | None:
    serial_rows = [row for row in rows if row.phase == "serial_recall"]
    if not serial_rows:
        return None
    return max(serial_rows, key=lambda row: (row.recall, row.ndcg))


def _best_concurrent_row(rows: list[PayloadSearchRunRow]) -> PayloadSearchRunRow | None:
    concurrent_rows = [row for row in rows if row.phase == "concurrent_qps"]
    if not concurrent_rows:
        return None
    return max(concurrent_rows, key=lambda row: row.qps)


def _best_qps_index(values: list[float]) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=lambda index: values[index])


def _value_at(values: list[Any], index: int | None) -> Any | None:
    if index is None or index >= len(values):
        return None
    return values[index]


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _title_from_key(key: str) -> str:
    return " ".join(part.capitalize() for part in key.split("_"))
