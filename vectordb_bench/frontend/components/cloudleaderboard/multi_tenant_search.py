from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CLOUD_MULTI_TENANT_SEARCH_CASE_ID = 800

PRODUCT_NAMES = {
    "pinecone_serverless": "Pinecone Serverless",
    "turbopuffer": "Turbopuffer",
    "zilliz_cloud_capacity_2cu": "Zilliz Cloud Capacity 2CU",
    "zilliz_cloud_tiered_1cu": "Zilliz Cloud Tiered 1CU",
}


class MultiTenantSearchParseError(ValueError):
    pass


@dataclass(frozen=True)
class MultiTenantSearchRow:
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
    dataset: str
    tenant_count: int
    tenant_prefix: str
    tenant_id_width: int
    tenant_id_format: str
    top_k: int | None
    qps: float
    concurrency: list[int]
    concurrency_signature: str
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


def load_multi_tenant_search_rows(raw_results_dir: Path | str) -> list[MultiTenantSearchRow]:
    root = Path(raw_results_dir)
    if not root.exists():
        return []

    rows = [_load_multi_tenant_search_file(root, json_file) for json_file in sorted(root.rglob("result_*.json"))]
    rows.sort(
        key=lambda row: (
            row.concurrency_signature,
            row.search_mode,
            row.filter_value if row.filter_value is not None else -1,
            row.payload_profile,
            row.product_name,
        )
    )
    return rows


def multi_tenant_search_records(rows: list[MultiTenantSearchRow]) -> list[dict[str, Any]]:
    return [
        {
            "Product": row.product_name,
            "Dataset": row.dataset,
            "Search Mode": _search_mode_display(row.search_mode),
            "Filter": row.filter_display,
            "Payload": _payload_display(row.payload_profile),
            "Tenant Count": row.tenant_count,
            "Tenant IDs": row.tenant_id_format,
            "Top K": row.top_k,
            "Concurrency": row.concurrency_signature,
            "Max QPS": row.qps,
            "Best Concurrency": row.best_concurrency,
            "P95 Latency (s)": row.best_latency_p95,
            "P99 Latency (s)": row.best_latency_p99,
            "Payload Bytes/Query": row.payload_bytes_per_query,
        }
        for row in rows
    ]


def _load_multi_tenant_search_file(root: Path, json_file: Path) -> MultiTenantSearchRow:
    product_key, search_mode, filter_key, payload_profile, phase = _parse_multi_tenant_search_path(root, json_file)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    result = _single_result(data, json_file)
    task_config = result.get("task_config", {})
    case_config = task_config.get("case_config", {})
    case_id = case_config.get("case_id")
    if case_id != CLOUD_MULTI_TENANT_SEARCH_CASE_ID:
        raise MultiTenantSearchParseError(
            f"{json_file} has case_id={case_id}; expected {CLOUD_MULTI_TENANT_SEARCH_CASE_ID}"
        )

    custom_case = case_config.get("custom_case") or {}
    payload_from_json = custom_case.get("payload_profile")
    if payload_from_json and payload_from_json != payload_profile:
        raise MultiTenantSearchParseError(
            f"{json_file} path payload_profile={payload_profile} differs from JSON payload_profile={payload_from_json}"
        )

    metrics = result.get("metrics") or {}
    concurrency = _as_int_list(metrics.get("conc_num_list", []))
    concurrency_qps = _as_float_list(metrics.get("conc_qps_list", []))
    best_index = _best_qps_index(concurrency_qps)
    tenant_count = int(custom_case.get("tenant_count") or 0)
    tenant_prefix = str(custom_case.get("tenant_prefix") or "")
    tenant_id_width = int(custom_case.get("tenant_id_width") or 0)
    filter_value = _filter_value(search_mode, custom_case)

    return MultiTenantSearchRow(
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
        dataset=str(custom_case.get("dataset_with_size_type") or ""),
        tenant_count=tenant_count,
        tenant_prefix=tenant_prefix,
        tenant_id_width=tenant_id_width,
        tenant_id_format=_tenant_id_format(tenant_prefix, tenant_count, tenant_id_width),
        top_k=_optional_int(case_config.get("k")),
        qps=float(metrics.get("qps") or (max(concurrency_qps) if concurrency_qps else 0)),
        concurrency=concurrency,
        concurrency_signature=_concurrency_signature(concurrency),
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


def _parse_multi_tenant_search_path(root: Path, json_file: Path) -> tuple[str, str, str, str, str]:
    try:
        relative = json_file.relative_to(root)
    except ValueError as exc:
        raise MultiTenantSearchParseError(f"{json_file} is not under {root}") from exc

    parts = relative.parts
    if len(parts) < 6:
        raise MultiTenantSearchParseError(
            f"{json_file} must follow <product>/<search_mode>/<rate>/<payload_profile>/<phase>/result_*.json"
        )
    product_key, search_mode, filter_key, payload_profile, phase = parts[:5]
    return product_key, search_mode, filter_key, payload_profile, phase


def _single_result(data: dict[str, Any], json_file: Path) -> dict[str, Any]:
    results = data.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise MultiTenantSearchParseError(f"{json_file} must contain exactly one result")
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


def _tenant_id_format(prefix: str, tenant_count: int, width: int) -> str:
    if tenant_count <= 0 or width <= 0:
        return ""
    return f"{prefix}{0:0{width}d}..{prefix}{tenant_count - 1:0{width}d}"


def _concurrency_signature(concurrency: list[int]) -> str:
    if not concurrency:
        return "unknown"
    return ",".join(f"c{value}" for value in concurrency)


def _title_from_key(key: str) -> str:
    return " ".join(part.capitalize() for part in key.split("_"))
