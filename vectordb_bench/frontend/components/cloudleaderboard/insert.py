from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CLOUD_INSERT_CASE_ID = 600

PRODUCT_NAMES = {
    "pinecone_serverless": "Pinecone Serverless",
    "turbopuffer": "Turbopuffer",
    "zilliz_cloud_tiered_4cu": "Zilliz Cloud Tiered 4CU",
    "zilliz_cloud_capacity_12cu": "Zilliz Cloud Capacity 12CU",
    "zillz_cloud_cap_12cu": "Zilliz Cloud Capacity 12CU",
}

MODE_NAMES = {
    "default": "Default",
    "bp_off": "Backpressure Off",
    "bp_on": "Backpressure On",
}


class CloudInsertParseError(ValueError):
    pass


@dataclass(frozen=True)
class CloudInsertRow:
    product_key: str
    product_name: str
    mode_key: str
    mode_display: str
    batch_key: str
    batch_size: int
    dataset: str
    duration: float | None
    load_concurrency: int
    db: str
    db_label: str
    inserted_count: int
    insert_rows_per_second: float
    insert_completion_seconds: float
    searchable_after_insert_seconds: float
    indexed_after_searchable_seconds: float
    total_readiness_seconds: float
    raw_path: str


def load_cloud_insert_rows(raw_results_dir: Path | str) -> list[CloudInsertRow]:
    root = Path(raw_results_dir)
    if not root.exists():
        return []

    rows = [_load_cloud_insert_file(root, json_file) for json_file in sorted(root.rglob("result_*.json"))]
    rows.sort(key=lambda row: (row.product_name, row.mode_display, row.batch_size))
    return rows


def cloud_insert_records(rows: list[CloudInsertRow]) -> list[dict[str, Any]]:
    return [
        {
            "Product": row.product_name,
            "Mode": row.mode_display,
            "Dataset": row.dataset,
            "Batch Size": row.batch_size,
            "Load Concurrency": row.load_concurrency,
            "Inserted Count": row.inserted_count,
            "Insert Rows/s": row.insert_rows_per_second,
            "Insert Completion (s)": row.insert_completion_seconds,
            "Searchable Delay (s)": row.searchable_after_insert_seconds,
            "Indexed Delay (s)": row.indexed_after_searchable_seconds,
            "Total Readiness (s)": row.total_readiness_seconds,
        }
        for row in rows
    ]


def _load_cloud_insert_file(root: Path, json_file: Path) -> CloudInsertRow:
    product_key, mode_key, batch_key = _parse_cloud_insert_path(root, json_file)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    result = _single_result(data, json_file)
    task_config = result.get("task_config", {})
    case_config = task_config.get("case_config", {})
    case_id = case_config.get("case_id")
    if case_id != CLOUD_INSERT_CASE_ID:
        raise CloudInsertParseError(f"{json_file} has case_id={case_id}; expected {CLOUD_INSERT_CASE_ID}")

    custom_case = case_config.get("custom_case") or {}
    batch_size = _optional_int(custom_case.get("batch_size")) or _batch_size_from_key(batch_key)
    metrics = result.get("metrics") or {}
    insert_completion_seconds = float(metrics.get("insert_completion_seconds") or 0)
    searchable_after_insert_seconds = float(metrics.get("searchable_after_insert_seconds") or 0)
    indexed_after_searchable_seconds = float(metrics.get("indexed_after_searchable_seconds") or 0)
    total_readiness_seconds = round(
        insert_completion_seconds + searchable_after_insert_seconds + indexed_after_searchable_seconds,
        4,
    )

    return CloudInsertRow(
        product_key=product_key,
        product_name=PRODUCT_NAMES.get(product_key, _title_from_key(product_key)),
        mode_key=mode_key,
        mode_display=MODE_NAMES.get(mode_key, _title_from_key(mode_key)),
        batch_key=batch_key,
        batch_size=batch_size,
        dataset=str(custom_case.get("dataset_with_size_type") or ""),
        duration=_optional_float(custom_case.get("duration")),
        load_concurrency=int(task_config.get("load_concurrency") or 0),
        db=str(task_config.get("db") or ""),
        db_label=str((task_config.get("db_config") or {}).get("db_label") or ""),
        inserted_count=int(metrics.get("inserted_count") or 0),
        insert_rows_per_second=float(metrics.get("insert_rows_per_second") or 0),
        insert_completion_seconds=insert_completion_seconds,
        searchable_after_insert_seconds=searchable_after_insert_seconds,
        indexed_after_searchable_seconds=indexed_after_searchable_seconds,
        total_readiness_seconds=total_readiness_seconds,
        raw_path=str(json_file),
    )


def _parse_cloud_insert_path(root: Path, json_file: Path) -> tuple[str, str, str]:
    try:
        relative = json_file.relative_to(root)
    except ValueError as exc:
        raise CloudInsertParseError(f"{json_file} is not under {root}") from exc

    parts = relative.parts
    if len(parts) < 3:
        raise CloudInsertParseError(f"{json_file} must follow <product>/<batch>/result_*.json")

    product_key = parts[0]
    if parts[1].startswith("batch_"):
        mode_key = "default"
        batch_key = parts[1]
    elif len(parts) >= 4 and parts[2].startswith("batch_"):
        mode_key = parts[1]
        batch_key = parts[2]
    else:
        raise CloudInsertParseError(
            f"{json_file} must follow <product>/<batch>/result_*.json "
            "or <product>/<mode>/<batch>/result_*.json"
        )
    return product_key, mode_key, batch_key


def _single_result(data: dict[str, Any], json_file: Path) -> dict[str, Any]:
    results = data.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise CloudInsertParseError(f"{json_file} must contain exactly one result")
    return results[0]


def _batch_size_from_key(batch_key: str) -> int:
    if not batch_key.startswith("batch_"):
        raise CloudInsertParseError(f"Unknown batch key: {batch_key}")
    raw_value = batch_key.removeprefix("batch_")
    if raw_value.endswith("k"):
        return int(raw_value[:-1]) * 1000
    return int(raw_value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _title_from_key(key: str) -> str:
    return " ".join(part.capitalize() for part in key.split("_"))
