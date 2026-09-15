from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, NoReturn

FieldDataType = Literal["int", "float", "bool", "string", "string[]", "vector"]
DistanceMetric = Literal["cosine"]
SearchMode = Literal["dense", "bm25"]


@dataclass(frozen=True)
class FieldSchema:
    data_type: FieldDataType
    dimensions: int | None = None
    metric: DistanceMetric | None = None
    full_text_search: bool = False
    filterable: bool | None = None
    nullable: bool = True

    def __post_init__(self) -> None:
        if self.data_type not in {"int", "float", "bool", "string", "string[]", "vector"}:
            msg = f"unsupported customized field type: {self.data_type}"
            raise ValueError(msg)
        if self.data_type == "vector":
            if self.dimensions is None or self.dimensions <= 0:
                raise ValueError("vector fields require positive dimensions")
            if self.metric not in {None, "cosine"}:
                msg = f"unsupported customized vector metric: {self.metric}"
                raise ValueError(msg)
        elif self.dimensions is not None or self.metric is not None:
            raise ValueError("dimensions and metric apply only to vector fields")
        if self.full_text_search and self.data_type not in {"string", "string[]"}:
            raise ValueError("full_text_search requires a string field")


@dataclass(frozen=True)
class CustomizedRow:
    id: str | int
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.id is None or self.id == "":
            raise ValueError("customized row id must not be empty")
        if "id" in self.fields:
            raise ValueError("customized row fields must not contain id")


@dataclass(frozen=True)
class CustomizedRequest:
    mode: SearchMode
    field: str
    value: Sequence[float] | str
    top_k: int = 100
    include_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"dense", "bm25"}:
            msg = f"unsupported customized query mode: {self.mode}"
            raise ValueError(msg)
        if not self.field:
            raise ValueError("customized query field must not be empty")
        if self.top_k <= 0:
            raise ValueError("customized query top_k must be positive")
        if self.mode == "dense" and isinstance(self.value, str):
            raise ValueError("dense customized queries require a vector value")
        if self.mode == "bm25" and not isinstance(self.value, str):
            raise ValueError("bm25 customized queries require a string value")


@dataclass(frozen=True)
class SearchPerformance:
    cache_hit_ratio: float | None = None
    cache_temperature: str | None = None
    server_total_ms: float | None = None
    query_execution_ms: float | None = None


@dataclass(frozen=True)
class SearchResult:
    ids: list[str | int]
    fields: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    performance: SearchPerformance = field(default_factory=SearchPerformance)


def validate_customized_rows(
    rows: Sequence[CustomizedRow],
    schema: Mapping[str, FieldSchema],
) -> None:
    if "id" in schema:
        raise ValueError("customized schema must not contain id")
    expected_fields = set(schema)
    for row in rows:
        actual_fields = set(row.fields)
        if actual_fields != expected_fields:
            msg = (
                f"customized row fields do not match schema: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
            raise ValueError(msg)
        for name, spec in schema.items():
            _validate_value(name, row.fields[name], spec)


def _invalid_field(name: str, requirement: str) -> NoReturn:
    msg = f"customized field {name} requires {requirement}"
    raise ValueError(msg)


def _validate_value(name: str, value: Any, spec: FieldSchema) -> None:
    if value is None:
        if not spec.nullable:
            _invalid_field(name, "a non-null value")
        return
    if spec.data_type == "int" and (not isinstance(value, int) or isinstance(value, bool)):
        _invalid_field(name, "int values")
    if spec.data_type == "float" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        _invalid_field(name, "float values")
    if spec.data_type == "float" and not isfinite(value):
        _invalid_field(name, "finite values")
    if spec.data_type == "bool" and not isinstance(value, bool):
        _invalid_field(name, "bool values")
    if spec.data_type == "string" and not isinstance(value, str):
        _invalid_field(name, "string values")
    if spec.data_type == "string[]" and (
        not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value)
    ):
        _invalid_field(name, "string-array values")
    if spec.data_type == "vector":
        try:
            vector = value.tolist() if hasattr(value, "tolist") else list(value)
        except TypeError as e:
            msg = f"customized field {name} requires vector values"
            raise ValueError(msg) from e
        if len(vector) != spec.dimensions or any(
            not isinstance(item, (int, float)) or isinstance(item, bool) or not isfinite(item) for item in vector
        ):
            _invalid_field(name, f"finite {spec.dimensions}d vectors")
