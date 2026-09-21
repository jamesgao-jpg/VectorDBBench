import hashlib
import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vectordb_bench import config
from vectordb_bench.backend.clients.api import VectorDB
from vectordb_bench.backend.customized import CustomizedRequest, CustomizedRow, FieldSchema

log = logging.getLogger(__name__)

MANIFEST_VERSION = 7
SEARCH_RESULT_VERSION = 1
SEARCH_ORDER_SEED = 20260914
QUERIES_FILE_VERSION = 1
DEFAULT_NAMESPACE_ROWS = 15_000
NAMESPACE_ID_WIDTH = 4

SOURCE_FIELDS = (
    pa.field("emb_768", pa.list_(pa.float32()), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("i32_region", pa.int32(), nullable=False),
    pa.field("f64_price", pa.float64(), nullable=False),
    pa.field("bool_active", pa.bool_(), nullable=False),
    pa.field("vc_uuid", pa.string(), nullable=False),
    pa.field("vc_tag", pa.string()),
    pa.field("vc_desc", pa.string(), nullable=False),
    pa.field("bluesky_json", pa.string(), nullable=False),
    pa.field("arr_str_labels", pa.list_(pa.string()), nullable=False),
    pa.field("$meta", pa.string(), nullable=False),
)
SOURCE_COLUMNS = tuple(field.name for field in SOURCE_FIELDS)
PREPARED_SCHEMA = pa.schema(
    [pa.field("id", pa.int64(), nullable=False)]
    + [
        pa.field("meta_json" if field.name == "$meta" else field.name, field.type, nullable=field.nullable)
        for field in SOURCE_FIELDS
    ]
)
CUSTOMIZED_SCHEMA = {
    "emb_768": FieldSchema("vector", dimensions=768, metric="cosine", nullable=False),
    "content": FieldSchema("string", full_text_search=True, nullable=False),
    "i32_region": FieldSchema("int", nullable=False),
    "f64_price": FieldSchema("float", nullable=False),
    "bool_active": FieldSchema("bool", nullable=False),
    "vc_uuid": FieldSchema("string", nullable=False),
    "vc_tag": FieldSchema("string"),
    "vc_desc": FieldSchema("string", nullable=False),
    "bluesky_json": FieldSchema("string", filterable=False, nullable=False),
    "arr_str_labels": FieldSchema("string[]", nullable=False),
    "meta_json": FieldSchema("string", filterable=False, nullable=False),
}


def _setup_checkpoint_states(path: Path, valid_namespaces: set[str]) -> dict[str, str]:
    states = {}
    if not path.exists():
        return states
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid setup checkpoint at line {line_number}") from error
            namespace = event.get("namespace")
            state = event.get("state")
            if namespace not in valid_namespaces or state not in {"started", "completed"}:
                raise ValueError(f"invalid setup checkpoint at line {line_number}")
            states[namespace] = state
    return states


def customized_schema(dense_field: str, bm25_field: str) -> dict[str, FieldSchema]:
    if dense_field not in CUSTOMIZED_SCHEMA or CUSTOMIZED_SCHEMA[dense_field].data_type != "vector":
        raise ValueError(f"dense search field must be a declared vector field: {dense_field}")
    if bm25_field not in CUSTOMIZED_SCHEMA or CUSTOMIZED_SCHEMA[bm25_field].data_type != "string":
        raise ValueError(f"BM25 search field must be a declared string field: {bm25_field}")
    return {
        name: replace(field, full_text_search=name == bm25_field)
        if field.data_type in {"string", "string[]"}
        else field
        for name, field in CUSTOMIZED_SCHEMA.items()
    }


@dataclass(frozen=True)
class NamespaceGroup:
    suffix: str
    rows_per_namespace: int
    namespace_count: int
    id_width: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.suffix):
            raise ValueError("namespace group suffix must contain only letters, numbers, '_' or '-'")
        if min(self.rows_per_namespace, self.namespace_count, self.id_width) <= 0:
            raise ValueError("namespace group sizes and id_width must be positive")

    @property
    def source_rows(self) -> int:
        return self.rows_per_namespace * self.namespace_count


def namespace_group(rows_per_namespace: int = DEFAULT_NAMESPACE_ROWS) -> NamespaceGroup:
    if rows_per_namespace <= 0:
        raise ValueError("rows_per_namespace must be positive")
    return NamespaceGroup(str(rows_per_namespace), rows_per_namespace, 1, NAMESPACE_ID_WIDTH)


DEFAULT_NAMESPACE_GROUP = namespace_group()


@dataclass(frozen=True)
class SearchQuery:
    """One shared out-of-sample query used against every namespace."""

    index: int
    dense: tuple[float, ...]
    bm25: str


def load_queries_file(path: Path) -> list[SearchQuery]:
    with path.open() as source:
        queries = json.load(source)
    if queries.get("version") != QUERIES_FILE_VERSION:
        raise ValueError(f"queries file must have version {QUERIES_FILE_VERSION}")
    entries = queries.get("queries", [])
    count = queries.get("count")
    if not isinstance(count, int) or count != len(entries) or count <= 0:
        raise ValueError("queries file must declare a positive count matching its entries")
    loaded = []
    for entry in entries:
        index = entry.get("index")
        dense = entry.get("dense")
        bm25 = entry.get("bm25")
        if (
            not isinstance(index, int)
            or not isinstance(dense, list)
            or len(dense) != 768
            or not all(isinstance(value, (int, float)) for value in dense)
            or not isinstance(bm25, str)
            or not bm25.strip()
        ):
            raise ValueError("queries file contains an invalid query entry")
        loaded.append(SearchQuery(index=index, dense=tuple(float(value) for value in dense), bm25=bm25))
    if sorted(query.index for query in loaded) != list(range(len(loaded))):
        raise ValueError("queries file indices must be 0..count-1")
    return loaded


@dataclass(frozen=True)
class NamespaceSpec:
    name: str
    rows: int
    source_start: int
    source_end: int


@dataclass(frozen=True)
class NamespaceBatch:
    namespace: NamespaceSpec
    record_batch: pa.RecordBatch
    first: bool
    last: bool

    def customized_rows(self, schema: dict[str, FieldSchema]) -> list[CustomizedRow]:
        columns = self.record_batch.to_pydict()
        return [
            CustomizedRow(
                id=columns["id"][index],
                fields={name: columns[name][index] for name in schema},
            )
            for index in range(self.record_batch.num_rows)
        ]


class PreparedMultiTenantDataset:
    def __init__(self, path: Path, group: NamespaceGroup = DEFAULT_NAMESPACE_GROUP):
        self.path = path.resolve()
        self.group = group
        if not self.path.is_file():
            raise FileNotFoundError(f"prepared Parquet does not exist: {self.path}")
        parquet_file = pq.ParquetFile(self.path, memory_map=True, pre_buffer=False)
        if parquet_file.schema_arrow != PREPARED_SCHEMA:
            raise ValueError("prepared Parquet schema does not match the turbopuffer multi-tenant contract")
        self.row_count = parquet_file.metadata.num_rows
        if self.row_count < group.source_rows:
            raise ValueError(
                f"prepared Parquet has {self.row_count} rows, but the setup requires {group.source_rows}"
            )

    @staticmethod
    def namespace_name(run_prefix: str, group: NamespaceGroup, ordinal: int) -> str:
        return f"{run_prefix}_{group.suffix}_{ordinal:0{group.id_width}d}"

    def namespace_specs(self, run_prefix: str) -> list[NamespaceSpec]:
        group = self.group
        return [
            NamespaceSpec(
                name=self.namespace_name(run_prefix, group, ordinal),
                rows=group.rows_per_namespace,
                source_start=(ordinal - 1) * group.rows_per_namespace,
                source_end=ordinal * group.rows_per_namespace,
            )
            for ordinal in range(1, group.namespace_count + 1)
        ]

    def iter_batches(
        self,
        run_prefix: str,
        batch_size: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        group = self.group
        specs = self.namespace_specs(run_prefix)
        parquet_file = pq.ParquetFile(self.path, memory_map=True, pre_buffer=True)
        consumed = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=PREPARED_SCHEMA.names):
            if consumed >= group.source_rows:
                break
            if batch.num_rows > group.source_rows - consumed:
                batch = batch.slice(0, group.source_rows - consumed)
            offset = 0
            while offset < batch.num_rows:
                namespace_index = consumed // group.rows_per_namespace
                rows_in_namespace = consumed % group.rows_per_namespace
                take = min(batch.num_rows - offset, group.rows_per_namespace - rows_in_namespace)
                yield NamespaceBatch(
                    namespace=specs[namespace_index],
                    record_batch=batch.slice(offset, take),
                    first=rows_in_namespace == 0,
                    last=rows_in_namespace + take == group.rows_per_namespace,
                )
                consumed += take
                offset += take
        if consumed != group.source_rows:
            raise ValueError(f"read {consumed} rows, expected {group.source_rows}")


class MultiTenantSetupRunner:
    def __init__(
        self,
        db: VectorDB,
        dataset: PreparedMultiTenantDataset,
        manifest_path: Path,
        run_prefix: str,
        *,
        queries_file: Path | None = None,
        dense_field: str = "emb_768",
        bm25_field: str = "content",
        batch_size: int = config.DEFAULT_INSERT_BATCH_SIZE,
        max_retries: int = config.MAX_INSERT_RETRY,
        retry_delay: float = 1.0,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_prefix):
            raise ValueError("run_prefix must contain only letters, numbers, '_' or '-'")
        if batch_size <= 0 or max_retries < 0 or retry_delay < 0:
            raise ValueError("batch_size must be positive and retry settings must be non-negative")
        if queries_file is None:
            raise ValueError("setup requires a shared queries file")
        self.db = db
        self.dataset = dataset
        self.group = dataset.group
        self.manifest_path = manifest_path.resolve()
        self.run_prefix = run_prefix
        self.dense_field = dense_field
        self.bm25_field = bm25_field
        self.schema = customized_schema(dense_field, bm25_field)
        self.queries_file = queries_file.resolve()
        self.queries = load_queries_file(self.queries_file)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.queries_sidecar = self.manifest_path.with_suffix(".queries.json")
        self.checkpoint_path = self.manifest_path.with_suffix(".checkpoints.jsonl")

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial")
        with partial.open("w") as output:
            json.dump(value, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)

    def _specs(self) -> list[NamespaceSpec]:
        return self.dataset.namespace_specs(self.run_prefix)

    def _write_queries_sidecar(self) -> None:
        self._write_json(
            self.queries_sidecar,
            {
                "version": QUERIES_FILE_VERSION,
                "count": len(self.queries),
                "dense_field": self.dense_field,
                "bm25_field": self.bm25_field,
                "queries": [
                    {"index": query.index, "dense": list(query.dense), "bm25": query.bm25}
                    for query in self.queries
                ],
            },
        )

    def _manifest(self) -> dict[str, Any]:
        specs = self._specs()
        search_order = [spec.name for spec in specs]
        random.Random(SEARCH_ORDER_SEED).shuffle(search_order)
        return {
            "version": MANIFEST_VERSION,
            "run_prefix": self.run_prefix,
            "rows_per_namespace": self.group.rows_per_namespace,
            "prepared_data": str(self.dataset.path),
            "prepared_rows": self.dataset.row_count,
            "prepared_size_bytes": self.dataset.path.stat().st_size,
            "field_mapping": {"$meta": "meta_json"},
            "schema": {name: asdict(field) for name, field in self.schema.items()},
            "search_fields": {"dense": self.dense_field, "bm25": self.bm25_field},
            "queries_file": self.queries_sidecar.name,
            "query_count": len(self.queries),
            "namespaces": [asdict(spec) for spec in specs],
            "search_order_seed": SEARCH_ORDER_SEED,
            "search_order": search_order,
            "checkpoint_file": self.checkpoint_path.name,
        }

    def _ensure_manifest(self) -> None:
        expected = self._manifest()
        if self.manifest_path.exists():
            with self.manifest_path.open() as source:
                existing = json.load(source)
            if existing != expected:
                raise ValueError(f"existing setup manifest does not match this run: {self.manifest_path}")
            return
        if self.checkpoint_path.exists():
            raise FileExistsError("setup artifacts exist without a matching manifest")
        self._write_json(self.manifest_path, expected)

    def _append_checkpoint(self, spec: NamespaceSpec, state: str, inserted_rows: int) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "namespace": spec.name,
                        "state": state,
                        "inserted_rows": inserted_rows,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
            os.fsync(output.fileno())

    def _insert(self, namespace: str, rows: list[CustomizedRow]) -> None:
        last_error = None
        for attempt in range(self.max_retries + 1):
            inserted, error = self.db.insert_customized_rows(rows, self.schema)
            if error is None and inserted == len(rows):
                return
            last_error = error or RuntimeError(f"insert returned {inserted} rows, expected {len(rows)}")
            if attempt < self.max_retries and self.retry_delay:
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"customized insert failed for namespace {namespace}") from last_error

    def run(self) -> dict[str, int | str]:
        if not self.db.supports_customized_api() or not self.db.supports_namespace_selection():
            raise NotImplementedError("multi-tenant setup requires customized rows and namespace selection")
        self._write_queries_sidecar()
        self._ensure_manifest()
        specs = self._specs()
        by_name = {spec.name: spec for spec in specs}
        states = _setup_checkpoint_states(self.checkpoint_path, set(by_name))
        completed = {name for name, state in states.items() if state == "completed"}
        if len(completed) == len(specs):
            return self._summary(0, completed, len(specs))

        newly_inserted = 0
        with self.db.init():
            active_name = None
            active_count = 0
            for batch in self.dataset.iter_batches(
                self.run_prefix,
                self.batch_size,
            ):
                spec = batch.namespace
                if spec.name in completed:
                    continue
                if batch.first:
                    if states.get(spec.name) is None:
                        if self.db.namespace_exists(spec.name):
                            raise FileExistsError(f"refusing existing namespace without checkpoint: {spec.name}")
                        self._append_checkpoint(spec, "started", 0)
                        states[spec.name] = "started"
                    self.db.select_namespace(spec.name)
                    active_name = spec.name
                    active_count = 0
                if active_name != spec.name:
                    raise RuntimeError(f"non-contiguous batches for namespace {spec.name}")
                rows = batch.customized_rows(self.schema)
                self._insert(spec.name, rows)
                active_count += len(rows)
                if batch.last:
                    if active_count != spec.rows:
                        raise RuntimeError(
                            f"namespace {spec.name} inserted {active_count} rows, expected {spec.rows}"
                        )
                    self._append_checkpoint(spec, "completed", active_count)
                    states[spec.name] = "completed"
                    completed.add(spec.name)
                    newly_inserted += active_count
                    active_name = None
        return self._summary(newly_inserted, completed, len(specs))

    def _summary(self, inserted_rows: int, completed: set[str], total_namespaces: int) -> dict[str, int | str]:
        return {
            "rows_per_namespace": self.group.rows_per_namespace,
            "inserted_rows": inserted_rows,
            "total_rows": self.group.source_rows,
            "completed_namespaces": len(completed),
            "total_namespaces": total_namespaces,
            "queries": len(self.queries),
            "manifest": str(self.manifest_path),
        }


class MultiTenantSearchIncomplete(RuntimeError):
    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        super().__init__(f"multi-tenant search is incomplete; summary={summary['summary_path']}")


class MultiTenantSearchRunner:
    def __init__(
        self,
        db: VectorDB,
        manifest_path: Path,
        mode: str,
        *,
        output_fields: tuple[str, ...] = (),
        top_k: int = 100,
    ):
        if mode not in {"dense", "bm25"}:
            raise ValueError("multi-tenant search mode must be dense or bm25")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(set(output_fields)) != len(output_fields) or "id" in output_fields:
            raise ValueError("output fields must be unique and must not contain id")

        self.db = db
        self.manifest_path = manifest_path.resolve()
        self.mode = mode
        self.output_fields = output_fields
        self.top_k = top_k
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.manifest_hash = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.event_path = self.manifest_path.with_name(f"{self.manifest_path.stem}.{mode}.search.jsonl")
        self.summary_path = self.manifest_path.with_name(f"{self.manifest_path.stem}.{mode}.summary.json")
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        if self.manifest.get("version") != MANIFEST_VERSION:
            raise ValueError(f"setup manifest must have version {MANIFEST_VERSION}")
        rows_per_namespace = self.manifest.get("rows_per_namespace")
        if not isinstance(rows_per_namespace, int) or rows_per_namespace <= 0:
            raise ValueError("setup manifest must declare positive rows_per_namespace")
        self.rows_per_namespace = rows_per_namespace
        schema = self.manifest.get("schema", {})
        search_fields = self.manifest.get("search_fields", {})
        self.search_field = search_fields.get(self.mode)
        field = schema.get(self.search_field, {})
        expected_type = "vector" if self.mode == "dense" else "string"
        if field.get("data_type") != expected_type:
            raise ValueError(f"manifest {self.mode} search field is not a declared {expected_type} field")
        if self.mode == "bm25" and not field.get("full_text_search"):
            raise ValueError("manifest BM25 field is not configured for full-text search")
        unknown_outputs = sorted(set(self.output_fields) - set(schema))
        if unknown_outputs:
            raise ValueError(f"output fields are not declared in the manifest: {unknown_outputs}")

        queries_file = self.manifest.get("queries_file")
        query_count = self.manifest.get("query_count")
        if not isinstance(queries_file, str) or not isinstance(query_count, int) or query_count <= 0:
            raise ValueError("setup manifest must declare its shared queries file")
        queries_path = (self.manifest_path.parent / queries_file).resolve()
        if self.manifest_path.parent not in queries_path.parents:
            raise ValueError(f"queries file escapes the manifest directory: {queries_file}")
        self.queries = load_queries_file(queries_path)
        if len(self.queries) != query_count:
            raise ValueError("manifest query_count does not match its queries file")

        entries = self.manifest.get("namespaces", [])
        self.namespaces = {entry["name"]: entry for entry in entries}
        order = self.manifest.get("search_order", [])
        if len(self.namespaces) != len(entries) or set(order) != set(self.namespaces) or len(order) != len(entries):
            raise ValueError("manifest namespaces and search order do not match")
        self.search_order = list(order)
        if not self.search_order:
            raise ValueError("manifest declares no namespaces to search")

        states = _setup_checkpoint_states(self.checkpoint_path, set(self.namespaces))
        incomplete = sorted(name for name in self.namespaces if states.get(name) != "completed")
        if incomplete:
            raise ValueError(f"setup is incomplete for {len(incomplete)} namespaces")

        other_mode = "bm25" if self.mode == "dense" else "dense"
        other_events = self.manifest_path.with_name(f"{self.manifest_path.stem}.{other_mode}.search.jsonl")
        if other_events.exists() and other_events.stat().st_size:
            raise ValueError("dense and BM25 cold measurements require separate setup manifests")

    @property
    def checkpoint_path(self) -> Path:
        return self.manifest_path.parent / self.manifest["checkpoint_file"]

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as output:
            output.write(json.dumps(value, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def _header(self) -> dict[str, Any]:
        return {
            "event": "header",
            "version": SEARCH_RESULT_VERSION,
            "manifest_sha256": self.manifest_hash,
            "mode": self.mode,
            "search_field": self.search_field,
            "output_fields": list(self.output_fields),
            "top_k": self.top_k,
            "query_count": len(self.queries),
        }

    def _events(self) -> list[dict[str, Any]]:
        if not self.event_path.exists():
            self._append_jsonl(self.event_path, self._header())
        with self.event_path.open() as source:
            events = [json.loads(line) for line in source if line.strip()]
        if not events or events[0] != self._header():
            raise ValueError(f"search checkpoint does not match this run: {self.event_path}")
        for event in events[1:]:
            if (
                event.get("namespace") not in self.namespaces
                or event.get("pass") not in {"first", "repeat"}
                or not isinstance(event.get("query_index"), int)
                or not 0 <= event["query_index"] < len(self.queries)
            ):
                raise ValueError(f"invalid search checkpoint event: {self.event_path}")
        return events

    @staticmethod
    def _states(events: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
        return {(event["namespace"], event["pass"], event["query_index"]): event for event in events[1:]}

    def _append_state(self, namespace: str, pass_name: str, query_index: int, event: str, **values: Any) -> dict[str, Any]:
        value = {"event": event, "namespace": namespace, "pass": pass_name, "query_index": query_index, **values}
        self._append_jsonl(self.event_path, value)
        return value

    def _normalize_interrupted(self, states: dict[tuple[str, str, int], dict[str, Any]]) -> None:
        for namespace in self.search_order:
            for query in self.queries:
                for pass_name in ("first", "repeat"):
                    state = states.get((namespace, pass_name, query.index))
                    if state and state["event"] == "started":
                        states[(namespace, pass_name, query.index)] = self._append_state(
                            namespace, pass_name, query.index, "indeterminate"
                        )

    def _query_value(self, query: SearchQuery) -> Any:
        return list(query.dense) if self.mode == "dense" else query.bm25

    def _query(
        self,
        namespace: str,
        pass_name: str,
        query_index: int,
        value: Any,
        *,
        disable_cache: bool,
    ) -> dict[str, Any]:
        self._append_state(namespace, pass_name, query_index, "started")
        started = time.perf_counter()
        try:
            request = CustomizedRequest(
                self.mode,
                self.search_field,
                value,
                self.top_k,
                self.output_fields,
                disable_cache=disable_cache,
            )
            results = self.db.search_customized_queries([request])
            if len(results) != 1:
                raise ValueError(f"customized search returned {len(results)} results, expected one")
            result = results[0]
            if set(result.fields) != set(self.output_fields):
                raise ValueError("customized search returned unexpected output fields")
            if any(len(values) != len(result.ids) for values in result.fields.values()):
                raise ValueError("customized search output field count does not match result IDs")
            latency_ms = (time.perf_counter() - started) * 1000
            return self._append_state(
                namespace,
                pass_name,
                query_index,
                "completed",
                client_latency_ms=round(latency_ms, 4),
                result_count=len(result.ids),
                performance=asdict(result.performance),
            )
        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000
            log.warning("Multi-tenant %s query failed for namespace=%s", self.mode, namespace)
            return self._append_state(
                namespace,
                pass_name,
                query_index,
                "error",
                client_latency_ms=round(latency_ms, 4),
                error_type=type(error).__name__,
            )

    @staticmethod
    def _latency_stats(values: list[float | None]) -> dict[str, float | int]:
        latencies = [value for value in values if value is not None]
        if not latencies:
            return {"count": 0}
        ordered = sorted(latencies)
        return {
            "count": len(ordered),
            "min_ms": round(float(ordered[0]), 4),
            "max_ms": round(float(ordered[-1]), 4),
            "average_ms": round(float(np.mean(ordered)), 4),
            "p50_ms": round(float(np.percentile(ordered, 50)), 4),
            "p95_ms": round(float(np.percentile(ordered, 95)), 4),
            "p99_ms": round(float(np.percentile(ordered, 99)), 4),
        }

    @staticmethod
    def _cache_temperatures(events: list[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(event["performance"].get("cache_temperature") for event in events)
        counts.pop(None, None)
        return dict(sorted(counts.items()))

    def _summary(self, states: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
        buckets = {
            "first": [
                (name, "first", query.index) for name in self.search_order for query in self.queries
            ],
            "repeat": [
                (name, "repeat", query.index) for name in self.search_order for query in self.queries
            ],
        }
        bucket_summary = {}
        for bucket, keys in buckets.items():
            events = [states[key] for key in keys if key in states]
            completed = [event for event in events if event["event"] == "completed"]
            bucket_summary[bucket] = {
                **self._latency_stats([event["client_latency_ms"] for event in completed]),
                "outcomes": dict(sorted(Counter(event["event"] for event in events).items())),
                "server_total_ms": self._latency_stats(
                    [event["performance"].get("server_total_ms") for event in completed]
                ),
                "query_execution_ms": self._latency_stats(
                    [event["performance"].get("query_execution_ms") for event in completed]
                ),
                "cache_temperature": self._cache_temperatures(completed),
                "cache_hit_ratio": self._latency_stats(
                    [event["performance"].get("cache_hit_ratio") for event in completed]
                ),
            }

        completed_namespaces = sum(
            all(
                states.get((name, pass_name, query.index), {}).get("event") == "completed"
                for pass_name in ("first", "repeat")
                for query in self.queries
            )
            for name in self.search_order
        )
        all_completed = [
            state
            for state in states.values()
            if state.get("event") == "completed" and state.get("namespace") in self.namespaces
        ]
        summary = {
            "status": "complete" if completed_namespaces == len(self.search_order) else "incomplete",
            "rows_per_namespace": self.rows_per_namespace,
            "mode": self.mode,
            "search_field": self.search_field,
            "output_fields": list(self.output_fields),
            "top_k": self.top_k,
            "query_count": len(self.queries),
            "total_rows": sum(self.namespaces[name]["rows"] for name in self.search_order),
            "completed_namespaces": completed_namespaces,
            "cache_temperature": self._cache_temperatures(all_completed),
            "cache_hit_ratio": self._latency_stats(
                [event["performance"].get("cache_hit_ratio") for event in all_completed]
            ),
            "first": bucket_summary["first"],
            "repeat": bucket_summary["repeat"],
            "event_path": str(self.event_path),
            "summary_path": str(self.summary_path),
        }
        MultiTenantSetupRunner._write_json(self.summary_path, summary)
        return summary

    def run(self) -> dict[str, Any]:
        if not self.db.supports_customized_api() or not self.db.supports_namespace_selection():
            raise NotImplementedError("multi-tenant search requires customized queries and namespace selection")
        states = self._states(self._events())
        self._normalize_interrupted(states)
        with self.db.init():
            for namespace in self.search_order:
                all_complete = all(
                    states.get((namespace, pass_name, query.index), {}).get("event") == "completed"
                    for pass_name in ("first", "repeat")
                    for query in self.queries
                )
                if all_complete:
                    continue
                self.db.select_namespace(namespace)
                for query in self.queries:
                    if (namespace, "first", query.index) not in states:
                        states[(namespace, "first", query.index)] = self._query(
                            namespace, "first", query.index, self._query_value(query), disable_cache=True
                        )
                for query in self.queries:
                    first_state = states.get((namespace, "first", query.index))
                    if first_state and first_state["event"] == "completed":
                        if states.get((namespace, "repeat", query.index), {}).get("event") != "completed":
                            states[(namespace, "repeat", query.index)] = self._query(
                                namespace, "repeat", query.index, self._query_value(query), disable_cache=False
                            )
                    elif first_state and (namespace, "repeat", query.index) not in states:
                        states[(namespace, "repeat", query.index)] = self._append_state(
                            namespace, "repeat", query.index, "skipped"
                        )
        summary = self._summary(states)
        if summary["status"] != "complete":
            raise MultiTenantSearchIncomplete(summary)
        return summary
