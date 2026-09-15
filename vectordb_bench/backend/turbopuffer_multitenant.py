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

MANIFEST_VERSION = 2
SEARCH_RESULT_VERSION = 1
SEARCH_ORDER_SEED = 20260914

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
    key: str
    suffix: str
    rows_per_namespace: int
    namespace_count: int
    id_width: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.key) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.suffix):
            raise ValueError("namespace group key and suffix must contain only letters, numbers, '_' or '-'")
        if min(self.rows_per_namespace, self.namespace_count, self.id_width) <= 0:
            raise ValueError("namespace group sizes and id_width must be positive")

    @property
    def source_rows(self) -> int:
        return self.rows_per_namespace * self.namespace_count


DEFAULT_NAMESPACE_GROUPS = (
    NamespaceGroup("A", "1000", 1_000, 3_000, 4),
    NamespaceGroup("B", "3000", 3_000, 1_000, 4),
    NamespaceGroup("C", "15000", 15_000, 200, 3),
    NamespaceGroup("D", "5m", 5_000_000, 1, 7),
)


@dataclass(frozen=True)
class NamespaceSpec:
    name: str
    group: str
    rows: int
    source_start: int
    source_end: int
    fixture: str


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

    def query_fixture(self, dense_field: str, bm25_field: str) -> dict[str, Any]:
        if not self.first or not self.record_batch.num_rows:
            raise ValueError("query fixture requires the first non-empty namespace batch")
        columns = self.record_batch.slice(0, 1).to_pydict()
        return {
            "namespace": self.namespace.name,
            "group": self.namespace.group,
            "row_count": self.namespace.rows,
            "id": columns["id"][0],
            "dense": {"field": dense_field, "value": columns[dense_field][0]},
            "bm25": {"field": bm25_field, "value": columns[bm25_field][0]},
        }


class PreparedMultiTenantDataset:
    def __init__(self, path: Path, groups: tuple[NamespaceGroup, ...] = DEFAULT_NAMESPACE_GROUPS):
        self.path = path.resolve()
        self.groups = groups
        if not self.path.is_file():
            raise FileNotFoundError(f"prepared Parquet does not exist: {self.path}")
        if len({group.key for group in groups}) != len(groups):
            raise ValueError("namespace group keys must be unique")
        parquet_file = pq.ParquetFile(self.path, memory_map=True, pre_buffer=False)
        if parquet_file.schema_arrow != PREPARED_SCHEMA:
            raise ValueError("prepared Parquet schema does not match the turbopuffer multi-tenant contract")
        self.row_count = parquet_file.metadata.num_rows
        required_rows = max((group.source_rows for group in groups), default=0)
        if self.row_count < required_rows:
            raise ValueError(f"prepared Parquet has {self.row_count} rows, but the setup requires {required_rows}")

    @staticmethod
    def namespace_name(run_prefix: str, group: NamespaceGroup, ordinal: int) -> str:
        return f"{run_prefix}_{group.suffix}_{ordinal:0{group.id_width}d}"

    def namespace_specs(self, run_prefix: str, fixture_directory: str) -> list[NamespaceSpec]:
        specs = []
        for group in self.groups:
            for ordinal in range(1, group.namespace_count + 1):
                start = (ordinal - 1) * group.rows_per_namespace
                name = self.namespace_name(run_prefix, group, ordinal)
                specs.append(
                    NamespaceSpec(
                        name=name,
                        group=group.key,
                        rows=group.rows_per_namespace,
                        source_start=start,
                        source_end=start + group.rows_per_namespace,
                        fixture=f"{fixture_directory}/{name}.json",
                    )
                )
        return specs

    def iter_group_batches(
        self,
        run_prefix: str,
        group: NamespaceGroup,
        fixture_directory: str,
        batch_size: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        specs = self.namespace_specs(run_prefix, fixture_directory)
        group_specs = [spec for spec in specs if spec.group == group.key]
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
                    namespace=group_specs[namespace_index],
                    record_batch=batch.slice(offset, take),
                    first=rows_in_namespace == 0,
                    last=rows_in_namespace + take == group.rows_per_namespace,
                )
                consumed += take
                offset += take
        if consumed != group.source_rows:
            raise ValueError(f"read {consumed} rows for group {group.key}, expected {group.source_rows}")


class MultiTenantSetupRunner:
    def __init__(
        self,
        db: VectorDB,
        dataset: PreparedMultiTenantDataset,
        manifest_path: Path,
        run_prefix: str,
        *,
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
        self.db = db
        self.dataset = dataset
        self.manifest_path = manifest_path.resolve()
        self.run_prefix = run_prefix
        self.dense_field = dense_field
        self.bm25_field = bm25_field
        self.schema = customized_schema(dense_field, bm25_field)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fixture_dir = self.manifest_path.with_suffix(".fixtures")
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
        return self.dataset.namespace_specs(self.run_prefix, self.fixture_dir.name)

    def _manifest(self) -> dict[str, Any]:
        specs = self._specs()
        search_order = [spec.name for spec in specs]
        random.Random(SEARCH_ORDER_SEED).shuffle(search_order)
        return {
            "version": MANIFEST_VERSION,
            "run_prefix": self.run_prefix,
            "prepared_data": str(self.dataset.path),
            "prepared_rows": self.dataset.row_count,
            "prepared_size_bytes": self.dataset.path.stat().st_size,
            "field_mapping": {"$meta": "meta_json"},
            "schema": {name: asdict(field) for name, field in self.schema.items()},
            "search_fields": {"dense": self.dense_field, "bm25": self.bm25_field},
            "groups": [asdict(group) for group in self.dataset.groups],
            "namespaces": [asdict(spec) for spec in specs],
            "search_order_seed": SEARCH_ORDER_SEED,
            "search_order": search_order,
            "checkpoint_file": self.checkpoint_path.name,
            "fixture_directory": self.fixture_dir.name,
        }

    def _ensure_manifest(self) -> None:
        expected = self._manifest()
        if self.manifest_path.exists():
            with self.manifest_path.open() as source:
                existing = json.load(source)
            if existing != expected:
                raise ValueError(f"existing setup manifest does not match this run: {self.manifest_path}")
            return
        if self.checkpoint_path.exists() or self.fixture_dir.exists():
            raise FileExistsError("setup artifacts exist without a matching manifest")
        self._write_json(self.manifest_path, expected)

    def _append_checkpoint(self, spec: NamespaceSpec, state: str, inserted_rows: int) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "namespace": spec.name,
                        "group": spec.group,
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
        self._ensure_manifest()
        specs = self._specs()
        by_name = {spec.name: spec for spec in specs}
        states = _setup_checkpoint_states(self.checkpoint_path, set(by_name))
        completed = {name for name, state in states.items() if state == "completed"}
        for name in completed:
            if not (self.manifest_path.parent / by_name[name].fixture).is_file():
                raise ValueError(f"completed namespace is missing its query fixture: {name}")
        if len(completed) == len(specs):
            return self._summary(0, completed, len(specs))

        newly_inserted = 0
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        with self.db.init():
            for group in self.dataset.groups:
                group_names = {spec.name for spec in specs if spec.group == group.key}
                if group_names <= completed:
                    continue
                active_name = None
                active_count = 0
                fixture = None
                for batch in self.dataset.iter_group_batches(
                    self.run_prefix,
                    group,
                    self.fixture_dir.name,
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
                        fixture = batch.query_fixture(self.dense_field, self.bm25_field)
                    if active_name != spec.name:
                        raise RuntimeError(f"non-contiguous batches for namespace {spec.name}")
                    rows = batch.customized_rows(self.schema)
                    self._insert(spec.name, rows)
                    active_count += len(rows)
                    if batch.last:
                        if active_count != spec.rows or fixture is None:
                            raise RuntimeError(
                                f"namespace {spec.name} inserted {active_count} rows, expected {spec.rows}"
                            )
                        self._write_json(self.manifest_path.parent / spec.fixture, fixture)
                        self._append_checkpoint(spec, "completed", active_count)
                        states[spec.name] = "completed"
                        completed.add(spec.name)
                        newly_inserted += active_count
                        active_name = None
                        fixture = None
        return self._summary(newly_inserted, completed, len(specs))

    def _summary(self, inserted_rows: int, completed: set[str], total_namespaces: int) -> dict[str, int | str]:
        return {
            "inserted_rows": inserted_rows,
            "completed_namespaces": len(completed),
            "total_namespaces": total_namespaces,
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
        group: str = "all",
        output_fields: tuple[str, ...] = (),
        top_k: int = 100,
    ):
        if mode not in {"dense", "bm25"}:
            raise ValueError("multi-tenant search mode must be dense or bm25")
        if group not in {"all", "A", "B", "C", "D"}:
            raise ValueError("multi-tenant group must be all, A, B, C, or D")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(set(output_fields)) != len(output_fields) or "id" in output_fields:
            raise ValueError("output fields must be unique and must not contain id")

        self.db = db
        self.manifest_path = manifest_path.resolve()
        self.mode = mode
        self.group = group
        self.output_fields = output_fields
        self.top_k = top_k
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.manifest_hash = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.event_path = self.manifest_path.with_name(f"{self.manifest_path.stem}.{mode}.search.jsonl")
        self.summary_path = self.manifest_path.with_name(
            f"{self.manifest_path.stem}.{mode}.{group.lower()}.summary.json"
        )
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        if self.manifest.get("version") != MANIFEST_VERSION:
            raise ValueError(f"setup manifest must have version {MANIFEST_VERSION}")
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

        entries = self.manifest.get("namespaces", [])
        self.namespaces = {entry["name"]: entry for entry in entries}
        order = self.manifest.get("search_order", [])
        if len(self.namespaces) != len(entries) or set(order) != set(self.namespaces) or len(order) != len(entries):
            raise ValueError("manifest namespaces and search order do not match")
        self.search_order = [name for name in order if self.group == "all" or self.namespaces[name]["group"] == self.group]
        if not self.search_order:
            raise ValueError(f"manifest contains no namespaces for group {self.group}")

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
        }

    def _events(self) -> list[dict[str, Any]]:
        if not self.event_path.exists():
            self._append_jsonl(self.event_path, self._header())
        with self.event_path.open() as source:
            events = [json.loads(line) for line in source if line.strip()]
        if not events or events[0] != self._header():
            raise ValueError(f"search checkpoint does not match this run: {self.event_path}")
        for event in events[1:]:
            if event.get("namespace") not in self.namespaces or event.get("pass") not in {"first", "repeat"}:
                raise ValueError(f"invalid search checkpoint event: {self.event_path}")
        return events

    @staticmethod
    def _states(events: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        return {(event["namespace"], event["pass"]): event for event in events[1:]}

    def _append_state(self, namespace: str, pass_name: str, event: str, **values: Any) -> dict[str, Any]:
        value = {"event": event, "namespace": namespace, "pass": pass_name, **values}
        self._append_jsonl(self.event_path, value)
        return value

    def _normalize_interrupted(self, states: dict[tuple[str, str], dict[str, Any]]) -> None:
        for namespace in self.search_order:
            first = states.get((namespace, "first"))
            repeat = states.get((namespace, "repeat"))
            if first and first["event"] == "started":
                states[(namespace, "first")] = self._append_state(namespace, "first", "indeterminate")
                if repeat is None:
                    states[(namespace, "repeat")] = self._append_state(namespace, "repeat", "skipped")
            elif first and first["event"] == "error" and repeat is None:
                states[(namespace, "repeat")] = self._append_state(namespace, "repeat", "skipped")
            elif first and first["event"] == "indeterminate" and repeat is None:
                states[(namespace, "repeat")] = self._append_state(namespace, "repeat", "skipped")
            elif repeat and repeat["event"] == "started":
                states[(namespace, "repeat")] = self._append_state(namespace, "repeat", "indeterminate")

    def _fixture(self, namespace: str) -> dict[str, Any]:
        relative = Path(self.namespaces[namespace]["fixture"])
        fixture_path = (self.manifest_path.parent / relative).resolve()
        if self.manifest_path.parent not in fixture_path.parents:
            raise ValueError(f"fixture path escapes the manifest directory: {relative}")
        with fixture_path.open() as source:
            fixture = json.load(source)
        query = fixture.get(self.mode, {})
        if fixture.get("namespace") != namespace or query.get("field") != self.search_field:
            raise ValueError(f"query fixture does not match the manifest: {fixture_path}")
        return query

    def _query(self, namespace: str, pass_name: str, query: dict[str, Any]) -> dict[str, Any]:
        self._append_state(namespace, pass_name, "started")
        started = time.perf_counter()
        try:
            request = CustomizedRequest(
                self.mode,
                self.search_field,
                query["value"],
                self.top_k,
                self.output_fields,
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
                "error",
                client_latency_ms=round(latency_ms, 4),
                error_type=type(error).__name__,
            )

    @staticmethod
    def _latency_stats(values: list[float | None]) -> dict[str, float | int]:
        latencies = [value for value in values if value is not None]
        if not latencies:
            return {"count": 0}
        return {
            "count": len(latencies),
            "average_ms": round(float(np.mean(latencies)), 4),
            "p50_ms": round(float(np.percentile(latencies, 50)), 4),
            "p95_ms": round(float(np.percentile(latencies, 95)), 4),
            "p99_ms": round(float(np.percentile(latencies, 99)), 4),
        }

    def _summary(self, states: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
        groups = {}
        for group in sorted({self.namespaces[name]["group"] for name in self.search_order}):
            names = [name for name in self.search_order if self.namespaces[name]["group"] == group]
            group_summary = {
                "rows_per_namespace": self.namespaces[names[0]]["rows"],
                "namespace_count": len(names),
            }
            for pass_name in ("first", "repeat"):
                events = [states[(name, pass_name)] for name in names if (name, pass_name) in states]
                group_summary[pass_name] = {
                    **self._latency_stats(
                        [event["client_latency_ms"] for event in events if event["event"] == "completed"]
                    ),
                    "outcomes": dict(sorted(Counter(event["event"] for event in events).items())),
                    "server_total_ms": self._latency_stats(
                        [
                            event["performance"].get("server_total_ms")
                            for event in events
                            if event["event"] == "completed"
                        ]
                    ),
                    "query_execution_ms": self._latency_stats(
                        [
                            event["performance"].get("query_execution_ms")
                            for event in events
                            if event["event"] == "completed"
                        ]
                    ),
                }
            groups[group] = group_summary

        completed_namespaces = sum(
            states.get((name, "first"), {}).get("event") == "completed"
            and states.get((name, "repeat"), {}).get("event") == "completed"
            for name in self.search_order
        )
        summary = {
            "status": "complete" if completed_namespaces == len(self.search_order) else "incomplete",
            "mode": self.mode,
            "group": self.group,
            "search_field": self.search_field,
            "output_fields": list(self.output_fields),
            "top_k": self.top_k,
            "namespace_count": len(self.search_order),
            "completed_namespaces": completed_namespaces,
            "groups": groups,
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
                first = states.get((namespace, "first"))
                repeat = states.get((namespace, "repeat"))
                if first is not None and (first["event"] != "completed" or repeat is not None):
                    continue
                self.db.select_namespace(namespace)
                query = self._fixture(namespace)
                if first is None:
                    first = self._query(namespace, "first", query)
                    states[(namespace, "first")] = first
                if first["event"] == "completed":
                    states[(namespace, "repeat")] = self._query(namespace, "repeat", query)
                else:
                    states[(namespace, "repeat")] = self._append_state(namespace, "repeat", "skipped")
        summary = self._summary(states)
        if summary["status"] != "complete":
            raise MultiTenantSearchIncomplete(summary)
        return summary
