import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vectordb_bench import config
from vectordb_bench.backend.clients.api import VectorDB
from vectordb_bench.backend.customized import CustomizedRow, FieldSchema

MANIFEST_VERSION = 1
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

    def customized_rows(self) -> list[CustomizedRow]:
        columns = self.record_batch.to_pydict()
        return [
            CustomizedRow(
                id=columns["id"][index],
                fields={name: columns[name][index] for name in CUSTOMIZED_SCHEMA},
            )
            for index in range(self.record_batch.num_rows)
        ]

    def query_fixture(self) -> dict[str, Any]:
        if not self.first or not self.record_batch.num_rows:
            raise ValueError("query fixture requires the first non-empty namespace batch")
        columns = self.record_batch.slice(0, 1).to_pydict()
        return {
            "namespace": self.namespace.name,
            "group": self.namespace.group,
            "row_count": self.namespace.rows,
            "id": columns["id"][0],
            "dense": {"field": "emb_768", "value": columns["emb_768"][0]},
            "bm25": {"field": "content", "value": columns["content"][0]},
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
            "schema": {name: asdict(field) for name, field in CUSTOMIZED_SCHEMA.items()},
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

    def _checkpoint_states(self, valid_namespaces: set[str]) -> dict[str, str]:
        states = {}
        if not self.checkpoint_path.exists():
            return states
        with self.checkpoint_path.open() as source:
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
            inserted, error = self.db.insert_customized_rows(rows, CUSTOMIZED_SCHEMA)
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
        states = self._checkpoint_states(set(by_name))
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
                        fixture = batch.query_fixture()
                    if active_name != spec.name:
                        raise RuntimeError(f"non-contiguous batches for namespace {spec.name}")
                    rows = batch.customized_rows()
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
