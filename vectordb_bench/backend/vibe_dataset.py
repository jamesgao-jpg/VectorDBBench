import importlib.metadata
import json
import logging
import math
import pathlib
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import ClassVar

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from vectordb_bench import config

from .data_source import DatasetSource
from .dataset import BaseDataset, DatasetManager, ParquetGroundTruth, SearchDatasetFiles, SizeLabel
from .filter import Filter, FilterOp, non_filter
from .vibe_catalog import VIBE_DATASETS, VIBE_REPO_ID, VIBE_REVISION, VibeDatasetSpec, get_vibe_dataset

log = logging.getLogger(__name__)

CONVERSION_SCHEMA_VERSION = 1
GROUND_TRUTH_WIDTH = 100
PARQUET_TARGET_BYTES = 512 * 1024 * 1024
CONVERSION_CHUNK_BYTES = 64 * 1024 * 1024


class VibeDataset(BaseDataset):
    name: str
    size: int
    dim: int
    use_shuffled: bool = False
    with_gt: bool = True
    with_remote_resource: bool = False
    _size_label: ClassVar[dict[int, SizeLabel]] = {}

    @property
    def label(self) -> str:
        return "VIBE"

    @property
    def full_name(self) -> str:
        return self.name

    @property
    def dir_name(self) -> str:
        return self.name

    @property
    def file_count(self) -> int:
        raw_bytes = self.size * self.dim * np.dtype(np.float32).itemsize
        return max(1, math.ceil(raw_bytes / PARQUET_TARGET_BYTES))


class VibeDatasetManager(DatasetManager):
    spec: VibeDatasetSpec
    result_metadata: dict | None = None

    @classmethod
    def from_name(cls, name: str) -> "VibeDatasetManager":
        spec = get_vibe_dataset(name)
        data = VibeDataset(
            name=spec.name,
            size=spec.size,
            dim=spec.dimension,
            metric_type=spec.metric_type,
            use_shuffled=False,
        )
        return cls(data=data, spec=spec)

    @property
    def preferred_source(self) -> DatasetSource:
        return DatasetSource.HuggingFace

    @property
    def data_dir(self) -> pathlib.Path:
        return pathlib.Path(
            config.DATASET_LOCAL_DIR,
            "vibe",
            self.spec.name,
            VIBE_REVISION,
            f"schema-v{CONVERSION_SCHEMA_VERSION}",
        )

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.data_dir / "manifest.json"

    def max_search_k(self, filters: Filter = non_filter) -> int | None:
        self._validate_unfiltered(filters)
        return GROUND_TRUTH_WIDTH

    def resolve_search_files(self, *, k: int, filters: Filter = non_filter) -> SearchDatasetFiles:
        self._validate_unfiltered(filters)
        if not 1 <= k <= GROUND_TRUTH_WIDTH:
            msg = f"VIBE supports K from 1 to {GROUND_TRUTH_WIDTH}, got {k}"
            raise ValueError(msg)
        return SearchDatasetFiles("test.parquet", "neighbors.parquet", width=GROUND_TRUTH_WIDTH)

    def prepare(
        self,
        source: DatasetSource = DatasetSource.HuggingFace,
        filters: Filter = non_filter,
        with_train_files: bool = True,
        with_scalar_labels: bool = False,
        k: int | None = None,
    ) -> bool:
        requested_k = config.K_DEFAULT if k is None else k
        self.resolve_search_files(k=requested_k, filters=filters)
        if not self._manifest_is_valid():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(self.data_dir / ".prepare.lock"):
                if not self._manifest_is_valid():
                    cache_dir = pathlib.Path(config.DATASET_LOCAL_DIR, "huggingface-cache")
                    paths = source.reader().read(
                        VIBE_REPO_ID,
                        [self.spec.filename],
                        cache_dir,
                        revision=VIBE_REVISION,
                    )
                    self._convert(paths[self.spec.filename])
        manifest = self._load_valid_manifest()
        self.result_metadata = self._metadata_from_manifest(manifest)
        return super().prepare(
            source=source,
            filters=filters,
            with_train_files=with_train_files,
            with_scalar_labels=with_scalar_labels,
            k=requested_k,
        )

    @staticmethod
    def _validate_unfiltered(filters: Filter) -> None:
        if filters.type != FilterOp.NonFilter:
            msg = "Canonical VIBE datasets do not contain scalar fields or filtered ground truth"
            raise ValueError(msg)

    def _load_valid_manifest(self) -> dict:
        if not self._manifest_is_valid():
            msg = f"Incomplete or invalid VIBE conversion manifest: {self.manifest_path}"
            raise ValueError(msg)
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _manifest_is_valid(self) -> bool:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        expected = {
            "schema_version": CONVERSION_SCHEMA_VERSION,
            "repository": VIBE_REPO_ID,
            "source": DatasetSource.HuggingFace.value,
            "requested_revision": VIBE_REVISION,
            "resolved_revision": VIBE_REVISION,
            "source_filename": self.spec.filename,
            "name": self.spec.name,
            "size": self.spec.size,
            "dimension": self.spec.dimension,
            "distribution": self.spec.distribution,
            "lifecycle": self.spec.lifecycle,
            "source_distance": self.spec.source_distance,
            "metric_type": self.spec.metric_type.value,
            "point_type": self.spec.point_type,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False
        outputs = manifest.get("outputs")
        expected_files = {*self.data.train_files, "test.parquet", "neighbors.parquet"}
        if not isinstance(outputs, dict) or set(outputs) != expected_files:
            return False
        for name, metadata in outputs.items():
            if not isinstance(metadata, dict):
                return False
            path = self.data_dir / name
            if not path.is_file() or path.stat().st_size != metadata.get("file_size"):
                return False
        return True

    def _convert(self, source_path: pathlib.Path) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.unlink(missing_ok=True)
        with h5py.File(source_path, "r") as source:
            source_info = self._validate_source(source)
            train = source["train"]
            test = source["test"]
            neighbors = source["neighbors"]
            vector_type = pa.from_numpy_dtype(train.dtype)
            train_rows_per_file = math.ceil(self.spec.size / self.data.file_count)
            outputs = {}
            for index, filename in enumerate(self.data.train_files):
                start = index * train_rows_per_file
                end = min(start + train_rows_per_file, self.spec.size)
                self._write_vectors(train, start, end, self.data_dir / filename, vector_type)
                outputs[filename] = self._parquet_metadata(self.data_dir / filename)
            self._write_vectors(test, 0, test.shape[0], self.data_dir / "test.parquet", vector_type)
            outputs["test.parquet"] = self._parquet_metadata(self.data_dir / "test.parquet")
            self._write_neighbors(neighbors, self.data_dir / "neighbors.parquet")
            outputs["neighbors.parquet"] = self._parquet_metadata(self.data_dir / "neighbors.parquet")
            self._validate_outputs(outputs, query_count=test.shape[0])

        manifest = {
            "schema_version": CONVERSION_SCHEMA_VERSION,
            "repository": VIBE_REPO_ID,
            "source": DatasetSource.HuggingFace.value,
            "requested_revision": VIBE_REVISION,
            "resolved_revision": VIBE_REVISION,
            "source_filename": self.spec.filename,
            "source_content_identifier": source_path.resolve().name,
            "source_attributes": source_info,
            "name": self.spec.name,
            "distribution": self.spec.distribution,
            "lifecycle": self.spec.lifecycle,
            "size": self.spec.size,
            "dimension": self.spec.dimension,
            "source_distance": self.spec.source_distance,
            "metric_type": self.spec.metric_type.value,
            "point_type": self.spec.point_type,
            "outputs": outputs,
            "prepared_at": datetime.now(UTC).isoformat(),
            "vectordb_bench_version": self._package_version(),
        }
        self._atomic_write_json(self.manifest_path, manifest)

    def _validate_source(self, source: h5py.File) -> dict:
        required_attrs = {"dimension", "distance", "point_type"}
        missing_attrs = required_attrs - set(source.attrs)
        required_arrays = {"train", "test", "neighbors", "distances"}
        missing_arrays = required_arrays - set(source)
        if missing_attrs or missing_arrays:
            msg = (
                f"Invalid VIBE HDF5 {self.spec.filename}: missing attrs={sorted(missing_attrs)}, "
                f"arrays={sorted(missing_arrays)}"
            )
            raise ValueError(msg)
        dimension = int(source.attrs["dimension"])
        distance = self._text_attr(source.attrs["distance"]).lower()
        point_type = self._text_attr(source.attrs["point_type"]).lower()
        train, test = source["train"], source["test"]
        neighbors, distances = source["neighbors"], source["distances"]
        if (
            dimension != self.spec.dimension
            or distance != self.spec.source_distance
            or point_type != self.spec.point_type
        ):
            msg = f"VIBE HDF5 metadata does not match catalog for {self.spec.name}"
            raise ValueError(msg)
        if train.shape != (self.spec.size, self.spec.dimension):
            msg = f"Unexpected train shape for {self.spec.name}: {train.shape}"
            raise ValueError(msg)
        if test.ndim != 2 or test.shape[1] != self.spec.dimension:
            msg = f"Unexpected test shape for {self.spec.name}: {test.shape}"
            raise ValueError(msg)
        expected_gt_shape = (test.shape[0], GROUND_TRUTH_WIDTH)
        if neighbors.shape != expected_gt_shape or distances.shape != expected_gt_shape:
            msg = f"Unexpected ground-truth shapes for {self.spec.name}: {neighbors.shape}, {distances.shape}"
            raise ValueError(msg)
        if train.dtype != test.dtype or not np.issubdtype(train.dtype, np.floating):
            msg = f"Unsupported vector dtype for {self.spec.name}: {train.dtype}/{test.dtype}"
            raise ValueError(msg)
        if not np.issubdtype(neighbors.dtype, np.integer):
            msg = f"VIBE neighbors must be integers, got {neighbors.dtype}"
            raise ValueError(msg)
        chunk_rows = self._chunk_rows(GROUND_TRUTH_WIDTH, neighbors.dtype.itemsize)
        for start in range(0, neighbors.shape[0], chunk_rows):
            values = neighbors[start : start + chunk_rows]
            if values.size and (values.min() < 0 or values.max() >= self.spec.size):
                msg = f"VIBE neighbor ID is outside [0, {self.spec.size})"
                raise ValueError(msg)
        return {
            "dimension": dimension,
            "distance": distance,
            "point_type": point_type,
            "vector_dtype": str(train.dtype),
            "query_count": test.shape[0],
            "ground_truth_width": GROUND_TRUTH_WIDTH,
        }

    def _write_vectors(
        self,
        source: h5py.Dataset,
        start: int,
        end: int,
        path: pathlib.Path,
        vector_type: pa.DataType,
    ) -> None:
        chunk_rows = self._chunk_rows(self.spec.dimension, source.dtype.itemsize)

        def tables():
            for offset in range(start, end, chunk_rows):
                stop = min(offset + chunk_rows, end)
                vectors = np.ascontiguousarray(source[offset:stop])
                flat = pa.array(vectors.reshape(-1), type=vector_type)
                yield pa.table(
                    {
                        "id": pa.array(np.arange(offset, stop, dtype=np.int64)),
                        "emb": pa.FixedSizeListArray.from_arrays(flat, self.spec.dimension),
                    }
                )

        self._atomic_write_parquet(path, tables())

    def _write_neighbors(self, source: h5py.Dataset, path: pathlib.Path) -> None:
        chunk_rows = self._chunk_rows(GROUND_TRUTH_WIDTH, source.dtype.itemsize)

        def tables():
            for start in range(0, source.shape[0], chunk_rows):
                end = min(start + chunk_rows, source.shape[0])
                values = np.ascontiguousarray(source[start:end], dtype=np.int64)
                yield pa.table(
                    {
                        "id": pa.array(np.arange(start, end, dtype=np.int64)),
                        "neighbors_id": pa.FixedSizeListArray.from_arrays(
                            pa.array(values.reshape(-1), type=pa.int64()),
                            GROUND_TRUTH_WIDTH,
                        ),
                    }
                )

        self._atomic_write_parquet(path, tables())

    def _validate_outputs(self, outputs: dict, *, query_count: int) -> None:
        train_rows = sum(outputs[name]["row_count"] for name in self.data.train_files)
        if train_rows != self.spec.size:
            msg = f"Prepared train row count {train_rows} does not match {self.spec.size}"
            raise ValueError(msg)
        if outputs["test.parquet"]["row_count"] != query_count:
            raise ValueError("Prepared test row count does not match source")
        if outputs["neighbors.parquet"]["row_count"] != query_count:
            raise ValueError("Prepared ground-truth row count does not match source")
        test_path = self.data_dir / "test.parquet"
        test_ids = pq.read_table(test_path, columns=[self.data.test_id_field]).column(0).to_pylist()
        expected_ids = list(range(query_count))
        if test_ids != expected_ids:
            raise ValueError("Prepared query IDs are not zero-based and contiguous")
        ParquetGroundTruth.from_file(
            self.data_dir / "neighbors.parquet",
            id_field=self.data.gt_id_field,
            neighbors_field=self.data.gt_neighbors_field,
            expected_query_ids=expected_ids,
            minimum_width=GROUND_TRUTH_WIDTH,
            expected_width=GROUND_TRUTH_WIDTH,
        )

    @staticmethod
    def _atomic_write_parquet(path: pathlib.Path, tables: Iterable[pa.Table]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        writer = None
        try:
            for table in tables:
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
            if writer is None:
                msg = f"Cannot write empty Parquet output: {path.name}"
                raise ValueError(msg)
            writer.close()
            writer = None
            temporary.replace(path)
        finally:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_json(path: pathlib.Path, value: dict) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _parquet_metadata(path: pathlib.Path) -> dict:
        parquet = pq.ParquetFile(path)
        return {
            "row_count": parquet.metadata.num_rows,
            "schema": str(parquet.schema_arrow),
            "file_size": path.stat().st_size,
        }

    @staticmethod
    def _chunk_rows(width: int, item_size: int) -> int:
        return max(1, CONVERSION_CHUNK_BYTES // max(1, width * item_size))

    @staticmethod
    def _text_attr(value: object) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _package_version() -> str:
        try:
            return importlib.metadata.version("vectordb-bench")
        except importlib.metadata.PackageNotFoundError:
            return "source"

    @staticmethod
    def _metadata_from_manifest(manifest: dict) -> dict:
        return {
            "name": manifest["name"],
            "distribution": manifest["distribution"],
            "lifecycle": manifest["lifecycle"],
            "source": manifest["source"],
            "repository": manifest["repository"],
            "filename": manifest["source_filename"],
            "revision": manifest["resolved_revision"],
            "source_distance": manifest["source_distance"],
            "metric_type": manifest["metric_type"],
            "point_type": manifest["point_type"],
        }


VibeDataset._size_label = {spec.size: SizeLabel(spec.size, "VIBE", 1) for spec in VIBE_DATASETS}
