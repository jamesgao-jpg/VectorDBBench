import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from vectordb_bench import config
from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import VibePerformance
from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig, MetricType
from vectordb_bench.backend.data_source import DatasetSource, HuggingFaceReader
from vectordb_bench.backend.dataset import SizeLabel
from vectordb_bench.backend.filter import LabelFilter, non_filter
from vectordb_bench.backend.vibe_catalog import VIBE_DATASETS, VIBE_REVISION
from vectordb_bench.backend.vibe_dataset import VibeDataset, VibeDatasetManager
from vectordb_bench.cli.cli import get_custom_case_config
from vectordb_bench.frontend.components.check_results.data import mergeTasks
from vectordb_bench.frontend.config.dbCaseConfigs import UI_CASE_CLUSTERS
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, CaseResult, CaseType, TaskConfig, TestResult
from vectordb_bench.restful.format_res import format_results


def test_vibe_catalog_is_the_advertised_24_dataset_set():
    assert len(VIBE_DATASETS) == 24
    assert len({spec.name for spec in VIBE_DATASETS}) == 24
    assert sum(spec.lifecycle == "active" for spec in VIBE_DATASETS) == 19
    assert sum(spec.lifecycle == "deprecated" for spec in VIBE_DATASETS) == 5
    assert sum(spec.distribution == "id" for spec in VIBE_DATASETS) == 15
    assert sum(spec.distribution == "ood" for spec in VIBE_DATASETS) == 9
    assert all(spec.filename == f"{spec.name}.hdf5" for spec in VIBE_DATASETS)
    ip_specs = [spec for spec in VIBE_DATASETS if spec.source_distance == "ip"]
    assert len(ip_specs) == 4
    assert all(spec.metric_type == MetricType.IP for spec in ip_specs)
    assert isinstance(DatasetSource.HuggingFace.reader(), HuggingFaceReader)


def test_hugging_face_reader_uses_pinned_single_file_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cached.hdf5"
    destination.write_bytes(b"hdf5")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(destination)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    paths = HuggingFaceReader().read(
        "vector-index-bench/vibe",
        ["glove-200-cosine.hdf5"],
        tmp_path / "cache",
        revision=VIBE_REVISION,
    )

    assert paths == {"glove-200-cosine.hdf5": destination}
    assert calls == [
        {
            "repo_id": "vector-index-bench/vibe",
            "filename": "glove-200-cosine.hdf5",
            "repo_type": "dataset",
            "revision": VIBE_REVISION,
            "cache_dir": tmp_path / "cache",
        }
    ]


def _tiny_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distance: str,
) -> tuple[VibeDatasetManager, Path, np.ndarray]:
    metric = MetricType.L2 if distance == "euclidean" else MetricType.IP if distance == "ip" else MetricType.COSINE
    spec = VIBE_DATASETS[0].model_copy(
        update={
            "name": f"tiny-{distance}",
            "size": 5,
            "dimension": 3,
            "source_distance": distance,
            "metric_type": metric,
            "filename": f"tiny-{distance}.hdf5",
        }
    )
    monkeypatch.setitem(VibeDataset._size_label, 5, SizeLabel(5, "VIBE", 1))
    monkeypatch.setattr(config, "DATASET_LOCAL_DIR", tmp_path / "datasets")
    data = VibeDataset(
        name=spec.name,
        size=spec.size,
        dim=spec.dimension,
        metric_type=spec.metric_type,
        use_shuffled=False,
    )
    manager = VibeDatasetManager(data=data, spec=spec)
    source_path = tmp_path / spec.filename
    train = np.arange(15, dtype=np.float32).reshape(5, 3) / 7
    queries = np.array([[0.25, -0.5, 1.5], [3.25, 2.5, -1.0]], dtype=np.float32)
    neighbors = np.tile(np.arange(100, dtype=np.int64) % 5, (2, 1))
    with h5py.File(source_path, "w") as source:
        source.attrs["dimension"] = 3
        source.attrs["distance"] = distance
        source.attrs["point_type"] = "float"
        source.create_dataset("train", data=train)
        source.create_dataset("test", data=queries)
        source.create_dataset("neighbors", data=neighbors)
        source.create_dataset("distances", data=np.zeros((2, 100), dtype=np.float32))
    return manager, source_path, queries


@pytest.mark.parametrize("distance", ["euclidean", "normalized", "ip"])
def test_vibe_conversion_preserves_vectors_metrics_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distance: str,
):
    manager, source_path, queries = _tiny_manager(tmp_path, monkeypatch, distance)
    calls = []

    class Reader:
        def read(
            self,
            dataset: str,
            files: list[str],
            local_ds_root: Path,
            *,
            revision: str | None = None,
        ) -> dict[str, Path]:
            calls.append((dataset, files, local_ds_root, revision))
            return {manager.spec.filename: source_path}

    monkeypatch.setattr(DatasetSource, "reader", lambda _source: Reader())
    assert manager.prepare(k=10)

    assert calls[0][3] == VIBE_REVISION
    assert manager.test_data == queries.tolist()
    assert manager.gt_data.width == 100
    assert manager.result_metadata["metric_type"] == manager.spec.metric_type.value
    assert manager.result_metadata["revision"] == VIBE_REVISION
    train = pl.read_parquet(manager.data_dir / manager.data.train_files[0])
    assert train["id"].to_list() == list(range(5))
    expected_train = np.arange(15, dtype=np.float32).reshape(5, 3) / np.float32(7)
    assert np.array_equal(np.asarray(train["emb"].to_list(), dtype=np.float32), expected_train)

    monkeypatch.setattr(DatasetSource, "reader", lambda _source: pytest.fail("valid manifest must be reused"))
    restored = VibeDatasetManager(data=manager.data, spec=manager.spec)
    assert restored.prepare(k=10)
    assert restored.result_metadata == manager.result_metadata


def test_vibe_rejects_invalid_ground_truth_and_noncanonical_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager, source_path, _ = _tiny_manager(tmp_path, monkeypatch, "ip")
    with h5py.File(source_path, "r+") as source:
        source["neighbors"][0, 0] = 5
    with pytest.raises(ValueError, match="outside"):
        manager._convert(source_path)

    assert manager.max_search_k(non_filter) == 100
    with pytest.raises(ValueError, match="K from 1 to 100"):
        manager.resolve_search_files(k=101)
    with pytest.raises(ValueError, match="do not contain scalar"):
        manager.resolve_search_files(k=10, filters=LabelFilter(label_percentage=0.5))


def test_vibe_case_cli_ui_and_preferred_source():
    case = VibePerformance(vibe_dataset="glove-200-cosine")
    assert case.dataset.data.metric_type == MetricType.COSINE
    assert case.dataset.preferred_source == DatasetSource.HuggingFace
    assert get_custom_case_config(
        {
            "case_type": "VibePerformance",
            "vibe_dataset": "glove-200-cosine",
            "dataset_with_size_type": None,
        }
    ) == {"vibe_dataset": "glove-200-cosine"}

    active = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label == "VIBE Search Performance")
    deprecated = next(cluster for cluster in UI_CASE_CLUSTERS if cluster.label.endswith("(Deprecated)"))
    assert len(active.uiCaseItems) == 19
    assert len(deprecated.uiCaseItems) == 5

    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.VibePerformance,
            custom_case={"vibe_dataset": "glove-200-cosine"},
        ),
    )
    runner = Assembler.assemble("run-id", task, DatasetSource.AliyunOSS)
    assert runner.dataset_source == DatasetSource.HuggingFace

    fts_case = CaseConfig(case_id=CaseType.FTSBm25Performance).case
    assert fts_case.dataset.preferred_source == DatasetSource.IR_DATASETS
    legacy_task = task.model_copy(update={"case_config": CaseConfig(case_id=CaseType.Performance768D1M)})
    assert Assembler.assemble("run-id", legacy_task, DatasetSource.AliyunOSS).dataset_source == DatasetSource.AliyunOSS

    with pytest.raises(ValueError, match="do not support filter"):
        VibePerformance(vibe_dataset="glove-200-cosine", filter_rate=0.5)


def test_vibe_k_above_100_fails_during_case_config_validation():
    with pytest.raises(ValueError, match="K from 1 to 100"):
        CaseConfig(
            case_id=CaseType.VibePerformance,
            custom_case={"vibe_dataset": "glove-200-cosine"},
            k=101,
        )


def test_vibe_result_metadata_is_optional_and_round_trips(tmp_path: Path):
    task = TaskConfig(
        db=DB.Test,
        db_config=DB.Test.config_cls(),
        db_case_config=EmptyDBCaseConfig(),
        case_config=CaseConfig(
            case_id=CaseType.VibePerformance,
            custom_case={"vibe_dataset": "glove-200-cosine"},
        ),
    )
    old_result = TestResult(
        run_id="old-vibe",
        task_label="old-vibe",
        results=[CaseResult(metrics=Metric(), task_config=task)],
    )
    old_payload = old_result.model_dump_for_output()
    old_payload["results"][0].pop("dataset_metadata")
    old_path = tmp_path / "old-result.json"
    old_path.write_text(json.dumps(old_payload), encoding="utf-8")
    assert TestResult.read_file(old_path).results[0].dataset_metadata is None

    metadata = {
        "name": "glove-200-cosine",
        "distribution": "id",
        "lifecycle": "active",
        "source": "HuggingFace",
        "repository": "vector-index-bench/vibe",
        "filename": "glove-200-cosine.hdf5",
        "revision": VIBE_REVISION,
        "source_distance": "cosine",
        "metric_type": "COSINE",
        "point_type": "float",
    }
    case_result = CaseResult(metrics=Metric(), task_config=task, dataset_metadata=metadata)
    payload = case_result.model_dump(mode="json")
    assert json.loads(json.dumps(payload))["dataset_metadata"] == metadata

    test_result = TestResult(run_id="vibe", task_label="vibe", results=[case_result])
    result_path = tmp_path / "vibe-result.json"
    result_path.write_text(json.dumps(test_result.model_dump_for_output()), encoding="utf-8")
    loaded_metadata = TestResult.read_file(result_path).results[0].dataset_metadata
    assert loaded_metadata is not None
    assert loaded_metadata.model_dump(mode="json") == metadata
    [rest_payload] = format_results([test_result], "vibe")
    assert rest_payload["dataset_metadata"] == metadata
    merged, failed = mergeTasks([case_result])
    assert not failed
    assert merged[0]["dataset_metadata"] == metadata
