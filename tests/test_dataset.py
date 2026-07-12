import logging
import pathlib
from typing import Any

import pytest
from pydantic import ValidationError

from vectordb_bench.backend.data_source import DatasetSource, ProgressCallback
from vectordb_bench.backend.dataset import Dataset, FtsDataset

log = logging.getLogger("vectordb_bench")


class RecordingReader:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def read(
        self,
        dataset: str,
        files: list[str],
        local_ds_root: pathlib.Path,
        progress_callback: ProgressCallback | None = None,
    ):
        self.calls.append(
            {
                "dataset": dataset,
                "files": files,
                "local_ds_root": local_ds_root,
                "progress_callback": progress_callback,
            }
        )


class TestDataSet:
    def test_prepare_forwards_progress_callback(self, monkeypatch: pytest.MonkeyPatch):
        reader = RecordingReader()
        monkeypatch.setattr(DatasetSource, "reader", lambda _source: reader)
        manager = Dataset.GIST.manager(100_000)

        def callback(_current: int, _total: int, _message: str):
            return None

        assert manager.prepare(progress_callback=callback)

        assert len(reader.calls) == 1
        assert reader.calls[0]["progress_callback"] is callback
        assert reader.calls[0]["files"] == manager.data.train_files

    def test_prepare_preserves_default_without_progress_callback(self, monkeypatch: pytest.MonkeyPatch):
        reader = RecordingReader()
        monkeypatch.setattr(DatasetSource, "reader", lambda _source: reader)
        manager = Dataset.GIST.manager(100_000)

        assert manager.prepare()

        assert reader.calls[0]["progress_callback"] is None

    def test_fts_prepare_forwards_progress_callback(self, monkeypatch: pytest.MonkeyPatch):
        reader = RecordingReader()
        monkeypatch.setattr(DatasetSource, "reader", lambda _source: reader)
        manager = FtsDataset.MSMARCO.manager(100_000)
        manager.data.with_gt = False
        monkeypatch.setattr(manager._translator, "load", object)

        def callback(_current: int, _total: int, _message: str):
            return None

        assert manager.prepare(source=DatasetSource.IR_DATASETS, progress_callback=callback)

        assert len(reader.calls) == 1
        assert reader.calls[0]["progress_callback"] is callback

    def test_iter_dataset(self):
        for ds in Dataset:
            log.info(ds)

    def test_cohere(self):
        cohere = Dataset.COHERE.get(100_000)
        log.info(cohere)
        assert cohere.name == "Cohere"
        assert cohere.size == 100_000
        assert cohere.label == "SMALL"
        assert cohere.dim == 768

    def test_cohere_error(self):
        with pytest.raises(ValidationError):
            Dataset.COHERE.get(9999)

    def test_iter_cohere(self):
        cohere_10m = Dataset.COHERE.manager(10_000_000)
        cohere_10m.prepare()

        import time
        before = time.time()
        for i in cohere_10m:
            log.debug(i.head(1))

        dur_iter = time.time() - before
        log.warning(f"iter through cohere_10m cost={dur_iter/60}min")

    # pytest -sv tests/test_dataset.py::TestDataSet::test_iter_laion
    def test_iter_laion(self):
        laion_100m = Dataset.LAION.manager(100_000_000)
        from vectordb_bench.backend.data_source import DatasetSource
        laion_100m.prepare(source=DatasetSource.AliyunOSS)

        import time
        before = time.time()
        for i in laion_100m:
            log.debug(i.head(1))

        dur_iter = time.time() - before
        log.warning(f"iter through laion_100m cost={dur_iter/60}min")

    def test_download_small(self):
        openai_50k = Dataset.OPENAI.manager(50_000)
        files = [
            "test.parquet",
            "neighbors.parquet",
            "neighbors_head_1p.parquet",
            "neighbors_tail_1p.parquet",
        ]

        file_path = openai_50k.data_dir.joinpath("test.parquet")
        import os

        DatasetSource.S3.reader().read(
            openai_50k.data.dir_name.lower(),
            files=files,
            local_ds_root=openai_50k.data_dir,
        )

        os.remove(file_path)
        DatasetSource.AliyunOSS.reader().read(
            openai_50k.data.dir_name.lower(),
            files=files,
            local_ds_root=openai_50k.data_dir,
        )
