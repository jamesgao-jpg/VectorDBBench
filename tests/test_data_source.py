import logging
import pathlib

import pytest

from vectordb_bench.backend import data_source
from vectordb_bench.backend.cases import type2case
from vectordb_bench.backend.data_source import AliyunOSSReader, AwsS3Reader, DatasetSource, IRDatasetsReader

log = logging.getLogger("vectordb_bench")


class FakeS3FileSystem:
    def __init__(self, files: dict[str, bytes], fail_on: str | None = None):
        self.files = files
        self.fail_on = fail_on
        self.destinations: list[pathlib.Path] = []

    def info(self, remote: str | pathlib.PurePath):
        return {"size": len(self.files[str(remote)])}

    def get_file(self, remote: str, local: str, callback: data_source.Callback):
        payload = self.files[remote]
        destination = pathlib.Path(local)
        self.destinations.append(destination)
        callback.set_size(len(payload))

        with destination.open("wb") as output:
            if remote == self.fail_on:
                output.truncate(len(payload))
                callback.relative_update(len(payload) // 2)
                raise RuntimeError("simulated download failure")

            midpoint = max(1, len(payload) // 2)
            for chunk in (payload[:midpoint], payload[midpoint:]):
                output.write(chunk)
                callback.relative_update(len(chunk))


class FakeObjectMeta:
    def __init__(self, content_length: int):
        self.content_length = content_length


class FakeAliyunBucket:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def get_object_meta(self, remote: str):
        return FakeObjectMeta(len(self.files[remote]))

    def get_object_to_file(self, remote: str, local: str | pathlib.Path):
        pathlib.Path(local).write_bytes(self.files[remote])


def make_s3_reader(files: dict[str, bytes], fail_on: str | None = None) -> AwsS3Reader:
    reader = AwsS3Reader.__new__(AwsS3Reader)
    reader.fs = FakeS3FileSystem(files, fail_on=fail_on)
    return reader


def test_s3_reader_reports_aggregate_progress_and_atomically_replaces_files(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    remote_root = AwsS3Reader.remote_root.rstrip("/")
    remote_files = {
        f"{remote_root}/demo/cached.bin": b"abc",
        f"{remote_root}/demo/download.bin": b"123456",
    }
    reader = make_s3_reader(remote_files)
    (tmp_path / "cached.bin").write_bytes(b"abc")
    progress = []
    ticks = iter(range(100))
    monkeypatch.setattr(data_source.time, "monotonic", lambda: next(ticks))

    reader.read(
        "demo",
        ["cached.bin", "download.bin"],
        tmp_path,
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )

    assert (tmp_path / "download.bin").read_bytes() == b"123456"
    assert not (tmp_path / "download.bin.part").exists()
    assert reader.fs.destinations == [tmp_path / "download.bin.part"]
    assert progress[0][:2] == (3, 9)
    assert progress[-1][:2] == (9, 9)
    assert all(total == 9 for _, total, _ in progress)
    assert [current for current, _, _ in progress] == sorted(current for current, _, _ in progress)
    assert any(3 < current < 9 and message == "Downloading download.bin" for current, _, message in progress)


def test_s3_reader_removes_partial_and_invalid_final_on_failure(tmp_path: pathlib.Path):
    remote = f"{AwsS3Reader.remote_root.rstrip('/')}/demo/download.bin"
    reader = make_s3_reader({remote: b"complete payload"}, fail_on=remote)
    final_file = tmp_path / "download.bin"
    final_file.write_bytes(b"bad")

    with pytest.raises(RuntimeError, match="simulated download failure"):
        reader.read("demo", ["download.bin"], tmp_path)

    assert not final_file.exists()
    assert not (tmp_path / "download.bin.part").exists()


def test_s3_reader_rejects_wrong_sized_partial(tmp_path: pathlib.Path):
    remote = f"{AwsS3Reader.remote_root.rstrip('/')}/demo/download.bin"
    reader = make_s3_reader({remote: b"payload"})

    def write_short_file(_remote: str, local: str, callback: data_source.Callback):
        callback.set_size(7)
        pathlib.Path(local).write_bytes(b"short")
        callback.relative_update(5)

    reader.fs.get_file = write_short_file

    with pytest.raises(OSError, match="not match with remote size"):
        reader.read("demo", ["download.bin"], tmp_path)

    assert not (tmp_path / "download.bin").exists()
    assert not (tmp_path / "download.bin.part").exists()


def test_aliyun_reader_accepts_progress_callback(tmp_path: pathlib.Path):
    remote = "benchmark/demo/data.bin"
    reader = AliyunOSSReader.__new__(AliyunOSSReader)
    reader.bucket = FakeAliyunBucket({remote: b"payload"})
    progress = []

    reader.read(
        "demo",
        ["data.bin"],
        tmp_path,
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )

    assert (tmp_path / "data.bin").read_bytes() == b"payload"
    assert progress[-1] == (7, 7, "Downloaded data.bin")


def test_ir_reader_accepts_progress_callback(tmp_path: pathlib.Path):
    loaded = []
    reader = IRDatasetsReader.__new__(IRDatasetsReader)
    reader.ir_datasets = type("FakeIRDatasets", (), {"load": lambda _self, dataset: loaded.append(dataset)})()
    progress = []

    reader.read(
        "beir/test",
        [],
        tmp_path,
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )

    assert loaded == ["beir/test"]
    assert progress == [
        (0, 0, "Loading dataset metadata for beir/test"),
        (0, 0, "Dataset metadata ready for beir/test"),
    ]


class TestReader:
    @pytest.mark.parametrize("type_case", [
        (k, v) for k, v in type2case.items()
    ])
    def test_type_cases(self, type_case):
        self.per_case_test(type_case)


    def per_case_test(self, type_case):
        t, ca_cls = type_case
        ca = ca_cls()
        log.info(f"test case: {t.name}, {ca.name}")

        filters = ca.filter_rate
        ca.dataset.prepare(source=DatasetSource.AliyunOSS, filters=filters)
        ali_trains = ca.dataset.train_files

        ca.dataset.prepare(filters=filters)
        s3_trains = ca.dataset.train_files

        assert ali_trains == s3_trains
