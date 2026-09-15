from pathlib import Path
from shutil import copyfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_turbopuffer_multitenant_data import OUTPUT_SCHEMA, prepare_dataset


def _source_table(rows: int, *, dimensions: int = 768) -> pa.Table:
    schema = pa.schema(
        [
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
        ]
    )
    return pa.Table.from_pydict(
        {
            "emb_768": [[float(index)] * dimensions for index in range(rows)],
            "content": [f"content-{index}" for index in range(rows)],
            "i32_region": list(range(rows)),
            "f64_price": [float(index) for index in range(rows)],
            "bool_active": [index % 2 == 0 for index in range(rows)],
            "vc_uuid": [f"uuid-{index}" for index in range(rows)],
            "vc_tag": [None if index % 2 else f"tag-{index}" for index in range(rows)],
            "vc_desc": [f"description-{index}" for index in range(rows)],
            "bluesky_json": ['{"kind":"sample"}'] * rows,
            "arr_str_labels": [["sample", str(index)] for index in range(rows)],
            "$meta": ['{"dyn_source":"test"}'] * rows,
        },
        schema=schema,
    )


class _FakeS3:
    def __init__(self, objects: dict[str, Path]):
        self.objects = objects
        self.downloaded = []

    def find(self, _root: str) -> list[str]:
        return list(reversed(self.objects))

    def info(self, remote_path: str) -> dict[str, int]:
        return {"size": self.objects[remote_path].stat().st_size}

    def download(self, remote_path: str, local_path: str) -> None:
        self.downloaded.append(remote_path)
        copyfile(self.objects[remote_path], local_path)


def test_prepare_dataset_downloads_enough_files_and_writes_exact_row_count(tmp_path: Path) -> None:
    objects = {}
    for index in range(3):
        source = tmp_path / f"source-{index}.parquet"
        pq.write_table(_source_table(2), source)
        objects[f"bucket/prefix/wide_table_{index:04}.parquet"] = source
    filesystem = _FakeS3(objects)
    output = tmp_path / "prepared.parquet"

    summary = prepare_dataset(
        "s3://bucket/prefix/",
        tmp_path / "downloads",
        output,
        target_rows=3,
        batch_rows=2,
        filesystem=filesystem,
    )

    assert filesystem.downloaded == [
        "bucket/prefix/wide_table_0000.parquet",
        "bucket/prefix/wide_table_0001.parquet",
    ]
    assert summary["source_files"] == 2
    assert summary["downloaded_rows"] == 4
    prepared = pq.read_table(output)
    assert prepared.schema == OUTPUT_SCHEMA
    assert prepared.num_rows == 3
    assert prepared.column("id").to_pylist() == [0, 1, 2]
    assert prepared.column("content").to_pylist() == ["content-0", "content-1", "content-0"]
    assert prepared.column("meta_json").to_pylist() == [
        '{"dyn_source":"test"}',
        '{"dyn_source":"test"}',
        '{"dyn_source":"test"}',
    ]
    assert "$meta" not in prepared.column_names


def test_prepare_dataset_rejects_wrong_vector_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_source_table(1, dimensions=767), source)
    remote_path = "bucket/prefix/wide_table_0000.parquet"

    with pytest.raises(ValueError, match="dimension other than 768"):
        prepare_dataset(
            "s3://bucket/prefix/",
            tmp_path / "downloads",
            tmp_path / "prepared.parquet",
            target_rows=1,
            filesystem=_FakeS3({remote_path: source}),
        )
