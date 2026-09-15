import json
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vectordb_bench.backend.turbopuffer_multitenant import (
    PREPARED_SCHEMA,
    MultiTenantSetupRunner,
    NamespaceGroup,
    PreparedMultiTenantDataset,
)


def _write_prepared(path: Path, rows: int = 5) -> None:
    pq.write_table(
        pa.Table.from_pydict(
            {
                "id": list(range(rows)),
                "emb_768": [[float(index)] * 768 for index in range(rows)],
                "content": [f"content-{index}" for index in range(rows)],
                "i32_region": list(range(rows)),
                "f64_price": [float(index) for index in range(rows)],
                "bool_active": [index % 2 == 0 for index in range(rows)],
                "vc_uuid": [f"uuid-{index}" for index in range(rows)],
                "vc_tag": [None if index % 2 else f"tag-{index}" for index in range(rows)],
                "vc_desc": [f"description-{index}" for index in range(rows)],
                "bluesky_json": ['{"kind":"sample"}'] * rows,
                "arr_str_labels": [["sample", str(index)] for index in range(rows)],
                "meta_json": ['{"dyn_source":"test"}'] * rows,
            },
            schema=PREPARED_SCHEMA,
        ),
        path,
    )


class _FakeDB:
    def __init__(self):
        self.active_namespace = None
        self.rows = {}
        self.insert_calls = 0
        self.fail_once_for = None

    @classmethod
    def supports_customized_api(cls) -> bool:
        return True

    @classmethod
    def supports_namespace_selection(cls) -> bool:
        return True

    @contextmanager
    def init(self):
        yield

    def namespace_exists(self, namespace: str) -> bool:
        return namespace in self.rows

    def select_namespace(self, namespace: str) -> None:
        self.active_namespace = namespace

    def insert_customized_rows(self, rows, schema):  # noqa: ANN001
        assert self.active_namespace is not None
        assert set(schema) == {
            "emb_768",
            "content",
            "i32_region",
            "f64_price",
            "bool_active",
            "vc_uuid",
            "vc_tag",
            "vc_desc",
            "bluesky_json",
            "arr_str_labels",
            "meta_json",
        }
        self.insert_calls += 1
        namespace_rows = self.rows.setdefault(self.active_namespace, {})
        for row in rows:
            namespace_rows[row.id] = row
        if self.fail_once_for == self.active_namespace:
            self.fail_once_for = None
            return 0, RuntimeError("injected failure")
        return len(rows), None


def _groups() -> tuple[NamespaceGroup, ...]:
    return (
        NamespaceGroup("A", "2", 2, 2, 2),
        NamespaceGroup("B", "4", 4, 1, 2),
    )


def test_setup_partitions_reused_rows_and_resumes_completed_namespaces(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared)
    dataset = PreparedMultiTenantDataset(prepared, _groups())
    db = _FakeDB()
    manifest = tmp_path / "setup.json"
    runner = MultiTenantSetupRunner(db, dataset, manifest, "run", batch_size=3, max_retries=0)

    summary = runner.run()

    assert summary["inserted_rows"] == 8
    assert summary["completed_namespaces"] == 3
    assert list(db.rows["run_2_01"]) == [0, 1]
    assert list(db.rows["run_2_02"]) == [2, 3]
    assert list(db.rows["run_4_01"]) == [0, 1, 2, 3]
    manifest_data = json.loads(manifest.read_text())
    assert len(manifest_data["namespaces"]) == 3
    assert sorted(manifest_data["search_order"]) == ["run_2_01", "run_2_02", "run_4_01"]
    fixture = json.loads((tmp_path / "setup.fixtures/run_2_02.json").read_text())
    assert fixture["id"] == 2
    assert fixture["dense"]["value"] == [2.0] * 768
    assert fixture["bm25"] == {"field": "content", "value": "content-2"}

    calls_before_resume = db.insert_calls
    resumed = runner.run()
    assert resumed["inserted_rows"] == 0
    assert resumed["completed_namespaces"] == 3
    assert db.insert_calls == calls_before_resume


def test_setup_resumes_started_namespace_with_idempotent_ids(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared, rows=2)
    group = (NamespaceGroup("A", "2", 2, 1, 2),)
    dataset = PreparedMultiTenantDataset(prepared, group)
    db = _FakeDB()
    db.fail_once_for = "run_2_01"
    runner = MultiTenantSetupRunner(
        db,
        dataset,
        tmp_path / "setup.json",
        "run",
        batch_size=2,
        max_retries=0,
    )

    with pytest.raises(RuntimeError, match="customized insert failed"):
        runner.run()
    assert list(db.rows["run_2_01"]) == [0, 1]

    summary = runner.run()
    assert summary["inserted_rows"] == 2
    assert summary["completed_namespaces"] == 1
    assert list(db.rows["run_2_01"]) == [0, 1]


def test_setup_refuses_untracked_existing_namespace(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared, rows=2)
    group = (NamespaceGroup("A", "2", 2, 1, 2),)
    db = _FakeDB()
    db.rows["run_2_01"] = {}
    runner = MultiTenantSetupRunner(
        db,
        PreparedMultiTenantDataset(prepared, group),
        tmp_path / "setup.json",
        "run",
        max_retries=0,
    )

    with pytest.raises(FileExistsError, match="refusing existing namespace"):
        runner.run()
