import json
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vectordb_bench.backend.turbopuffer_multitenant import (
    NAMESPACE_PROFILE_COUNTS,
    PREPARED_SCHEMA,
    MultiTenantSetupRunner,
    NamespaceGroup,
    PreparedMultiTenantDataset,
    namespace_groups,
    namespace_profile,
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


def _write_queries(path: Path, count: int = 3) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "count": count,
                "dense_field": "emb_768",
                "bm25_field": "content",
                "queries": [
                    {"index": index, "dense": [float(index + 100)] * 768, "bm25": f"query-{index}"}
                    for index in range(count)
                ],
            }
        )
    )


class _FakeDB:
    def __init__(self):
        self.active_namespace = None
        self.rows = {}
        self.insert_calls = 0
        self.fail_once_for = None
        self.last_schema = None

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
        self.last_schema = schema
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


def test_namespace_profiles_keep_row_shapes_and_bound_source_rows() -> None:
    expected_counts = {"small": 20, "medium": 100, "large": 300}

    assert NAMESPACE_PROFILE_COUNTS == expected_counts
    for profile, count in expected_counts.items():
        groups = namespace_groups(profile)
        assert [group.rows_per_namespace for group in groups] == [1_000, 3_000, 15_000, 5_000_000]
        assert [group.namespace_count for group in groups] == [count, count, count, 1]
        assert max(group.source_rows for group in groups) <= 5_000_000
        assert namespace_profile(groups) == profile

        no_5m = namespace_groups(profile, include_5m=False)
        assert [group.key for group in no_5m] == ["A", "B", "C"]
        assert [group.rows_per_namespace for group in no_5m] == [1_000, 3_000, 15_000]
        assert [group.namespace_count for group in no_5m] == [count, count, count]
        assert namespace_profile(no_5m) == f"{profile}-no-5m"

    with pytest.raises(ValueError, match="small, medium, or large"):
        namespace_groups("unknown")


def test_setup_partitions_reused_rows_and_resumes_completed_namespaces(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared)
    queries = tmp_path / "queries.json"
    _write_queries(queries, count=2)
    dataset = PreparedMultiTenantDataset(prepared, _groups())
    db = _FakeDB()
    manifest = tmp_path / "setup.json"
    runner = MultiTenantSetupRunner(
        db, dataset, manifest, "run", queries_file=queries, batch_size=3, max_retries=0
    )

    summary = runner.run()

    assert summary["inserted_rows"] == 8
    assert summary["profile"] == "custom"
    assert summary["total_rows"] == 8
    assert summary["completed_namespaces"] == 3
    assert summary["queries"] == 2
    assert list(db.rows["run_2_01"]) == [0, 1]
    assert list(db.rows["run_2_02"]) == [2, 3]
    assert list(db.rows["run_4_01"]) == [0, 1, 2, 3]
    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["version"] == 5
    assert len(manifest_data["namespaces"]) == 3
    assert manifest_data["profile"] == "custom"
    assert manifest_data["search_fields"] == {"dense": "emb_768", "bm25": "content"}
    assert manifest_data["queries_file"] == "setup.queries.json"
    assert manifest_data["query_count"] == 2
    assert sorted(manifest_data["search_order"]) == ["run_2_01", "run_2_02", "run_4_01"]
    queries_sidecar = json.loads((tmp_path / "setup.queries.json").read_text())
    assert queries_sidecar["count"] == 2
    assert queries_sidecar["queries"][0]["dense"][0] == 100.0
    assert queries_sidecar["queries"][0]["bm25"] == "query-0"

    calls_before_resume = db.insert_calls
    resumed = runner.run()
    assert resumed["inserted_rows"] == 0
    assert resumed["completed_namespaces"] == 3
    assert db.insert_calls == calls_before_resume


def test_setup_can_select_the_manifest_bm25_field(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared, rows=2)
    queries = tmp_path / "queries.json"
    _write_queries(queries, count=2)
    db = _FakeDB()
    manifest = tmp_path / "setup.json"

    MultiTenantSetupRunner(
        db,
        PreparedMultiTenantDataset(prepared, (NamespaceGroup("A", "2", 2, 1, 2),)),
        manifest,
        "run",
        queries_file=queries,
        bm25_field="vc_desc",
        max_retries=0,
    ).run()

    assert db.last_schema["content"].full_text_search is False
    assert db.last_schema["vc_desc"].full_text_search is True
    assert json.loads(manifest.read_text())["search_fields"]["bm25"] == "vc_desc"


def test_setup_resumes_started_namespace_with_idempotent_ids(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared, rows=2)
    queries = tmp_path / "queries.json"
    _write_queries(queries, count=2)
    group = (NamespaceGroup("A", "2", 2, 1, 2),)
    dataset = PreparedMultiTenantDataset(prepared, group)
    db = _FakeDB()
    db.fail_once_for = "run_2_01"
    runner = MultiTenantSetupRunner(
        db,
        dataset,
        tmp_path / "setup.json",
        "run",
        queries_file=queries,
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
    queries = tmp_path / "queries.json"
    _write_queries(queries, count=2)
    group = (NamespaceGroup("A", "2", 2, 1, 2),)
    db = _FakeDB()
    db.rows["run_2_01"] = {}
    runner = MultiTenantSetupRunner(
        db,
        PreparedMultiTenantDataset(prepared, group),
        tmp_path / "setup.json",
        "run",
        queries_file=queries,
        max_retries=0,
    )

    with pytest.raises(FileExistsError, match="refusing existing namespace"):
        runner.run()


def test_setup_requires_a_queries_file(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared, rows=5)

    with pytest.raises(ValueError, match="requires a shared queries file"):
        MultiTenantSetupRunner(
            _FakeDB(),
            PreparedMultiTenantDataset(prepared, _groups()),
            tmp_path / "setup.json",
            "run",
        )


def test_setup_rejects_invalid_queries_file(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    _write_prepared(prepared, rows=5)
    queries = tmp_path / "queries.json"
    _write_queries(queries, count=2)
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "version": 1,
                "count": 1,
                "dense_field": "emb_768",
                "bm25_field": "content",
                "queries": [{"index": 0, "dense": [1.0], "bm25": "query-0"}],
            }
        )
    )

    with pytest.raises(ValueError, match="invalid query entry"):
        MultiTenantSetupRunner(
            _FakeDB(),
            PreparedMultiTenantDataset(prepared, _groups()),
            tmp_path / "setup.json",
            "run",
            queries_file=bad,
        )

    assert not (tmp_path / "setup.json").exists()
    assert not (tmp_path / "setup.queries.json").exists()
