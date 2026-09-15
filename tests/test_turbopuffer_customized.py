from types import SimpleNamespace
from typing import Any

import pytest

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.clients.turbopuffer.turbopuffer import TurboPuffer
from vectordb_bench.backend.customized import CustomizedRequest, CustomizedRow, FieldSchema


def _db(namespace: Any) -> TurboPuffer:
    db = TurboPuffer.__new__(TurboPuffer)
    db.ns = namespace
    db.expr = None
    db.metric = "cosine_distance"
    db._is_fts = False
    db._scalar_id_field = "id"
    db._text_field = "text"
    db._vector_field = "vector"
    db.db_case_config = TurboPufferIndexConfig(metric_type=MetricType.COSINE)
    return db


def _wide_schema() -> dict[str, FieldSchema]:
    return {
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


def test_turbopuffer_inserts_complete_customized_schema() -> None:
    class Namespace:
        kwargs = None

        def write(self, **kwargs):
            self.kwargs = kwargs

    namespace = Namespace()
    db = _db(namespace)
    row = CustomizedRow(
        id="wide-table-0",
        fields={
            "emb_768": [0.0] * 768,
            "content": "cold namespace query",
            "i32_region": 1,
            "f64_price": 1.5,
            "bool_active": True,
            "vc_uuid": "2fd58409-3d75-4b16-a797-7757e49a59d9",
            "vc_tag": None,
            "vc_desc": "description",
            "bluesky_json": '{"did":"example"}',
            "arr_str_labels": ["a", "b"],
            "meta_json": '{"dyn_source":"sample"}',
        },
    )

    assert db.insert_customized_rows([row], _wide_schema()) == (1, None)
    assert namespace.kwargs["upsert_columns"]["id"] == ["wide-table-0"]
    assert namespace.kwargs["schema"]["emb_768"] == {"type": "[768]f32", "ann": True}
    assert namespace.kwargs["schema"]["content"]["full_text_search"] is True
    assert namespace.kwargs["schema"]["arr_str_labels"]["type"] == "[]string"
    assert namespace.kwargs["schema"]["meta_json"]["filterable"] is False
    assert namespace.kwargs["distance_metric"] == "cosine_distance"


def test_turbopuffer_searches_dense_and_bm25_with_performance() -> None:
    class Namespace:
        calls = []

        def query(self, **kwargs):
            self.calls.append(kwargs)
            field = kwargs["rank_by"][0]
            return SimpleNamespace(
                rows=[{"id": f"{field}-1", field: "returned"}],
                performance={
                    "cache_hit_ratio": 0.25,
                    "cache_temperature": "cold",
                    "server_total_ms": 12,
                    "query_execution_ms": 9,
                },
            )

    namespace = Namespace()
    db = _db(namespace)
    results = db.search_customized_queries(
        [
            CustomizedRequest("dense", "emb_768", [0.0] * 768, include_fields=("content",)),
            CustomizedRequest("bm25", "content", "cold namespace", include_fields=("content",)),
        ]
    )

    assert namespace.calls[0]["rank_by"][:2] == ("emb_768", "ANN")
    assert namespace.calls[1]["rank_by"] == ("content", "BM25", "cold namespace")
    assert results[0].ids == ["emb_768-1"]
    assert results[1].fields == {"content": ["returned"]}
    assert results[1].performance.cache_hit_ratio == 0.25
    assert results[1].performance.server_total_ms == 12.0


def test_search_documents_accepts_customized_text_field() -> None:
    class Namespace:
        kwargs = None

        def query(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(rows=[{"id": "result-1"}], performance=None)

    namespace = Namespace()
    db = _db(namespace)

    assert db.search_documents("cold namespace", field_name="content") == ["result-1"]
    assert namespace.kwargs["rank_by"] == ("content", "BM25", "cold namespace")


def test_customized_rows_reject_schema_drift() -> None:
    namespace = SimpleNamespace(write=lambda **_kwargs: None)
    db = _db(namespace)
    row = CustomizedRow(id=1, fields={"content": "text", "unexpected": 1})

    with pytest.raises(ValueError, match=r"extra=\['unexpected'\]"):
        db.insert_customized_rows([row], {"content": FieldSchema("string")})


def test_turbopuffer_selects_and_checks_customized_namespace() -> None:
    class Namespace:
        def __init__(self, name: str):
            self.name = name

        def exists(self) -> bool:
            return self.name == "existing"

    class Client:
        def namespace(self, name: str) -> Namespace:
            return Namespace(name)

    db = _db(Namespace("default"))
    db.client = Client()

    assert db.supports_namespace_selection()
    assert db.namespace_exists("existing")
    assert not db.namespace_exists("new")
    db.select_namespace("new")
    assert db.namespace == "new"
    assert db.ns.name == "new"
