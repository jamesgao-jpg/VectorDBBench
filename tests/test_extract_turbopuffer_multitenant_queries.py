import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.extract_turbopuffer_multitenant_queries import extract_queries


def _write_source(path: Path, rows: int = 5) -> None:
    pq.write_table(
        pa.Table.from_pydict(
            {
                "emb_768": [[float(index)] * 768 for index in range(rows)],
                "content": [f"content-{index}" for index in range(rows)],
                "other": list(range(rows)),
            }
        ),
        path,
    )


def test_extract_writes_the_first_rows_as_queries(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=5)

    queries = extract_queries(source, count=3, dense_field="emb_768", bm25_field="content")

    assert queries["version"] == 1
    assert queries["count"] == 3
    assert queries["dense_field"] == "emb_768"
    assert queries["bm25_field"] == "content"
    assert [entry["index"] for entry in queries["queries"]] == [0, 1, 2]
    assert queries["queries"][0]["dense"] == [0.0] * 768
    assert queries["queries"][1]["bm25"] == "content-1"


def test_extract_rejects_bad_dense_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "emb_768": [[1.0, 2.0]],
                "content": ["text"],
            }
        ),
        source,
    )

    with pytest.raises(ValueError, match="768-dimensional"):
        extract_queries(source, count=1, dense_field="emb_768", bm25_field="content")


def test_extract_rejects_missing_columns_or_too_few_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pydict({"emb_768": [[1.0] * 768]}), source)

    with pytest.raises(ValueError, match="must declare emb_768 and content columns"):
        extract_queries(source, count=1, dense_field="emb_768", bm25_field="content")

    with pytest.raises(ValueError, match="1 rows, but 2 queries"):
        extract_queries(source, count=2, dense_field="emb_768", bm25_field="content")


def test_extract_json_output_round_trips_through_loader(tmp_path: Path) -> None:
    from scripts.extract_turbopuffer_multitenant_queries import write_json
    from vectordb_bench.backend.turbopuffer_multitenant import load_queries_file

    source = tmp_path / "source.parquet"
    _write_source(source, rows=5)
    output = tmp_path / "queries.json"
    write_json(output, extract_queries(source, count=2, dense_field="emb_768", bm25_field="content"))

    loaded = load_queries_file(output)
    assert [query.index for query in loaded] == [0, 1]
    assert len(loaded[0].dense) == 768
    assert loaded[1].bm25 == "content-1"


def test_extract_cli_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.extract_turbopuffer_multitenant_queries import main

    source = tmp_path / "source.parquet"
    _write_source(source, rows=3)
    output = tmp_path / "queries.json"
    monkeypatch.setattr(
        "sys.argv",
        ["extract", "--input", str(source), "--output", str(output), "--count", "2"],
    )

    main()

    assert json.loads(output.read_text())["count"] == 2
    assert not output.with_name("queries.json.partial").exists()
