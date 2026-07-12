import logging
from datetime import datetime
from pathlib import Path

import pytest

from vectordb_bench import config
from vectordb_bench.backend.clients import DB, IndexType
from vectordb_bench.models import (
    CaseConfig,
    CaseResult,
    CaseType,
    Metric,
    TaskConfig,
    TestResult,
)

log = logging.getLogger("vectordb_bench")


class TestModels:
    def test_test_result_reads_iso_timestamp(self, tmp_path: Path):
        timestamp = "2026-07-09T08:02:27.004381+00:00"
        result_file = tmp_path / "result_iso_timestamp.json"
        result_file.write_text(
            '{"run_id":"iso-timestamp","task_label":"iso-timestamp","results":[],"timestamp":"'
            f'{timestamp}"}}'
        )

        result = TestResult.read_file(result_file)

        assert result.timestamp == pytest.approx(datetime.fromisoformat(timestamp).timestamp())

    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.Milvus,
                db_config=DB.Milvus.config(),
                db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
                case_config=CaseConfig(case_id=CaseType.Performance10M),
            ),
            metrics=Metric(),
        )

        test_result = TestResult(run_id=10000, results=[result])
        test_result.flush()

        with pytest.raises(ValueError):
            result = TestResult.read_file('nosuchfile.json')

    def test_test_result_read_write(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            res = TestResult.read_file(json_file)
            res.flush()

    def test_test_result_merge(self):
        result_dir = config.RESULTS_LOCAL_DIR
        all_results = []

        first_result = None
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)

            for cr in res.results:
                all_results.append(cr)

            if not first_result:
                first_result = res

        tr = TestResult(
            run_id=first_result.run_id,
            task_label="standard",
            results=all_results,
        )
        tr.flush()

    def test_test_result_display(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            log.info(json_file)
            res = TestResult.read_file(json_file)
            res.display()
