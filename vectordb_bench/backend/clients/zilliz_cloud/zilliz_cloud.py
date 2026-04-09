"""Wrapper around the ZillizCloud vector database over VectorDB"""

import logging
import time

from ..api import DBCaseConfig
from ..milvus.milvus import Milvus

log = logging.getLogger(__name__)


class ZillizCloud(Milvus):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVDBBench",
        drop_old: bool = False,
        name: str = "ZillizCloud",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
            name=name,
            **kwargs,
        )

    def _optimize(self):
        """Zilliz Cloud denies GetPersistentSegmentInfo API.
        Skip _wait_for_segments_sorted, keep everything else."""
        log.info(f"{self.name} optimizing before search")
        try:
            self.client.flush(self.collection_name)
            # Skip _wait_for_segments_sorted — denied on Zilliz Cloud
            self._wait_for_index()
            try:
                compaction_id = self.client.compact(self.collection_name, target_size=(2**63 - 1))
                if compaction_id > 0:
                    self._wait_for_compaction(compaction_id)
                log.info(f"{self.name} force merge compaction completed.")
                self._wait_for_index()
            except Exception as e:
                log.warning(f"{self.name} compact error: {e}, skipping")
            self.client.refresh_load(self.collection_name)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None
