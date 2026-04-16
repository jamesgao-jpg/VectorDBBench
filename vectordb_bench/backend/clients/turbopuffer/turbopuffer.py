"""Wrapper around the TurboPuffer vector database over VectorDB"""

import logging
import os
import time
from contextlib import contextmanager

import turbopuffer as tpuf

from vectordb_bench.backend.clients.turbopuffer.config import TurboPufferIndexConfig
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB

log = logging.getLogger(__name__)

# Env-var controlled per-stage timing for diagnosing backpressure / indexer behavior.
# Set STAGE_ROWS=200000 to log a line every 200k rows inserted.
STAGE_ROWS = int(os.environ.get("STAGE_ROWS", "0"))

# Set DISABLE_BACKPRESSURE=1 to pass disable_backpressure=True on every write,
# recommended by turbopuffer for bulk loads (avoids 429 stalls above 2 GiB unindexed).
# Tradeoff: strongly consistent queries fail above the threshold during the load.
DISABLE_BACKPRESSURE = os.environ.get("DISABLE_BACKPRESSURE", "0") == "1"

# Set TPUF_MAX_RETRIES=0 to disable SDK auto-retry of 429s; any RateLimitError
# will surface as an exception (caught by our handler and logged) instead of
# being silently retried with backoff. Useful for verifying backpressure hits.
TPUF_MAX_RETRIES = os.environ.get("TPUF_MAX_RETRIES")

# Set TPUF_EVENTUAL_CONSISTENCY=1 to pass consistency={"level":"eventual"} on
# every query. Avoids 429s on reads when the namespace has a large unindexed
# backlog, at the cost of only seeing indexed data + first 128 MiB of unindexed.
TPUF_EVENTUAL_CONSISTENCY = os.environ.get("TPUF_EVENTUAL_CONSISTENCY", "0") == "1"

# Set TPUF_SKIP_CACHE_WARM=1 to make optimize() a no-op (no hint_cache_warm,
# no sleep). Required for cold-search tests so the first query reads from
# object storage and reports true cold latency.
TPUF_SKIP_CACHE_WARM = os.environ.get("TPUF_SKIP_CACHE_WARM", "0") == "1"

# Set RETURN_VECTOR=1 to include the raw vector field in every search response
# (include_attributes=["vector"]). Used by the recall_field_query experiment to
# measure the latency/QPS overhead of returning payload with each hit. Default
# off preserves prior id-only behavior byte-for-byte.
RETURN_VECTOR = os.environ.get("RETURN_VECTOR", "0") == "1"


class TurboPuffer(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TurboPufferIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.api_key = db_config.get("api_key", "")
        self.region = db_config.get("region", "")
        self.api_base_url = db_config.get("api_base_url")
        self.namespace = db_config.get("namespace", "")
        self.db_case_config = db_case_config
        self.metric = db_case_config.parse_metric()

        self._vector_field = "vector"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"

        self.with_scalar_labels = with_scalar_labels

        # Per-stage insertion timing state (active only when STAGE_ROWS > 0).
        self._stage_rows = STAGE_ROWS
        self._stage_total = 0
        self._stage_start_time = None
        self._stage_idx = 0

        if drop_old:
            log.info(f"Drop old. delete the namespace: {self.namespace}")
            tmp_client = self._create_client()
            ns = tmp_client.namespace(self.namespace)
            try:
                ns.delete_all()
            except Exception as e:
                log.warning(f"Failed to delete all. Error: {e}")
            tmp_client = None

    def _create_client(self) -> tpuf.Turbopuffer:
        client_kwargs = {"api_key": self.api_key, "region": self.region}
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url
        if TPUF_MAX_RETRIES is not None:
            client_kwargs["max_retries"] = int(TPUF_MAX_RETRIES)
        return tpuf.Turbopuffer(**client_kwargs)

    @contextmanager
    def init(self):
        self.client = self._create_client()
        self.ns = self.client.namespace(self.namespace)
        yield

    def optimize(self, data_size: int | None = None):
        if TPUF_SKIP_CACHE_WARM:
            log.info("TPUF_SKIP_CACHE_WARM=1 set; skipping hint_cache_warm() and warmup sleep")
            return
        # turbopuffer responds to the request
        #   once the cache warming operation has been started.
        # It does not wait for the operation to complete,
        #   which can take multiple minutes for large namespaces.
        self.ns.hint_cache_warm()
        log.info(f"warming up but no api waiting for complete. just sleep {self.db_case_config.time_wait_warmup}s")
        time.sleep(self.db_case_config.time_wait_warmup)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        try:
            write_kwargs = {"distance_metric": self.metric}
            if DISABLE_BACKPRESSURE:
                write_kwargs["disable_backpressure"] = True
            if self.with_scalar_labels:
                self.ns.write(
                    upsert_columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: embeddings,
                        self._scalar_label_field: labels_data,
                    },
                    **write_kwargs,
                )
            else:
                self.ns.write(
                    upsert_columns={
                        self._scalar_id_field: metadata,
                        self._vector_field: embeddings,
                    },
                    **write_kwargs,
                )
        except tpuf.RateLimitError as e:
            log.warning(f"429 RateLimitError on insert (rows={len(embeddings)}): {e}")
        except tpuf.APIStatusError as e:
            status = getattr(e, "status_code", None)
            log.warning(f"APIStatusError status={status} on insert (rows={len(embeddings)}): {e}")
        except Exception as e:
            log.warning(f"Failed to insert. Error: {e}")

        # Per-stage timing: log a stage summary line each time cumulative row count
        # crosses a multiple of STAGE_ROWS.
        if self._stage_rows > 0:
            now = time.perf_counter()
            if self._stage_start_time is None:
                self._stage_start_time = now
            self._stage_total += len(embeddings)
            if self._stage_total >= (self._stage_idx + 1) * self._stage_rows:
                dur = now - self._stage_start_time
                rate = (self._stage_total - self._stage_idx * self._stage_rows) / dur if dur > 0 else 0.0
                log.info(
                    f"STAGE {self._stage_idx + 1}: rows={self._stage_total} "
                    f"stage_dur={dur:.2f}s stage_rate={rate:.1f} rows/s"
                )
                self._stage_idx += 1
                self._stage_start_time = now

        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        query_kwargs = {
            "rank_by": ("vector", "ANN", query),
            "top_k": k,
            "filters": self.expr,
        }
        if TPUF_EVENTUAL_CONSISTENCY:
            query_kwargs["consistency"] = {"level": "eventual"}
        if RETURN_VECTOR:
            query_kwargs["include_attributes"] = [self._vector_field]
        res = self.ns.query(**query_kwargs)
        return [int(row.id) for row in res.rows] if res.rows is not None else []

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        elif filters.type == FilterOp.NumGE:
            self.expr = (self._scalar_id_field, "Gte", filters.int_value)
        elif filters.type == FilterOp.StrEqual:
            self.expr = (self._scalar_label_field, "Eq", filters.label_value)
        else:
            msg = f"Not support Filter for TurboPuffer - {filters}"
            raise ValueError(msg)
