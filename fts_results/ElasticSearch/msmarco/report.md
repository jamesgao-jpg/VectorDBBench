# ElasticSearch MS MARCO FTS E2E Report

## Prequsites

- Backend: Elasticsearch single-node container, invoked through VectorDBBench `elasticcloudhnsw`.
- Dataset family: MS MARCO.
- Current committed raw results: `MS MARCO Small (100K documents)`, `MS MARCO Medium (1M documents)`, and `MS MARCO Large (8.8M documents)` on the original `m5d.2xlarge` server and the later `r7i.4xlarge` server.
- Run dates represented here: 2026-05-28, 2026-06-01, 2026-06-02, 2026-06-03, and 2026-06-04.
- Source runbook: `docs/fts-backends/elasticsearch.md`.
- Raw result directory: `raw_results/`.
- Current result JSONs have connection fields masked by VectorDBBench.

### Physical Machine Stats

Client machine:

- EC2 type: `i8g.2xlarge`.
- OS: Ubuntu 22.04, Linux `6.8.0-1053-aws`, `aarch64`.
- CPU: 8 vCPU, Neoverse-V2, 1 thread per core.
- Memory: about 61 GiB RAM, no swap.
- Disk quota: `/dev/root` ext4, 485 GiB total, 102 GiB available at last check.
- Role: runs VectorDBBench from `/home/ubuntu/VectorDBBench`.

Original server machine:

- EC2 type: `m5d.2xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 8 vCPU, Intel Xeon Platinum 8175M, 4 physical cores, 2 threads per core.
- Memory: about 30 GiB usable RAM plus 31 GiB swap.
- Disk quota: root XFS filesystem 500 GiB total with 451 GiB available at last check; additional 279 GiB NVMe mounted at `/data`.
- Docker: overlay2 under `/var/lib/docker` on the root filesystem.
- Role: runs the Elasticsearch container.

Rerun server machine:

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Elasticsearch container for the `r7i` rerun.

<!-- BEGIN 20260621 MATH GT SERVER STATS -->

Math-GT rerun server machine:

- EC2 type: `i8g.4xlarge`.
- OS: Ubuntu 22.04, Linux `6.8.0-1057-aws`, `aarch64`.
- CPU: 16 vCPU, Neoverse-V2, 1 thread per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: `/dev/root` ext4, 485 GiB total, 462 GiB available at verification.
- Role: runs fresh Milvus, Elasticsearch, and Vespa server deployments for the 2026-06-21 math-GT ids-only rerun.

<!-- END 20260621 MATH GT SERVER STATS -->

## Server Setup

Validated deployment:

- Elasticsearch image: `docker.elastic.co/elasticsearch/elasticsearch:8.16.0`.
- Container name: `es01`.
- Docker memory limit: `-m 8g`.
- JVM heap: no explicit `ES_JAVA_OPTS` is set in the current baseline; Elasticsearch 8 auto-sizes from the container limit.
- JVM heap confirmed on the `r7i` rerun: `-Xms4096m -Xmx4096m`.
- Docker volume: `esdata01`.
- Security: disabled for isolated private benchmark networking.

### FTS Index, Analyzer, And Ranking Configuration

Elasticsearch used a plain text mapping for FTS. VDBBench explicitly configured the mapping and index-level settings, but did not configure a custom analyzer, search analyzer, similarity, stopword list, or analysis block.

Effective mapping:

- `doc_id`: `keyword`.
- `text`: `text`.

Effective index settings:

- `number_of_shards=1`.
- `number_of_replicas=0`.
- `refresh_interval=30s`.
- force merge enabled by VDBBench after load.

Inherited Elasticsearch product defaults:

- default analyzer for `text`: `standard`.
- standard analyzer behavior: standard tokenizer, lowercase filter, stop filter disabled by default.
- default similarity: BM25.
- BM25 defaults: `k1=1.2`, `b=0.75`, `discount_overlaps=true`.
- FTS query shape: `match` query against the `text` field; the query text is analyzed with the field analyzer.

The `elasticcloudhnsw` CLI command name is misleading for these FTS results. When `--case-type FTSmsmarcoPerformance` is selected, VDBBench replaces the vector/HNSW case config with `ElasticCloudFtsConfig`, so HNSW parameters such as `m`, `ef_construction`, and vector `num_candidates` were not used for FTS. The relevant code paths are `ElasticCloudFtsConfig` in `vectordb_bench/backend/clients/elastic_cloud/config.py`, index creation in `vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py`, and FTS search via Elasticsearch `match` query in the same client.

Reproducible fresh-deploy script:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo sysctl -w vm.max_map_count=1048576

sudo docker rm -f es01 >/dev/null 2>&1 || true
sudo docker volume rm esdata01 >/dev/null 2>&1 || true
sudo docker volume create esdata01
sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:8.16.0

sudo docker run -d --name es01 \
  -p 0.0.0.0:9200:9200 \
  --restart unless-stopped \
  --ulimit nofile=65535:65535 \
  -m 8g \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -v esdata01:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.16.0

curl -fsS "http://127.0.0.1:9200/_cluster/health?pretty&wait_for_status=yellow&timeout=90s"
```

Fresh teardown script used after the `r7i` rerun:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f es01
sudo docker volume rm esdata01
sudo docker ps -a
sudo docker volume ls
```

## VDBBench Running

Exact client script for the committed MS MARCO Small runs:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "${SERVER_HOST}" \
  --port "9200" \
  --task-label "fts-e2e-elastic-msmarco-small" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The stability rerun used the same command with task label `fts-e2e-elastic-msmarco-small-stability`.
The `r7i.4xlarge` rerun used the same command with task label `fts-e2e-elastic-msmarco-small-r7i`.
The `r7i.4xlarge` medium run used the same command with task label `fts-e2e-elastic-msmarco-medium-r7i` and dataset size `MS MARCO Medium (1M documents)`.

Exact client script for the `r7i.4xlarge` MS MARCO Small text-payload run:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"

python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --host "${SERVER_HOST}" \
  --port 9200 \
  --task-label fts-e2e-elastic-msmarco-small-text-r7i \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --payload-profile text \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 --concurrency-duration 30 \
  --num-concurrency "1,10,20,40,60,80" \
  --concurrency-timeout 3600
```

Exact client script for the `r7i.4xlarge` MS MARCO Medium ids-only and text-payload matrix:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_TAG="20260604T041648Z"

for PAYLOAD_PROFILE in ids_only text; do
  if [[ "${PAYLOAD_PROFILE}" == "ids_only" ]]; then
    LABEL_PAYLOAD="ids"
  else
    LABEL_PAYLOAD="text"
  fi

  python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
    --host "${SERVER_HOST}" \
    --port 9200 \
    --task-label "fts-msmarco-medium-elastic-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "MS MARCO Medium (1M documents)" \
    --payload-profile "${PAYLOAD_PROFILE}" \
    --drop-old --load --search-serial --search-concurrent \
    --k 100 --concurrency-duration 30 \
    --num-concurrency "1,10,20,40,60,80" \
    --concurrency-timeout 3600
done
```

Exact client script for the `r7i.4xlarge` MS MARCO Large ids-only and text-payload matrix:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench_fts_ver1_run
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_TAG="20260604T113624Z"

for PAYLOAD_PROFILE in ids_only text; do
  if [[ "${PAYLOAD_PROFILE}" == "ids_only" ]]; then
    LABEL_PAYLOAD="ids"
    PAYLOAD_ARGS=()
  else
    LABEL_PAYLOAD="text"
    PAYLOAD_ARGS=(--payload-profile text)
  fi

  python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
    --host "${SERVER_HOST}" \
    --port 9200 \
    --task-label "fts-msmarco-large-elastic-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "MS MARCO Large (8.8M documents)" \
    "${PAYLOAD_ARGS[@]}" \
    --drop-old --load --search-serial --search-concurrent \
    --k 100 --concurrency-duration 30 \
    --num-concurrency "1,10,20,40,60,80" \
    --concurrency-timeout 3600
done
```

Effective Elasticsearch FTS case config from the raw JSON:

- `number_of_shards=1`
- `number_of_replicas=0`
- `refresh_interval=30s`
- `use_force_merge=true`
- `use_ssl=false`
- `verify_certs=true`

<!-- BEGIN 20260621 MATH GT VDBBENCH SCRIPT -->

Sanitized client script for the 2026-06-21 math-GT ids-only rerun:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_STAMP=20260621T150656Z
export CONCURRENCY=1,10,20,40,60,80
export CONCURRENCY_DURATION=30
export CONCURRENCY_TIMEOUT=3600
export LOAD_CONCURRENCY=0
export K=100
export PAYLOAD_PROFILE=ids_only

DATASET_LABELS=(
  "MS MARCO Small (100K documents)"
  "MS MARCO Medium (1M documents)"
  "MS MARCO Large (8.8M documents)"
)

for DATASET_LABEL in "${DATASET_LABELS[@]}"; do
  case "${DATASET_LABEL}" in
    *Small*) SIZE_KEY=small ;;
    *Medium*) SIZE_KEY=medium ;;
    *Large*) SIZE_KEY=large ;;
  esac
  TASK_LABEL="fts-matrix-elastic-msmarco-${SIZE_KEY}-ids-mathgt-${RUN_STAMP}"

  python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
  --scheme http \
  --host "${SERVER_HOST}" \
  --port 9200 \
  --password "<elastic-password>" \
    --task-label "${TASK_LABEL}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "${DATASET_LABEL}" \
    --payload-profile "${PAYLOAD_PROFILE}" \
    --drop-old --load --search-serial --search-concurrent \
    --load-concurrency "${LOAD_CONCURRENCY}" \
    --k "${K}" \
    --concurrency-duration "${CONCURRENCY_DURATION}" \
    --num-concurrency "${CONCURRENCY}" \
    --concurrency-timeout "${CONCURRENCY_TIMEOUT}"
done
```

<!-- END 20260621 MATH GT VDBBENCH SCRIPT -->

## Result

| Task label | Dataset size | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/5/10/20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fts-e2e-elastic-msmarco-small` | 100K | 72.4312 | 1227.3373 | 0.9116 | 0.7159 | 0.6665 | 0.0050 | 0.0098 | 143.5116 / 365.3610 / 672.9976 / 1227.3373 |
| `fts-e2e-elastic-msmarco-small` | 100K | 59.2031 | 3100.5973 | 0.9118 | 0.7159 | 0.6665 | 0.0031 | 0.0040 | 422.7782 / 1967.8125 / 2861.4449 / 3100.5973 |
| `fts-e2e-elastic-msmarco-small-stability` | 100K | 58.9212 | 3113.2707 | 0.9118 | 0.7159 | 0.6665 | 0.0031 | 0.0039 | 416.5153 / 1991.3322 / 2823.6226 / 3113.2707 |
| `fts-e2e-elastic-msmarco-small-r7i` | 100K | 59.4276 | 8689.3499 | 0.9118 | 0.7159 | 0.6665 | 0.0030 | 0.0035 | 396.5015 / 2534.0129 / 5536.5659 / 8689.3499 |
| `fts-e2e-elastic-msmarco-medium-r7i` | 1M | 148.5277 | 3986.3734 | 0.8028 | 0.5222 | 0.4526 | 0.0065 | 0.0089 | 240.5419 / 1348.5987 / 2719.8093 / 3986.3734 |

Latest `r7i.4xlarge` rerun vs previous `m5d.2xlarge` stability run:

- QPS increased from `3113.2707` to `8689.3499` (+179.1%).
- Load duration changed from `58.9212s` to `59.4276s` (+0.9%).
- Recall stayed unchanged at `0.9118`.
- p95 changed from `0.0031s` to `0.0030s`; p99 changed from `0.0039s` to `0.0035s`.

MS MARCO Small text payload rerun on `r7i.4xlarge`:

| Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fts-e2e-elastic-msmarco-small-text-r7i` | 57.8052 | 4177.1357 | 0.9118 | 0.7159 | 0.6665 | 0.0046 | 0.0051 | 242.0833 / 2599.9459 / 3941.7042 / 4158.8964 / 4177.1357 / 4155.8592 |

Text payload details:

- `payload_profile=text`.
- Returned fields: document ID fields plus `_source.text`.
- Estimated payload bytes per query from VectorDBBench: `53200`.

MS MARCO Medium six-concurrency rerun on `r7i.4xlarge`:

| Payload | Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ids_only` | `fts-msmarco-medium-elastic-ids-c1-10-20-40-60-80-r7i-20260604T041648Z` | 140.1544 | 4473.8674 | 0.8028 | 0.5222 | 0.4526 | 0.0063 | 0.0086 | 260.6360 / 2883.9739 / 4166.7860 / 4405.5505 / 4473.8674 / 4458.2345 |
| `text` | `fts-msmarco-medium-elastic-text-c1-10-20-40-60-80-r7i-20260604T041648Z` | 139.6663 | 2696.5048 | 0.8028 | 0.5222 | 0.4526 | 0.0079 | 0.0101 | 178.1539 / 1787.4005 / 2605.4203 / 2688.4985 / 2680.9282 / 2696.5048 |

MS MARCO Medium stability comparison against the previous `r7i.4xlarge` ids-only run:

- At the shared concurrency level `20`, QPS changed from `3986.3734` to `4166.7860` (+4.5%).
- The new six-concurrency run reached a higher peak QPS of `4473.8674` at concurrency `60` (+12.2% vs the previous peak at concurrency `20`).
- Load duration changed from `148.5277s` to `140.1544s` (-5.6%).
- Recall stayed unchanged at `0.8028`.
- Text payload QPS was `2696.5048`, which is 39.7% below the new ids-only rerun at the same concurrency list.

MS MARCO Large six-concurrency run on `r7i.4xlarge`:

| Payload | Task label | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrent QPS at 1/10/20/40/60/80 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ids_only` | `fts-msmarco-large-elastic-ids-c1-10-20-40-60-80-r7i-20260604T113624Z` | 991.9284 | 1279.2869 | 0.6230 | 0.2733 | 0.1862 | 0.0243 | 0.0389 | 92.4571 / 899.7783 / 1217.7168 / 1279.2869 / 1240.2423 / 1202.4733 |
| `text` | `fts-msmarco-large-elastic-text-c1-10-20-40-60-80-r7i-20260604T113624Z` | 966.5160 | 952.0335 | 0.6230 | 0.2733 | 0.1862 | 0.0278 | 0.0432 | 11.7464 / 311.8365 / 942.3865 / 952.0335 / 947.4759 / 921.7487 |

MS MARCO Large observations:

- ElasticSearch ids-only peaked at concurrency `40`; higher concurrency reduced QPS slightly.
- Text-payload QPS was `952.0335`, which is 25.6% below ids-only at the same concurrency list.
- Recall stayed unchanged at `0.6230`.

<!-- BEGIN 20260616 SEMANTIC QREL RERUN -->

### 2026-06-16 i8g semantic/qrel rerun

These rows are preserved as the last completed pre-math-GT rerun on the `i8g.4xlarge` server. They use the legacy IR-dataset qrel/semantic recall path and were produced before the 2026-06-21 parquet math-GT changes. Do not compare their recall values as the same recall contract as the `math GT 2026-06-21` rows.

Run context:

- Server environment: `i8g.4xlarge`, Ubuntu 22.04, `aarch64`, 16 vCPU, about 123 GiB RAM.
- Client branch/head: `fts_impl_only@81905ec4cef9c5f85b2f20136dce15e8f911e013`.
- Dataset source and ground truth path: `ir_datasets`; logs show `Loaded ground truth ... into memory` before the parquet ground-truth implementation landed.
- Load concurrency: `8`.
- Search concurrency: `1,10,20,40,60,80`.
- Scope: completed ElasticSearch `MS MARCO Large (8.8M documents)` ids-only and text-payload runs only.

Sanitized client command shape:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/home/ubuntu/VectorDBBench/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_STAMP=20260616T112000Z
export LOAD_CONCURRENCY=8
export CONCURRENCY=1,10,20,40,60,80

for PAYLOAD_PROFILE in ids_only text; do
  if [[ "${PAYLOAD_PROFILE}" == "ids_only" ]]; then
    LABEL_PAYLOAD=ids_only
    PAYLOAD_ARGS=()
  else
    LABEL_PAYLOAD=text
    PAYLOAD_ARGS=(--payload-profile text)
  fi

  python3.11 -m vectordb_bench.cli.vectordbbench elasticcloudhnsw \
    --scheme http \
    --host "${SERVER_HOST}" \
    --port 9200 \
    --password "<elastic-password>" \
    --task-label "fts_rerun_elastic_msmarco_large_${LABEL_PAYLOAD}_${RUN_STAMP}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "MS MARCO Large (8.8M documents)" \
    "${PAYLOAD_ARGS[@]}" \
    --drop-old --load --search-serial --search-concurrent \
    --load-concurrency "${LOAD_CONCURRENCY}" \
    --k 100 \
    --concurrency-duration 30 \
    --num-concurrency "${CONCURRENCY}" \
    --concurrency-timeout 3600
done
```

| Task label | Payload | Load s | Insert s | Optimize s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `fts_rerun_elastic_msmarco_large_ids_only_20260616T112000Z` | ids_only | 360.3714 | 238.7995 | 121.5719 | 1789.5592 | 0.6228 | 0.0000 | n/a | 0.0268 | 0.0433 | 1/10/20/40/60/80 | 90.7488 / 1018.0375 / 1623.0575 / 1717.3223 / 1789.5592 / 1767.1579 |
| `fts_rerun_elastic_msmarco_large_text_20260616T112000Z` | text | 334.1155 | 243.1559 | 90.9596 | 1253.0604 | 0.6230 | 0.0000 | n/a | 0.0307 | 0.0474 | 1/10/20/40/60/80 | 9.9555 / 144.4961 / 1140.3133 / 1226.0161 / 1251.2690 / 1253.0604 |

<!-- END 20260616 SEMANTIC QREL RERUN -->

<!-- BEGIN 20260621 MATH GT IDS ONLY -->

### 2026-06-21 Math-GT ids-only rerun

These rows use generated BM25 mathematical ground truth from `neighbors.parquet` instead of IR dataset qrels. The primary quality metric for this rerun is recall; the current JSONs emit `ndcg=0.0` and no `mrr` field. The JSON `inserted_count` field is `null`, so inserted count is intentionally not reported here.

| Dataset size | Task label | Payload | Load s | Insert s | Optimize s | QPS | Recall | NDCG | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MS MARCO Small (100K documents) | `fts-matrix-elastic-msmarco-small-ids-mathgt-20260621T150656Z` | ids_only | 34.5372 | 3.5575 | 30.9797 | 12792.5025 | 0.9416 | 0.0000 | 0.0023 | 0.0028 | 1/10/20/40/60/80 | 601.7296 / 5910.0014 / 9375.6363 / 11945.9017 / 12685.7963 / 12792.5025 |
| MS MARCO Medium (1M documents) | `fts-matrix-elastic-msmarco-medium-ids-mathgt-20260621T150656Z` | ids_only | 64.7575 | 34.0057 | 30.7518 | 5852.4564 | 0.9366 | 0.0000 | 0.0062 | 0.0089 | 1/10/20/40/60/80 | 332.6692 / 3155.8439 / 4885.9889 / 5796.7833 / 5852.4564 / 5812.2195 |
| MS MARCO Large (8.8M documents) | `fts-matrix-elastic-msmarco-large-ids-mathgt-20260621T150656Z` | ids_only | 439.0781 | 348.8843 | 90.1938 | 1713.7078 | 0.9304 | 0.0000 | 0.0283 | 0.0462 | 1/10/20/40/60/80 | 91.7264 / 1036.2800 / 1599.7248 / 1711.8981 / 1713.7078 / 1711.1540 |

<!-- END 20260621 MATH GT IDS ONLY -->
