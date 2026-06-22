# ElasticSearch HotpotQA FTS E2E Report

## Prequsites

- Backend: Elasticsearch single-node container, invoked through VectorDBBench `elasticcloudhnsw`.
- Dataset family: HotpotQA.
- Current committed raw results: `HotpotQA Medium (1M documents)`, historical `HotpotQA Large (5.2M documents)`, and a `HotpotQA Large (5.2M documents)` text-payload matrix run on the `r7i.4xlarge` server.
- Run dates represented here: 2026-06-02 through 2026-06-04.
- Source runbook: `docs/fts-backends/elasticsearch.md`.
- Raw result directory: `raw_results/`.
- The current FTS CLI uses `FTSmsmarcoPerformance` as the generic FTS case type; the dataset is selected by `--dataset-with-size-type`.

### Physical Machine Stats

Client machine:

- EC2 type: `i8g.2xlarge`.
- OS: Ubuntu 22.04, Linux `6.8.0-1053-aws`, `aarch64`.
- CPU: 8 vCPU, Neoverse-V2, 1 thread per core.
- Memory: about 61 GiB RAM, no swap.
- Disk quota: `/dev/root` ext4, 485 GiB total, 102 GiB available at last check.
- Role: runs VectorDBBench from `/home/ubuntu/VectorDBBench`.

Server machine:

- EC2 type: `r7i.4xlarge`.
- OS: Amazon Linux 2023, Linux `6.1.55-75.123.amzn2023.x86_64`, `x86_64`.
- CPU: 16 vCPU, Intel Xeon Platinum 8488C, 8 physical cores, 2 threads per core.
- Memory: about 123 GiB RAM, no swap.
- Disk quota: root filesystem 500 GiB total with 491 GiB available after teardown.
- Docker: Docker `24.0.5`, Docker Compose `v2.27.0`.
- Role: runs the Elasticsearch container.

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
- JVM heap confirmed on the `r7i` run: `-Xms4096m -Xmx4096m`.
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

Fresh teardown script used after the run:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f es01
sudo docker volume rm esdata01
sudo docker ps -a
sudo docker volume ls
```

## VDBBench Running

Exact client script for the committed HotpotQA runs:

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
  --task-label "fts-e2e-elastic-hotpotqa-medium-r7i" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The committed HotpotQA Large run used the same command with task label `fts-e2e-elastic-hotpotqa-large-r7i` and dataset size `HotpotQA Large (5.2M documents)`.

Exact client script for the 2026-06-04 HotpotQA Large text-payload matrix run:

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
  --task-label "fts-matrix-elastic-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
  --payload-profile text \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "20,40,80" \
  --concurrency-timeout 3600
```

Exact client script for the 2026-06-04 HotpotQA Medium ids-only and text-payload matrix:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_TAG="20260604T074646Z"

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
    --port "9200" \
    --task-label "fts-hotpotqa-medium-elastic-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "HotpotQA Medium (1M documents)" \
    "${PAYLOAD_ARGS[@]}" \
    --drop-old --load --search-serial --search-concurrent \
    --k 100 \
    --concurrency-duration 30 \
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
  "HotpotQA Small (100K documents)"
  "HotpotQA Medium (1M documents)"
  "HotpotQA Large (5.2M documents)"
)

for DATASET_LABEL in "${DATASET_LABELS[@]}"; do
  case "${DATASET_LABEL}" in
    *Small*) SIZE_KEY=small ;;
    *Medium*) SIZE_KEY=medium ;;
    *Large*) SIZE_KEY=large ;;
  esac
  TASK_LABEL="fts-matrix-elastic-hotpotqa-${SIZE_KEY}-ids-mathgt-${RUN_STAMP}"

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

The ids-only matrix run `fts-matrix-elastic-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z` is intentionally excluded from the result table because VDBBench emitted only a zero-metric failure placeholder JSON. Log evidence shows the run loaded successfully (`load_duration=545.7043s`) and completed concurrency 20/40 (`447.5993 / 480.0184 QPS`), but the parent process hung after starting concurrency 80 and was terminated with `RUN_FAILED_143`.

| Task label | Dataset size | Payload | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `fts-e2e-elastic-hotpotqa-medium-r7i` | 1M | ids_only | 142.0589 | 1410.3787 | 0.8378 | 0.7287 | 0.8598 | 0.0159 | 0.0224 | 1/5/10/20 | 111.5252 / 558.2304 / 1034.1176 / 1410.3787 |
| `fts-hotpotqa-medium-elastic-ids-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | ids_only | 139.2256 | 1581.7165 | 0.8378 | 0.7287 | 0.8598 | 0.0150 | 0.0212 | 1/10/20/40/60/80 | 119.3122 / 1106.0982 / 1552.4142 / 1581.7165 / 1574.0491 / 1579.6451 |
| `fts-hotpotqa-medium-elastic-text-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | text | 140.1574 | 1238.7840 | 0.8378 | 0.7287 | 0.8598 | 0.0171 | 0.0237 | 1/10/20/40/60/80 | 94.8516 / 870.2786 / 1203.6183 / 1230.3996 / 1238.7840 / 1229.2721 |
| `fts-e2e-elastic-hotpotqa-large-r7i` | 5.2M | ids_only | 550.6164 | 476.2610 | 0.7637 | 0.6243 | 0.7549 | 0.0503 | 0.0755 | 1/5/10/20 | 41.0129 / 202.7703 / 356.3845 / 476.2610 |
| `fts-matrix-elastic-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z` | 5.2M | text | 554.4492 | 435.1027 | 0.7637 | 0.6243 | 0.7549 | 0.0518 | 0.0766 | 20/40/80 | 402.3090 / 435.1027 / 434.3993 |

<!-- BEGIN 20260616 SEMANTIC QREL RERUN -->

### 2026-06-16 i8g semantic/qrel rerun

These rows are preserved as the last completed pre-math-GT rerun on the `i8g.4xlarge` server. They use the legacy IR-dataset qrel/semantic recall path and were produced before the 2026-06-21 parquet math-GT changes. Do not compare their recall values as the same recall contract as the `math GT 2026-06-21` rows.

Run context:

- Server environment: `i8g.4xlarge`, Ubuntu 22.04, `aarch64`, 16 vCPU, about 123 GiB RAM.
- Client branch/head: `fts_impl_only@81905ec4cef9c5f85b2f20136dce15e8f911e013`.
- Dataset source and ground truth path: `ir_datasets`; logs show `Loaded ground truth ... into memory` before the parquet ground-truth implementation landed.
- Load concurrency: `8`.
- Search concurrency: `1,10,20,40,60,80`.
- Scope: completed ElasticSearch `HotpotQA Large (5.2M documents)` ids-only and text-payload runs only.

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
    --task-label "fts_rerun_elastic_hotpotqa_large_${LABEL_PAYLOAD}_${RUN_STAMP}" \
    --case-type FTSmsmarcoPerformance \
    --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
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
| `fts_rerun_elastic_hotpotqa_large_ids_only_20260616T112000Z` | ids_only | 211.0087 | 145.8880 | 65.1207 | 685.6198 | 0.7637 | 0.0000 | n/a | 0.0540 | 0.0804 | 1/10/20/40/60/80 | 41.9719 / 414.9978 / 650.3724 / 682.7910 / 683.6943 / 685.6198 |
| `fts_rerun_elastic_hotpotqa_large_text_20260616T112000Z` | text | 209.8852 | 144.7063 | 65.1788 | 602.5509 | 0.7637 | 0.0000 | n/a | 0.0573 | 0.0842 | 1/10/20/40/60/80 | 12.5968 / 337.1613 / 563.9477 / 590.6620 / 591.3812 / 602.5509 |

<!-- END 20260616 SEMANTIC QREL RERUN -->

<!-- BEGIN 20260621 MATH GT IDS ONLY -->

### 2026-06-21 Math-GT ids-only rerun

These rows use generated BM25 mathematical ground truth from `neighbors.parquet` instead of IR dataset qrels. The primary quality metric for this rerun is recall; the current JSONs emit `ndcg=0.0` and no `mrr` field. The JSON `inserted_count` field is `null`, so inserted count is intentionally not reported here.

| Dataset size | Task label | Payload | Load s | Insert s | Optimize s | QPS | Recall | NDCG | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HotpotQA Small (100K documents) | `fts-matrix-elastic-hotpotqa-small-ids-mathgt-20260621T150656Z` | ids_only | 34.3497 | 3.3234 | 31.0263 | 6461.6069 | 0.8750 | 0.0000 | 0.0045 | 0.0057 | 1/10/20/40/60/80 | 368.5025 / 3467.8430 / 5348.3790 / 6428.9145 / 6455.8945 / 6461.6069 |
| HotpotQA Medium (1M documents) | `fts-matrix-elastic-hotpotqa-medium-ids-mathgt-20260621T150656Z` | ids_only | 65.2426 | 34.2934 | 30.9491 | 2073.6098 | 0.8620 | 0.0000 | 0.0166 | 0.0237 | 1/10/20/40/60/80 | 123.4348 / 1226.2127 / 1918.1030 / 2057.6807 / 2073.6098 / 2072.1953 |
| HotpotQA Large (5.2M documents) | `fts-matrix-elastic-hotpotqa-large-ids-mathgt-20260621T150656Z` | ids_only | 256.6338 | 194.9027 | 61.7311 | 659.2480 | 0.8553 | 0.0000 | 0.0570 | 0.0851 | 1/10/20/40/60/80 | 38.6843 / 408.8350 / 622.3226 / 659.1277 / 656.1379 / 659.2480 |

<!-- END 20260621 MATH GT IDS ONLY -->
