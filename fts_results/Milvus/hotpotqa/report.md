# Milvus HotpotQA FTS E2E Report

## Prequsites

- Backend: Milvus standalone.
- Dataset family: HotpotQA.
- Current committed raw results: `HotpotQA Medium (1M documents)`, historical `HotpotQA Large (5.2M documents)`, and `HotpotQA Large (5.2M documents)` matrix runs with ids-only and text payloads on the `r7i.4xlarge` server.
- Run dates represented here: 2026-06-02, 2026-06-03, and 2026-06-04.
- Source runbook: `docs/fts-backends/milvus.md`.
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
- Role: runs the Milvus standalone server deployment.

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

- Milvus image: `milvusdb/milvus:v2.6.17`.
- etcd image: `quay.io/coreos/etcd:v3.5.25`.
- MinIO image: `minio/minio:RELEASE.2024-12-18T13-15-44Z`.
- Deployment: official `milvus-standalone-docker-compose.yml`.
- MQ config: `MQ_TYPE=woodpecker` from the official `v2.6.17` compose file.
- Persistent data: `~/milvus-standalone/volumes`.

### FTS Index, Analyzer, And Ranking Configuration

Milvus did not use an implicit product-selected default index in these FTS runs. VDBBench explicitly created an analyzer-enabled `text` field, a generated `sparse_vector` field, and a Milvus BM25 function from `text` to `sparse_vector`. The searchable index was then created on `sparse_vector`.

Effective FTS index configuration:

- `index_type=SPARSE_INVERTED_INDEX`.
- `metric_type=BM25`.
- `inverted_index_algo=DAAT_MAXSCORE`.
- `bm25_k1=1.5`.
- `bm25_b=0.75`.
- `drop_ratio_search=null`, so no search-time sparse-vector drop ratio was applied.

Effective analyzer configuration:

- tokenizer: `standard`.
- lowercase: enabled.
- token length filter: max token length `40`.
- stop words: `null`.

This is a VDBBench-explicit FTS default, not a Milvus product default chosen by omission. The Milvus docs describe the same BM25 FTS shape, using a text field, BM25 function, sparse vector field, and sparse/BM25 index; `DAAT_MAXSCORE` is documented as the default inverted-index algorithm, but this benchmark still sent it explicitly. The relevant code paths are `MilvusFtsConfig` in `vectordb_bench/backend/clients/milvus/config.py`, collection/index creation in `vectordb_bench/backend/clients/milvus/milvus.py`, and search on `anns_field="sparse_vector"` with raw text queries.

Reproducible fresh-deploy script:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

export MILVUS_VERSION=v2.6.17
wget "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml" \
  -O docker-compose.yml

sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "300m",
    "max-file": "3"
  }
}
JSON

sudo systemctl restart docker
sudo docker compose pull
sudo docker compose down -v --remove-orphans || true
sudo rm -rf volumes
sudo docker compose up -d
sudo docker compose ps
curl -fsS http://127.0.0.1:9091/healthz
```

Fresh teardown script used after the run:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/milvus-standalone
sudo docker compose down -v --remove-orphans
sudo rm -rf volumes
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

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
  --task-label "fts-e2e-milvus-hotpotqa-medium-r7i" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Medium (1M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "1,5,10,20" \
  --concurrency-timeout 3600
```

The committed HotpotQA Large run used the same command with task label `fts-e2e-milvus-hotpotqa-large-r7i` and dataset size `HotpotQA Large (5.2M documents)`.

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

  python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
    --uri "http://${SERVER_HOST}:19530" \
    --task-label "fts-hotpotqa-medium-milvus-${LABEL_PAYLOAD}-c1-10-20-40-60-80-r7i-${RUN_TAG}" \
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

Exact client script for the 2026-06-03 HotpotQA Large matrix runs:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/tmp/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
  --task-label "fts-matrix-milvus-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "20,40,80" \
  --concurrency-timeout 3600

python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
  --task-label "fts-matrix-milvus-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z" \
  --case-type FTSmsmarcoPerformance \
  --dataset-with-size-type "HotpotQA Large (5.2M documents)" \
  --payload-profile text \
  --drop-old --load --search-serial --search-concurrent \
  --k 100 \
  --concurrency-duration 30 \
  --num-concurrency "20,40,80" \
  --concurrency-timeout 3600
```

Effective Milvus FTS case config from the raw JSON:

- `index_type=SPARSE_INVERTED_INDEX`
- `metric_type=BM25`
- `inverted_index_algo=DAAT_MAXSCORE`
- `bm25_k1=1.5`
- `bm25_b=0.75`
- `analyzer_tokenizer=standard`
- `analyzer_enable_lowercase=true`
- `analyzer_max_token_length=40`
- `num_shards=1`
- `replica_number=1`

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
  TASK_LABEL="fts-matrix-milvus-hotpotqa-${SIZE_KEY}-ids-mathgt-${RUN_STAMP}"

  python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
  --uri "http://${SERVER_HOST}:19530" \
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

| Task label | Dataset size | Payload | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `fts-e2e-milvus-hotpotqa-medium-r7i` | 1M | ids_only | 2031.2796 | 1596.6340 | 0.8378 | 0.7246 | 0.8561 | 0.0123 | 0.0170 | 1/5/10/20 | 146.0629 / 726.2209 / 1273.3380 / 1596.6340 |
| `fts-hotpotqa-medium-milvus-ids-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | ids_only | 2040.9336 | 1865.4681 | 0.8378 | 0.7246 | 0.8561 | 0.0122 | 0.0170 | 1/10/20/40/60/80 | 255.0087 / 1364.3522 / 1378.9975 / 1702.7745 / 1851.3955 / 1865.4681 |
| `fts-hotpotqa-medium-milvus-text-c1-10-20-40-60-80-r7i-20260604T074646Z` | 1M | text | 2033.2594 | 1714.0357 | 0.8378 | 0.7246 | 0.8561 | 0.0124 | 0.0170 | 1/10/20/40/60/80 | 223.9828 / 1224.1637 / 1558.9074 / 1669.1467 / 1687.2785 / 1714.0357 |
| `fts-e2e-milvus-hotpotqa-large-r7i` | 5.2M | ids_only | 10583.8485 | 394.4417 | 0.7573 | 0.6129 | 0.7410 | 0.0212 | 0.0299 | 1/5/10/20 | 88.1695 / 336.8579 / 388.2553 / 394.4417 |
| `fts-matrix-milvus-hotpotqa-large-ids-c20-40-80-r7i-20260603T061706Z` | 5.2M | ids_only | 10583.8402 | 411.7323 | 0.7573 | 0.6129 | 0.7410 | 0.0211 | 0.0305 | 20/40/80 | 400.7550 / 407.1847 / 411.7323 |
| `fts-matrix-milvus-hotpotqa-large-text-c20-40-80-r7i-20260603T061706Z` | 5.2M | text | 10583.7873 | 409.4366 | 0.7573 | 0.6129 | 0.7410 | 0.0214 | 0.0308 | 20/40/80 | 395.3148 / 407.5527 / 409.4366 |

<!-- BEGIN 20260616 SEMANTIC QREL RERUN -->

### 2026-06-16 i8g semantic/qrel rerun

These rows are preserved as the last completed pre-math-GT rerun on the `i8g.4xlarge` server. They use the legacy IR-dataset qrel/semantic recall path and were produced before the 2026-06-21 parquet math-GT changes. Do not compare their recall values as the same recall contract as the `math GT 2026-06-21` rows.

Run context:

- Server environment: `i8g.4xlarge`, Ubuntu 22.04, `aarch64`, 16 vCPU, about 123 GiB RAM.
- Client branch/head: `fts_impl_only@81905ec4cef9c5f85b2f20136dce15e8f911e013`.
- Dataset source and ground truth path: `ir_datasets`; logs show `Loaded ground truth ... into memory` before the parquet ground-truth implementation landed.
- Load concurrency: `8`.
- Search concurrency: `1,10,20,40,60,80`.
- Scope: completed Milvus `HotpotQA Large (5.2M documents)` ids-only and text-payload runs only.

Sanitized client command shape:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/VectorDBBench
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
export RESULTS_LOCAL_DIR=/home/ubuntu/VectorDBBench/vectordb_bench/results
export NUM_PER_BATCH=100
export SERVER_HOST="<server-private-host-or-dns>"
export RUN_STAMP=20260616T101622Z
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

  python3.11 -m vectordb_bench.cli.vectordbbench milvusfts \
    --uri "http://${SERVER_HOST}:19530" \
    --task-label "fts_rerun_milvus_hotpotqa_large_${LABEL_PAYLOAD}_${RUN_STAMP}" \
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
| `fts_rerun_milvus_hotpotqa_large_ids_only_20260616T101622Z` | ids_only | 315.2651 | 103.8342 | 211.4308 | 782.8244 | 0.7673 | 0.0000 | n/a | 0.0450 | 0.0674 | 1/10/20/40/60/80 | 45.9053 / 471.7161 / 743.6551 / 769.3053 / 782.8244 / 779.0495 |
| `fts_rerun_milvus_hotpotqa_large_text_20260616T101622Z` | text | 303.2773 | 103.3764 | 199.9009 | 791.0526 | 0.7673 | 0.0000 | n/a | 0.0441 | 0.0659 | 1/10/20/40/60/80 | 47.1694 / 474.1473 / 749.5954 / 781.3073 / 789.7035 / 791.0526 |

<!-- END 20260616 SEMANTIC QREL RERUN -->

<!-- BEGIN 20260621 MATH GT IDS ONLY -->

### 2026-06-21 Math-GT ids-only rerun

These rows use generated BM25 mathematical ground truth from `neighbors.parquet` instead of IR dataset qrels. The primary quality metric for this rerun is recall; the current JSONs emit `ndcg=0.0` and no `mrr` field. The JSON `inserted_count` field is `null`, so inserted count is intentionally not reported here.

| Dataset size | Task label | Payload | Load s | Insert s | Optimize s | QPS | Recall | NDCG | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HotpotQA Small (100K documents) | `fts-matrix-milvus-hotpotqa-small-ids-mathgt-20260621T150656Z` | ids_only | 22.1836 | 2.5370 | 19.6467 | 8166.9534 | 0.9219 | 0.0000 | 0.0034 | 0.0042 | 1/10/20/40/60/80 | 432.6747 / 3883.8084 / 5972.5282 / 7139.4660 / 7824.2637 / 8166.9534 |
| HotpotQA Medium (1M documents) | `fts-matrix-milvus-hotpotqa-medium-ids-mathgt-20260621T150656Z` | ids_only | 90.3465 | 23.1773 | 67.1692 | 2266.5843 | 0.9179 | 0.0000 | 0.0142 | 0.0196 | 1/10/20/40/60/80 | 134.4163 / 1315.1920 / 2002.1491 / 2216.3537 / 2257.2126 / 2266.5843 |
| HotpotQA Large (5.2M documents) | `fts-matrix-milvus-hotpotqa-large-ids-mathgt-20260621T150656Z` | ids_only | 302.1242 | 116.4104 | 185.7137 | 784.9749 | 0.9127 | 0.0000 | 0.0450 | 0.0672 | 1/10/20/40/60/80 | 46.2159 / 477.6217 / 745.7581 / 776.9463 / 781.7123 / 784.9749 |

<!-- END 20260621 MATH GT IDS ONLY -->
