# FTS E2E Results Master Report

This report compares the latest committed representative result for each backend and dataset configuration. Backend-specific historical reruns remain in each leaf `report.md`; this master view is for cross-backend comparison.

All local-server rows use the `r7i.4xlarge` server unless marked otherwise. TurboPuffer is a managed external backend, so its row is not directly comparable on server hardware. `Load s` is the VectorDBBench load duration. `QPS`, `p95 s`, and `p99 s` are from the search result in the raw JSON. Each row states its payload profile and concurrency list explicitly.

<!-- BEGIN 20260616 SEMANTIC QREL NOTE -->

Rows with context `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 are the completed pre-math-GT Milvus/ElasticSearch large-dataset reruns from `fts_impl_only@81905ec`. They use the legacy IR-dataset qrel/semantic recall path, so their recall values are not the same measurement contract as rows labeled `math GT 2026-06-21`. Vespa did not complete a corresponding committed result JSON in that rerun and is intentionally omitted.

<!-- END 20260616 SEMANTIC QREL NOTE -->

<!-- BEGIN 20260621 MATH GT NOTE -->

Rows with context `i8g.4xlarge`, math GT 2026-06-21 use generated BM25 mathematical ground truth from `neighbors.parquet` instead of IR dataset qrels. For those rows, recall is the primary quality metric; `NDCG` is the JSON-emitted `0.0`, and `MRR` is `n/a` because the result JSONs do not include MRR.

<!-- END 20260621 MATH GT NOTE -->

## Table of Contents

- [FTS Index And Ranking Configuration](#fts-index-and-ranking-configuration)
- [MS MARCO Small (100K documents)](#ms-marco-small-100k-documents)
- [MS MARCO Medium (1M documents)](#ms-marco-medium-1m-documents)
- [MS MARCO Large (8.8M documents)](#ms-marco-large-88m-documents)
- [HotpotQA Small (100K documents)](#hotpotqa-small-100k-documents)
- [HotpotQA Medium (1M documents)](#hotpotqa-medium-1m-documents)
- [HotpotQA Large (5.2M documents)](#hotpotqa-large-52m-documents)

## FTS Index And Ranking Configuration

The FTS runs did not use one uniform "backend default" configuration. Milvus and Vespa used explicit VDBBench FTS schemas/indexes. Elasticsearch used a plain text mapping with VDBBench index settings, then inherited Elasticsearch analyzer and BM25 defaults.

| Backend | Was Product Default? | Indexed Field | Index / Mapping | Analyzer / Linguistics | Ranking / Similarity | Explicit Args |
|---|---|---|---|---|---|---|
| Milvus | No. VDBBench explicitly configured the FTS sparse index and BM25 parameters. | Analyzer-enabled `text` field feeds generated `sparse_vector`. | `SPARSE_INVERTED_INDEX` on `sparse_vector`; BM25 function maps `text` to `sparse_vector`. | `standard` tokenizer, lowercase enabled, max token length `40`, stop words `null`. | `metric_type=BM25`; `bm25_k1=1.5`; `bm25_b=0.75`. | `inverted_index_algo=DAAT_MAXSCORE`; `drop_ratio_search=null`. |
| ElasticSearch | Partly. VDBBench explicitly configured mapping and index settings; analyzer and similarity were Elasticsearch defaults. | `text`. | Mapping: `doc_id: keyword`, `text: text`; standard Lucene inverted index for `text`. | Default `standard` analyzer: standard tokenizer, lowercase filter, stop filter disabled by default. | Default BM25 similarity: `k1=1.2`, `b=0.75`, `discount_overlaps=true`. | `number_of_shards=1`, `number_of_replicas=0`, `refresh_interval=30s`, force merge enabled. No HNSW/vector index settings were used for FTS. |
| Vespa | No. VDBBench explicitly deployed an FTS Vespa schema and rank profile. | `text`. | `text` is `string` with `index` and `summary`, plus `index: enable-bm25`; `id` is `summary` and `attribute`. | Vespa default string index text processing where not overridden, including tokenized text matching, normalization, and default stemming `best`. | Explicit rank profile `bm25` with first phase `bm25(text)`; BM25 parameters use Vespa defaults `k1=1.2`, `b=0.75`. | Query uses `userQuery()`, `ranking=bm25`, `default-index=text`, `type=any`; `VespaFtsConfig` has no extra index/search args, so raw JSON records `db_case_config={}`. |

Code references: Milvus FTS config is in `vectordb_bench/backend/clients/milvus/config.py` and index/search execution is in `vectordb_bench/backend/clients/milvus/milvus.py`. Elasticsearch FTS mapping/settings are in `vectordb_bench/backend/clients/elastic_cloud/config.py` and FTS search uses a `match` query in `vectordb_bench/backend/clients/elastic_cloud/elastic_cloud.py`. Vespa FTS schema and query construction are in `vectordb_bench/backend/clients/vespa/vespa.py`, with empty FTS case config in `vectordb_bench/backend/clients/vespa/config.py`.

Official docs used for product defaults: Milvus full text search, Elasticsearch standard analyzer and similarity docs, and Vespa BM25/schema/linguistics docs.

## MS MARCO Small (100K documents)

Text-payload rows used `payload_profile=text`, `k=100`, `concurrency_duration=30`, and explicit concurrency `1,10,20,40,60,80`. Local-server rows used the same `r7i.4xlarge` server host, recorded outside the repo in the local host config.

| Backend | Payload | Context | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Milvus | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 21.3279 | 11404.4883 | 0.9880 | 0.0000 | n/a | 0.0022 | 0.0027 | 1/10/20/40/60/80 | 624.4250 / 5402.9614 / 8562.6935 / 10240.8866 / 11019.5892 / 11404.4883 |
| ElasticSearch | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 34.5372 | 12792.5025 | 0.9416 | 0.0000 | n/a | 0.0023 | 0.0028 | 1/10/20/40/60/80 | 601.7296 / 5910.0014 / 9375.6363 / 11945.9017 / 12685.7963 / 12792.5025 |
| Vespa | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 27.6829 | 735.3580 | 0.7466 | 0.0000 | n/a | 0.0159 | 0.0196 | 1/10/20/40/60/80 | 104.8972 / 735.3580 / 514.5597 / 387.6114 / 676.5868 / 623.4075 |
| Milvus | ids_only | `r7i.4xlarge` | 230.3305 | 9359.8351 | 0.9157 | 0.7157 | 0.6653 | 0.0026 | 0.0029 | 1/5/10/20 | 528.3714 / 3129.5306 / 5750.1304 / 9359.8351 |
| Milvus | text | `r7i.4xlarge` | 230.4392 | 9569.0676 | 0.9157 | 0.7157 | 0.6653 | 0.0029 | 0.0032 | 1/10/20/40/60/80 | 468.2255 / 4857.3898 / 8011.7230 / 9279.3577 / 9569.0676 / 9266.8844 |
| ElasticSearch | ids_only | `r7i.4xlarge` | 59.4276 | 8689.3499 | 0.9118 | 0.7159 | 0.6665 | 0.0030 | 0.0035 | 1/5/10/20 | 396.5015 / 2534.0129 / 5536.5659 / 8689.3499 |
| ElasticSearch | text | `r7i.4xlarge` | 57.8052 | 4177.1357 | 0.9118 | 0.7159 | 0.6665 | 0.0046 | 0.0051 | 1/10/20/40/60/80 | 242.0833 / 2599.9459 / 3941.7042 / 4158.8964 / 4177.1357 / 4155.8592 |
| Vespa | ids_only | `r7i.4xlarge` | 79.2473 | 734.5241 | 0.9416 | 0.7509 | 0.7015 | 0.0184 | 0.0230 | 1/5/10/20 | 91.7244 / 512.5352 / 347.9730 / 734.5241 |
| Vespa | text | `r7i.4xlarge` | 78.8999 | 788.0555 | 0.9416 | 0.7509 | 0.7015 | 0.0193 | 0.0236 | 1/10/20/40/60/80 | 64.6508 / 786.3064 / 131.0499 / 788.0555 / 422.9538 / 365.3142 |
| TurboPuffer | ids_only | managed backend | 290.5625 | 257.3771 | 0.9125 | 0.7156 | 0.6659 | 0.0840 | 0.1081 | 1/5/10/20 | 1.3357 / 49.4967 / 126.8548 / 257.3771 |

## MS MARCO Medium (1M documents)

Rows below include the 2026-06-04 six-concurrency rerun and the 2026-06-21 math-GT ids-only rerun, both using explicit concurrency `1,10,20,40,60,80`. Older ids-only `1,5,10,20` baselines remain in the backend-specific reports for stability comparison.

| Backend | Payload | Context | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Milvus | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 77.0604 | 5662.2877 | 0.9896 | 0.0000 | n/a | 0.0062 | 0.0086 | 1/10/20/40/60/80 | 305.2597 / 2913.6627 / 4352.3168 / 5138.1525 / 5530.6816 / 5662.2877 |
| ElasticSearch | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 64.7575 | 5852.4564 | 0.9366 | 0.0000 | n/a | 0.0062 | 0.0089 | 1/10/20/40/60/80 | 332.6692 / 3155.8439 / 4885.9889 / 5796.7833 / 5852.4564 / 5812.2195 |
| Vespa | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 225.0491 | 305.7255 | 0.7298 | 0.0000 | n/a | 0.1068 | 0.1413 | 1/10/20/40/60/80 | 19.0626 / 199.0932 / 292.6319 / 296.4212 / 305.7255 / 302.4968 |
| Milvus | ids_only | `r7i.4xlarge` | 2048.1231 | 5139.5920 | 0.8048 | 0.5174 | 0.4458 | 0.0053 | 0.0071 | 1/10/20/40/60/80 | 433.1311 / 2976.3670 / 3973.3733 / 4750.5123 / 5053.7822 / 5139.5920 |
| Milvus | text | `r7i.4xlarge` | 2048.2360 | 4677.4078 | 0.8048 | 0.5174 | 0.4458 | 0.0057 | 0.0075 | 1/10/20/40/60/80 | 378.3115 / 2732.7863 / 3656.8277 / 4353.0352 / 4602.9011 / 4677.4078 |
| ElasticSearch | ids_only | `r7i.4xlarge` | 140.1544 | 4473.8674 | 0.8028 | 0.5222 | 0.4526 | 0.0063 | 0.0086 | 1/10/20/40/60/80 | 260.6360 / 2883.9739 / 4166.7860 / 4405.5505 / 4473.8674 / 4458.2345 |
| ElasticSearch | text | `r7i.4xlarge` | 139.6663 | 2696.5048 | 0.8028 | 0.5222 | 0.4526 | 0.0079 | 0.0101 | 1/10/20/40/60/80 | 178.1539 / 1787.4005 / 2605.4203 / 2688.4985 / 2680.9282 / 2696.5048 |
| Vespa | ids_only | `r7i.4xlarge` | 581.5774 | 257.0647 | 0.8409 | 0.5499 | 0.4767 | 0.1231 | 0.1688 | 1/10/20/40/60/80 | 17.2619 / 153.1753 / 217.6157 / 230.2074 / 238.9556 / 257.0647 |
| Vespa | text | `r7i.4xlarge` | 581.4244 | 251.4636 | 0.8409 | 0.5499 | 0.4767 | 0.1248 | 0.1702 | 1/10/20/40/60/80 | 15.0937 / 133.4407 / 199.3716 / 209.5499 / 234.2022 / 251.4636 |

## MS MARCO Large (8.8M documents)

Rows below include the 2026-06-04/2026-06-05 six-concurrency run and the 2026-06-21 math-GT ids-only rerun. All rows used `k=100`, `concurrency_duration=30`, and explicit concurrency `1,10,20,40,60,80`.

| Backend | Payload | Context | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Milvus | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 495.1239 | 1351.2833 | 0.9910 | 0.0000 | n/a | 0.0176 | 0.0269 | 1/10/20/40/60/80 | 124.6555 / 1045.9747 / 1226.9505 / 1289.9100 / 1343.4557 / 1351.2833 |
| ElasticSearch | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 439.0781 | 1713.7078 | 0.9304 | 0.0000 | n/a | 0.0283 | 0.0462 | 1/10/20/40/60/80 | 91.7264 / 1036.2800 / 1599.7248 / 1711.8981 / 1713.7078 / 1711.1540 |
| Vespa | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 2011.2197 | 194.8595 | 0.6907 | 0.0000 | n/a | 0.4443 | 0.4448 | 1/10/20/40/60/80 | 2.8601 / 38.1890 / 61.7164 / 108.3861 / 153.9399 / 194.8595 |
| Milvus | ids_only | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 463.7911 | 1309.2858 | 0.6279 | 0.0000 | n/a | 0.0188 | 0.0287 | 1/10/20/40/60/80 | 119.0172 / 1024.5452 / 1213.7577 / 1280.3891 / 1288.6748 / 1309.2858 |
| Milvus | text | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 630.4357 | 1636.3888 | 0.6279 | 0.0000 | n/a | 0.0243 | 0.0374 | 1/10/20/40/60/80 | 91.4653 / 925.6603 / 1435.8955 / 1625.9694 / 1630.2646 / 1636.3888 |
| ElasticSearch | ids_only | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 360.3714 | 1789.5592 | 0.6228 | 0.0000 | n/a | 0.0268 | 0.0433 | 1/10/20/40/60/80 | 90.7488 / 1018.0375 / 1623.0575 / 1717.3223 / 1789.5592 / 1767.1579 |
| ElasticSearch | text | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 334.1155 | 1253.0604 | 0.6230 | 0.0000 | n/a | 0.0307 | 0.0474 | 1/10/20/40/60/80 | 9.9555 / 144.4961 / 1140.3133 / 1226.0161 / 1251.2690 / 1253.0604 |
| Milvus | ids_only | `r7i.4xlarge` | 17874.1539 | 738.2857 | 0.6206 | 0.2695 | 0.1824 | 0.0091 | 0.0133 | 1/10/20/40/60/80 | 203.7730 / 701.5716 / 707.9453 / 722.6776 / 735.3773 / 738.2857 |
| Milvus | text | `r7i.4xlarge` | 17864.1118 | 743.7173 | 0.6206 | 0.2695 | 0.1824 | 0.0099 | 0.0140 | 1/10/20/40/60/80 | 189.9733 / 700.0473 / 692.9175 / 730.2865 / 735.8742 / 743.7173 |
| ElasticSearch | ids_only | `r7i.4xlarge` | 991.9284 | 1279.2869 | 0.6230 | 0.2733 | 0.1862 | 0.0243 | 0.0389 | 1/10/20/40/60/80 | 92.4571 / 899.7783 / 1217.7168 / 1279.2869 / 1240.2423 / 1202.4733 |
| ElasticSearch | text | `r7i.4xlarge` | 966.5160 | 952.0335 | 0.6230 | 0.2733 | 0.1862 | 0.0278 | 0.0432 | 1/10/20/40/60/80 | 11.7464 / 311.8365 / 942.3865 / 952.0335 / 947.4759 / 921.7487 |
| Vespa | ids_only | `r7i.4xlarge` | 4987.2328 | 192.5008 | 0.5689 | 0.2475 | 0.1673 | 0.4454 | 0.4460 | 1/10/20/40/60/80 | 4.1287 / 30.7389 / 56.8053 / 103.6033 / 149.3731 / 192.5008 |
| Vespa | text | `r7i.4xlarge` | 4989.0550 | 187.5886 | 0.5687 | 0.2475 | 0.1674 | 0.4525 | 0.4537 | 1/10/20/40/60/80 | 3.3315 / 22.3210 / 53.0334 / 100.4980 / 140.6589 / 187.5886 |

Vespa text completed but emitted timeout/docsum warnings at concurrency 60 and 80. The raw JSON is valid and included.

## HotpotQA Small (100K documents)

Rows below are the 2026-06-21 math-GT ids-only rerun using explicit concurrency `1,10,20,40,60,80`.

| Backend | Payload | Context | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Milvus | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 22.1836 | 8166.9534 | 0.9219 | 0.0000 | n/a | 0.0034 | 0.0042 | 1/10/20/40/60/80 | 432.6747 / 3883.8084 / 5972.5282 / 7139.4660 / 7824.2637 / 8166.9534 |
| ElasticSearch | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 34.3497 | 6461.6069 | 0.8750 | 0.0000 | n/a | 0.0045 | 0.0057 | 1/10/20/40/60/80 | 368.5025 / 3467.8430 / 5348.3790 / 6428.9145 / 6455.8945 / 6461.6069 |
| Vespa | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 21.7408 | 640.2794 | 0.3496 | 0.0000 | n/a | 0.0268 | 0.0317 | 1/10/20/40/60/80 | 55.3992 / 593.0678 / 506.3641 / 640.2794 / 573.0491 / 638.3529 |

## HotpotQA Medium (1M documents)

Rows below are the 2026-06-04 six-concurrency rerun using explicit concurrency `1,10,20,40,60,80`. Older ids-only `1,5,10,20` baselines and the previous failed Vespa text-payload attempt remain in the backend-specific reports.

| Backend | Payload | Context | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Milvus | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 90.3465 | 2266.5843 | 0.9179 | 0.0000 | n/a | 0.0142 | 0.0196 | 1/10/20/40/60/80 | 134.4163 / 1315.1920 / 2002.1491 / 2216.3537 / 2257.2126 / 2266.5843 |
| ElasticSearch | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 65.2426 | 2073.6098 | 0.8620 | 0.0000 | n/a | 0.0166 | 0.0237 | 1/10/20/40/60/80 | 123.4348 / 1226.2127 / 1918.1030 / 2057.6807 / 2073.6098 / 2072.1953 |
| Vespa | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 221.5715 | 186.5870 | 0.4818 | 0.0000 | n/a | 0.2124 | 0.2624 | 1/10/20/40/60/80 | 7.6518 / 81.2353 / 125.9907 / 131.0931 / 153.0614 / 186.5870 |
| Milvus | ids_only | `r7i.4xlarge` | 2040.9336 | 1865.4681 | 0.8378 | 0.7246 | 0.8561 | 0.0122 | 0.0170 | 1/10/20/40/60/80 | 255.0087 / 1364.3522 / 1378.9975 / 1702.7745 / 1851.3955 / 1865.4681 |
| Milvus | text | `r7i.4xlarge` | 2033.2594 | 1714.0357 | 0.8378 | 0.7246 | 0.8561 | 0.0124 | 0.0170 | 1/10/20/40/60/80 | 223.9828 / 1224.1637 / 1558.9074 / 1669.1467 / 1687.2785 / 1714.0357 |
| ElasticSearch | ids_only | `r7i.4xlarge` | 139.2256 | 1581.7165 | 0.8378 | 0.7287 | 0.8598 | 0.0150 | 0.0212 | 1/10/20/40/60/80 | 119.3122 / 1106.0982 / 1552.4142 / 1581.7165 / 1574.0491 / 1579.6451 |
| ElasticSearch | text | `r7i.4xlarge` | 140.1574 | 1238.7840 | 0.8378 | 0.7287 | 0.8598 | 0.0171 | 0.0237 | 1/10/20/40/60/80 | 94.8516 / 870.2786 / 1203.6183 / 1230.3996 / 1238.7840 / 1229.2721 |
| Vespa | ids_only | `r7i.4xlarge` | 579.2518 | 181.5240 | 0.8309 | 0.7208 | 0.8500 | 0.2628 | 0.3223 | 1/10/20/40/60/80 | 6.6652 / 58.1128 / 81.3796 / 104.5544 / 140.9124 / 181.5240 |
| Vespa | text | `r7i.4xlarge` | 579.3951 | 177.2947 | 0.8309 | 0.7208 | 0.8500 | 0.2683 | 0.3313 | 1/10/20/40/60/80 | 5.2029 / 55.8888 / 79.5598 / 100.9787 / 138.3337 / 177.2947 |

Vespa text completed in the 2026-06-04 rerun, but emitted backend timeout warnings during concurrency 60 and 80. The raw JSON is valid and included.

## HotpotQA Large (5.2M documents)

Historical matrix rows used explicit concurrency `20,40,80`; the 2026-06-21 math-GT ids-only rows used `1,10,20,40,60,80`. All rows used `k=100` and `concurrency_duration=30`. Payload `ids_only` returns ids only; payload `text` returns ids plus text payload.

| Backend | Payload | Context | Load s | QPS | Recall | NDCG | MRR | p95 s | p99 s | Concurrency | Concurrent QPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Milvus | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 302.1242 | 784.9749 | 0.9127 | 0.0000 | n/a | 0.0450 | 0.0672 | 1/10/20/40/60/80 | 46.2159 / 477.6217 / 745.7581 / 776.9463 / 781.7123 / 784.9749 |
| ElasticSearch | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 256.6338 | 659.2480 | 0.8553 | 0.0000 | n/a | 0.0570 | 0.0851 | 1/10/20/40/60/80 | 38.6843 / 408.8350 / 622.3226 / 659.1277 / 656.1379 / 659.2480 |
| Vespa | ids_only | `i8g.4xlarge`, math GT 2026-06-21 | 1187.6098 | 176.2716 | 0.5727 | 0.0000 | n/a | 0.4443 | 0.4450 | 1/10/20/40/60/80 | 2.4158 / 25.1704 / 48.4437 / 90.4134 / 132.7175 / 176.2716 |
| Milvus | ids_only | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 315.2651 | 782.8244 | 0.7673 | 0.0000 | n/a | 0.0450 | 0.0674 | 1/10/20/40/60/80 | 45.9053 / 471.7161 / 743.6551 / 769.3053 / 782.8244 / 779.0495 |
| Milvus | text | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 303.2773 | 791.0526 | 0.7673 | 0.0000 | n/a | 0.0441 | 0.0659 | 1/10/20/40/60/80 | 47.1694 / 474.1473 / 749.5954 / 781.3073 / 789.7035 / 791.0526 |
| ElasticSearch | ids_only | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 211.0087 | 685.6198 | 0.7637 | 0.0000 | n/a | 0.0540 | 0.0804 | 1/10/20/40/60/80 | 41.9719 / 414.9978 / 650.3724 / 682.7910 / 683.6943 / 685.6198 |
| ElasticSearch | text | `i8g.4xlarge`, semantic/qrel recall, 2026-06-16 | 209.8852 | 602.5509 | 0.7637 | 0.0000 | n/a | 0.0573 | 0.0842 | 1/10/20/40/60/80 | 12.5968 / 337.1613 / 563.9477 / 590.6620 / 591.3812 / 602.5509 |
| ElasticSearch | ids_only | `r7i.4xlarge` | 550.6164 | 476.2610 | 0.7637 | 0.6243 | 0.7549 | 0.0503 | 0.0755 | 1/5/10/20 | 41.0129 / 202.7703 / 356.3845 / 476.2610 |
| ElasticSearch | text | `r7i.4xlarge` | 554.4492 | 435.1027 | 0.7637 | 0.6243 | 0.7549 | 0.0518 | 0.0766 | 20/40/80 | 402.3090 / 435.1027 / 434.3993 |
| Milvus | ids_only | `r7i.4xlarge` | 10583.8485 | 394.4417 | 0.7573 | 0.6129 | 0.7410 | 0.0212 | 0.0299 | 1/5/10/20 | 88.1695 / 336.8579 / 388.2553 / 394.4417 |
| Milvus | ids_only | `r7i.4xlarge` | 10583.8402 | 411.7323 | 0.7573 | 0.6129 | 0.7410 | 0.0211 | 0.0305 | 20/40/80 | 400.7550 / 407.1847 / 411.7323 |
| Milvus | text | `r7i.4xlarge` | 10583.7873 | 409.4366 | 0.7573 | 0.6129 | 0.7410 | 0.0214 | 0.0308 | 20/40/80 | 395.3148 / 407.5527 / 409.4366 |
| Vespa | ids_only | `r7i.4xlarge` | 2954.2589 | 46.3472 | 0.6754 | 0.5460 | 0.6640 | 0.4460 | 0.4465 | 1/5/10/20 | 3.3531 / 13.1559 / 24.4787 / 46.3472 |

Excluded milestone run: `elastic/hotpotqa-large/ids` with concurrency `20,40,80` loaded successfully (`545.7043s`) and completed concurrency 20/40 (`447.5993 / 480.0184 QPS`), then hung after starting concurrency 80. VDBBench emitted only a zero-metric placeholder JSON, so no raw result is committed or compared.
