# Hugging Face Dataset Source and VIBE Dataset Support

Status: Implemented; Level 1 verification complete
Target project: VectorDBBench
Design baseline: VectorDBBench `24863376eceab8e0fdaa2c39c5cae5e78c969ed4`
Hugging Face dataset revision: `07b387891a221b7b073b83d2f752b76462e5fa03`

## Summary

VectorDBBench will gain a reusable Hugging Face dataset source and a first-class
VIBE performance case. A user selects a VIBE dataset by name; VectorDBBench then
downloads the corresponding HDF5 artifact from Hugging Face, validates it,
converts it into VectorDBBench's existing Parquet layout, and runs the normal
load and search pipeline.

The first release supports the complete advertised VIBE catalog: 19 active
datasets and five deprecated datasets. This includes L2, cosine, normalized,
and inner-product datasets, as well as both in-distribution (ID) and
out-of-distribution (OOD) query workloads.

Canonical VIBE cases are unfiltered. This is a property of the published VIBE
workload, not a limitation of the Hugging Face source: VIBE publishes
unfiltered top-100 ground truth but no scalar filter fields or filter-specific
ground truth.

## Evidence Status

- **VERIFIED:** the current VectorDBBench source behavior cited in this document
  was inspected at the design baseline above.
- **VERIFIED:** the active/deprecated catalog, HDF5 structure, and published
  top-100 ground-truth depth were checked against the official VIBE dataset
  card at the pinned Hugging Face revision.
- **VERIFIED:** normalized and IP distance behavior was checked against VIBE
  source commit `6b81f95ae572f87df049b7cdb7fff97537325f7f`.
- **PROPOSED:** class names, cache layout, source-resolution rules, public case
  shape, and verification steps describe the intended implementation.
- **GUESS:** none. Any later correctness-relevant uncertainty must be resolved
  before implementation or recorded here as a blocking question.

## Motivation

VectorDBBench's current remote readers fetch files that are already in the
format expected by `DatasetManager`: training, query, ground-truth, and optional
scalar-label Parquet files. The contract and current S3/OSS implementations are
in [`data_source.py`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/data_source.py#L28-L171),
and the preparation path is in
[`dataset.py`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/dataset.py#L480-L547).

VIBE instead publishes one HDF5 artifact per dataset. Each artifact contains
the corpus vectors, query vectors, and exact neighbors together. The official
[VIBE dataset card](https://huggingface.co/datasets/vector-index-bench/vibe)
documents the HDF5 attributes and datasets. Consequently, adding only a new
`DatasetSource` enum member would not make VIBE consumable: the feature needs a
transport layer and a format adapter.

## Goals

- Add Hugging Face as a reusable dataset source provider.
- Make every dataset in the advertised VIBE catalog selectable by name.
- Preserve VIBE's distribution, lifecycle, point-type, and distance metadata.
- Support L2, cosine, normalized, and IP workloads without silently changing
  vector values or similarity semantics.
- Convert VIBE HDF5 artifacts into the existing VectorDBBench Parquet contract.
- Reuse the existing dataset iterator, load runner, serial quality search,
  concurrent search, result schema, and database clients.
- Make downloads and conversion reproducible, resumable, and safe after an
  interrupted process.
- Store source provenance in exported VIBE results so they remain interpretable
  without the local conversion cache.
- Preserve existing S3, Aliyun OSS, `ir_datasets`, CLI, REST, and UI behavior.

## Non-goals

- Reproducing VIBE's algorithm harness, parameter grids, or published result
  website inside VectorDBBench.
- Running a benchmark automatically as part of implementation or unit tests.
- Creating scalar labels or filtered ground truth for VIBE datasets.
- Consuming VIBE's OOD `learn` and `learn_neighbors` arrays in the first
  release. They are training/tuning data from the query distribution and are
  not part of VectorDBBench's existing performance-case contract.
- Guaranteeing that every VectorDBBench database/index implementation supports
  every VIBE metric or dimension.
- Supporting auxiliary HDF5 objects that are present in the Hugging Face
  repository but absent from the advertised VIBE catalog, such as derived
  uint8, int8, binary, and intermediate Chamfer datasets. They can be added in
  a later catalog revision using the same source and conversion architecture.
- Redistributing VIBE data as part of the VectorDBBench Python package.

## Terminology

- **Dataset source provider:** transport that resolves remote artifacts into
  local files, such as S3, OSS, `ir_datasets`, or Hugging Face.
- **Raw artifact:** the immutable HDF5 file downloaded from Hugging Face.
- **Prepared dataset:** the Parquet files consumed by VectorDBBench.
- **Catalog entry:** declarative metadata describing one selectable VIBE
  dataset and its source artifact.
- **Canonical VIBE case:** an unfiltered performance case using the published
  VIBE queries and top-100 ground truth without modifying the workload.
- **Derived workload:** a workload built from VIBE vectors but with generated
  labels, different queries, recomputed ground truth, or other semantic changes.

## Existing Constraints

1. `DatasetSource` is currently a global run-level choice except that the
   assembler forces `IR_DATASETS` for FTS. See
   [`assembler.py`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/assembler.py#L22-L46).
2. `DatasetManager.prepare()` asks the selected reader for filenames already
   meaningful to VectorDBBench, then loads `test.parquet` and validates the
   ground truth. See
   [`dataset.py`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/dataset.py#L480-L543).
3. Training files are streamed through PyArrow rather than loaded in full. See
   [`DataSetIterator`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/dataset.py#L615-L666).
4. Vector ground truth requires query IDs aligned with `test.parquet` and at
   least the requested width. See
   [`ParquetGroundTruth`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/dataset.py#L207-L278).
5. Existing filtered cases select filter-specific ground-truth filenames and
   may require scalar-label files. See
   [`filter.py`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/filter.py#L34-L93).
6. VDBBench already models `L2`, `COSINE`, and `IP` as distinct metrics. See
   [`MetricType`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/clients/api.py#L13-L20).

## Proposed Architecture

```text
VIBE catalog entry
        |
        v
HuggingFaceReader -- hf_hub_download(one pinned HDF5 file)
        |
        v
Hugging Face raw cache
        |
        v
VibeDatasetManager -- validate and stream-convert
        |
        v
revisioned Parquet cache
  train-*.parquet
  test.parquet
  neighbors.parquet
  manifest.json
        |
        v
existing DatasetManager / DataSetIterator / runners / database clients
```

The Hugging Face reader owns transport and provider caching. The VIBE dataset
manager owns VIBE-specific validation and conversion. It is important not to
put HDF5 interpretation inside `HuggingFaceReader`, because other Hugging Face
repositories may publish Parquet, JSON, Arrow, or different HDF5 schemas.

## Dataset Source Contract

### Source registration

Add `DatasetSource.HuggingFace` and make `DatasetSource.reader()` return a
`HuggingFaceReader` for it. The dependency is `huggingface_hub`.

The reader continues to operate on an explicit repository and file list. Its
contract gains an optional keyword-only `revision` and returns the resolved
local paths:

```python
def read(
    self,
    dataset: str,
    files: list[str],
    local_ds_root: pathlib.Path,
    *,
    revision: str | None = None,
) -> dict[str, pathlib.Path]:
    ...
```

For Hugging Face:

- `dataset` is the repository ID, for example `vector-index-bench/vibe`.
- `files` contains exact repository-relative filenames.
- `revision` is required for first-class catalog entries.
- `repo_type="dataset"` is always supplied.
- `hf_hub_download()` is called once per requested file.
- `snapshot_download()` is not used because it would resolve the entire
  repository rather than the selected dataset artifact.
- The standard Hugging Face cache is used; the reader must not copy a large
  cached artifact merely to satisfy a VectorDBBench directory convention.
- The SDK's standard authentication and endpoint configuration are honored.
  Tokens must never be accepted through the case config or written to logs.

The official Hugging Face documentation describes single-file downloads,
revision selection, and cache behavior in the
[`hf_hub_download` guide](https://huggingface.co/docs/huggingface_hub/guides/download).

S3 and OSS readers may return their existing local paths after download;
callers that do not need the result may continue ignoring it. Their network and
filesystem behavior otherwise remains unchanged. `IRDatasetsReader` continues
to let `ir_datasets` manage its cache.

### Source resolution

The global S3/OSS setting is a mirror selection for legacy datasets. It is not
appropriate for a dataset whose only declared resource is a Hugging Face
artifact. Add a nullable `preferred_source` property to dataset managers:

```text
VibeDatasetManager -> HuggingFace
FtsDatasetManager  -> IR_DATASETS
other managers     -> no preference
```

The assembler resolves the source as:

```python
actual_source = case.dataset.preferred_source or run_level_source
```

This generalizes the existing FTS special case and prevents an Aliyun checkbox
or `DATASET_SOURCE=S3` default from redirecting VIBE requests to the wrong
provider. Existing public interfaces remain valid. A future generic source
selector can call a typed `set_dataset_source()` method, while the existing
`set_download_address(use_aliyun, use_ir_datasets)` method remains available
for compatibility.

## VIBE Catalog

### Catalog model

Add a small immutable `VibeDatasetSpec` model rather than 24 dataset classes:

```python
class VibeDatasetSpec(BaseModel):
    name: str
    distribution: Literal["id", "ood"]
    lifecycle: Literal["active", "deprecated"]
    modality: str
    size: int
    dimension: int
    source_distance: Literal["euclidean", "cosine", "normalized", "ip"]
    metric_type: MetricType
    point_type: str
    filename: str
    resource_tier: Literal["standard", "large", "very_large"]
```

Repository ID and revision are catalog-level constants rather than repeated in
every row:

```text
repo_id  = vector-index-bench/vibe
revision = 07b387891a221b7b073b83d2f752b76462e5fa03
```

The catalog is checked into VectorDBBench. It is not reconstructed from
filenames and is not fetched dynamically at startup. Updating the catalog must
be a reviewable code change that also updates the pinned Hugging Face revision.
This keeps historical benchmark inputs reproducible.

`resource_tier` is advisory UI/CLI metadata. It never hides or rejects a
dataset. In particular, DPR-Jina and MSMARCO remain selectable even though they
require substantially more memory and disk than the smaller datasets.

### Advertised dataset inventory

The following entries come from the official
[VIBE dataset card](https://huggingface.co/datasets/vector-index-bench/vibe).
Deprecated entries remain downloadable and selectable but are visually marked
as deprecated.

| Lifecycle | Distribution | Dataset | Modality | Corpus rows | Dimension | VIBE distance | VDBBench metric |
|---|---|---|---|---:|---:|---|---|
| Active | ID | `agnews-mxbai-1024-euclidean` | Text | 769,382 | 1024 | euclidean | L2 |
| Active | ID | `arxiv-nomic-768-normalized` | Text | 1,344,643 | 768 | normalized/any | COSINE |
| Active | ID | `dpr-jina-768-normalized` | Text | 20,969,760 | 768 | normalized/any | COSINE |
| Active | ID | `glove-200-cosine` | Word | 1,192,514 | 200 | cosine | COSINE |
| Active | ID | `gooaq-distilroberta-768-normalized` | Text | 1,475,024 | 768 | normalized/any | COSINE |
| Active | ID | `imagenet-clip-512-normalized` | Image | 1,281,167 | 512 | normalized/any | COSINE |
| Active | ID | `inaturalist-resnet-2048-cosine` | Image | 499,000 | 2048 | cosine | COSINE |
| Active | ID | `landmark-dino-768-cosine` | Image | 760,757 | 768 | cosine | COSINE |
| Active | ID | `landmark-nomic-768-normalized` | Image | 760,757 | 768 | normalized/any | COSINE |
| Active | ID | `msmarco-qwen-1024-normalized` | Text | 8,840,823 | 1024 | normalized/any | COSINE |
| Active | ID | `yahoo-minilm-384-normalized` | Text | 677,305 | 384 | normalized/any | COSINE |
| Active | OOD | `hotpotqa-harrier-640-normalized` | Text | 5,233,329 | 640 | normalized/any | COSINE |
| Active | OOD | `imagenet-align-640-normalized` | Text-to-Image | 1,281,167 | 640 | normalized/any | COSINE |
| Active | OOD | `laion-clip-512-normalized` | Text-to-Image | 1,000,448 | 512 | normalized/any | COSINE |
| Active | OOD | `yandex-200-cosine` | Text-to-Image | 1,000,000 | 200 | cosine | COSINE |
| Active | OOD | `cqadupstack-lemur-2048-ip` | Multi-vector encoding | 457,149 | 2048 | IP | IP |
| Active | OOD | `cqadupstack-muvera-5120-ip` | Multi-vector encoding | 457,149 | 5120 | IP | IP |
| Active | OOD | `yi-128-ip` | Attention | 187,843 | 128 | IP | IP |
| Active | OOD | `llama-128-ip` | Attention | 256,921 | 128 | IP | IP |
| Deprecated | ID | `ccnews-nomic-768-normalized` | Text | 495,328 | 768 | normalized/any | COSINE |
| Deprecated | ID | `celeba-resnet-2048-cosine` | Image | 201,599 | 2048 | cosine | COSINE |
| Deprecated | OOD | `coco-nomic-768-normalized` | Text-to-Image | 282,360 | 768 | normalized/any | COSINE |
| Deprecated | ID | `codesearchnet-jina-768-cosine` | Code | 1,374,067 | 768 | cosine | COSINE |
| Deprecated | ID | `simplewiki-openai-3072-normalized` | Text | 260,372 | 3072 | normalized/any | COSINE |

The lifecycle labels follow the current dataset card. The historical ID/OOD
classification of deprecated datasets follows VIBE's earlier
[`d31513f8` evaluation table](https://github.com/vector-index-bench/vibe/blob/d31513f83b2ac85c6f54ee1e4719f7c845a9cdf6/README.md#L185-L210).
Catalog tests must lock these values so a later reclassification is explicit.

### Catalog boundary

At the pinned revision, the Hugging Face repository contains additional HDF5
files that are not listed in the dataset card's active or deprecated tables.
The first release does not present those as canonical selectable VIBE cases.
This distinction prevents an implementation detail or intermediate artifact
from silently becoming a public benchmark workload.

Adding one later requires:

1. an explicit catalog row;
2. a supported point type and metric mapping;
3. shape and semantic tests;
4. a lifecycle and distribution classification; and
5. a revision bump when the artifact is not present at the current pin.

## HDF5 Validation and Conversion

### Required source structure

The converter requires these HDF5 attributes:

- `dimension`
- `distance`
- `point_type`

It requires these arrays:

- `train`: `(corpus_count, dimension)`
- `test`: `(query_count, dimension)`
- `neighbors`: `(query_count, 100)`
- `distances`: `(query_count, 100)`

OOD files may additionally contain `learn` and `learn_neighbors`. The VIBE
generator writes the attributes and core arrays in
[`datasets.py`](https://github.com/vector-index-bench/vibe/blob/6b81f95ae572f87df049b7cdb7fff97537325f7f/vibe/datasets.py#L177-L193).

Before writing output, preparation validates:

- required attributes and arrays exist;
- the HDF5 distance and dimension match the catalog entry;
- corpus and query arrays are rank two and have the expected dimension;
- corpus row count matches the catalog entry;
- neighbor and distance shapes match the query count;
- ground-truth width is exactly 100 for the pinned catalog revision;
- every neighbor ID is an integer in `[0, corpus_count)`;
- vector point type is supported by the catalog entry; and
- generated query IDs will be unique and aligned across test and truth output.

A catalog mismatch is a hard error. VectorDBBench must not silently update its
metadata from a mutable remote artifact.

### Prepared Parquet schema

The converter emits the existing custom-dataset-compatible fields:

| File | Field | Type | Meaning |
|---|---|---|---|
| `train-XX-of-NN.parquet` | `id` | int64 | Zero-based corpus row index |
| `train-XX-of-NN.parquet` | `emb` | fixed-size list of source numeric type | Corpus vector |
| `test.parquet` | `id` | int64 | Zero-based query row index |
| `test.parquet` | `emb` | fixed-size list of source numeric type | Query vector |
| `neighbors.parquet` | `id` | int64 | Same query ID as `test.parquet` |
| `neighbors.parquet` | `neighbors_id` | fixed-size list of int64 | Published top-100 corpus IDs |

The IDs intentionally equal HDF5 row positions because VIBE neighbor IDs index
the corpus rows. The converter does not reorder, sample, shuffle, normalize,
or cast floating-point vectors to a lower precision.

Training conversion is chunked directly from HDF5 into partitioned Parquet.
Partition sizing is based on an approximate uncompressed byte target, not a
fixed row count, so high-dimensional datasets do not create disproportionately
large files. Test and ground-truth arrays are much smaller but should still be
read and written in bounded batches when practical.

`distances` is validated but not emitted in the first release because the
existing vector recall path consumes neighbor IDs. Dropping it from prepared
Parquet does not change Recall@K semantics. The raw HDF5 remains in the Hugging
Face cache if later metrics need published distances.

### Metric semantics

The mapping is exact at the ranking level:

| HDF5 `distance` | VDBBench metric | Handling |
|---|---|---|
| `euclidean` | `MetricType.L2` | Preserve vectors; backend performs L2 search. |
| `cosine` | `MetricType.COSINE` | Preserve vectors; backend performs cosine search. |
| `normalized` | `MetricType.COSINE` | Preserve already-normalized vectors; backend performs cosine search. |
| `ip` | `MetricType.IP` | Preserve vectors; backend performs maximum inner-product search. |

VIBE normalizes the vectors when it writes a `normalized` dataset and evaluates
them with `1 - dot(a, b)`, as shown in
[`datasets.py`](https://github.com/vector-index-bench/vibe/blob/6b81f95ae572f87df049b7cdb7fff97537325f7f/vibe/datasets.py#L177-L181)
and
[`distance.py`](https://github.com/vector-index-bench/vibe/blob/6b81f95ae572f87df049b7cdb7fff97537325f7f/vibe/distance.py#L43-L55).
For unit-normalized vectors this has the same ordering as cosine similarity, so
`MetricType.COSINE` preserves the published neighbors.

The catalog and HDF5 attribute are both checked. A filename suffix alone never
selects the metric. In particular, IP datasets are not normalized or mapped to
cosine as a compatibility fallback.

### Ground-truth depth

The published `neighbors` array contains 100 neighbors per query. Canonical
VIBE cases therefore accept `1 <= k <= 100`. A larger K fails during case
validation, before database creation or data loading. Smaller K values use the
first K published neighbors through the existing ground-truth reader.

## Cache and Recovery Model

Raw provider artifacts and prepared benchmark files have different lifecycles:

```text
<DATASET_LOCAL_DIR>/
  huggingface-cache/                 # managed through huggingface_hub
  vibe/
    <dataset-name>/
      <hf-revision>/
        schema-v1/
          train-00-of-N.parquet
          ...
          test.parquet
          neighbors.parquet
          manifest.json
```

The exact Hugging Face cache location may honor the SDK's standard environment
configuration. The prepared path always includes dataset name, resolved commit,
and conversion schema version. Changing any of these creates a different cache
entry rather than mutating an older benchmark input.

Preparation follows this sequence:

1. Look for a complete manifest and all declared output files.
2. Validate the manifest against the catalog and requested revision.
3. If valid, reuse the prepared dataset without opening the HDF5 file.
4. Otherwise, resolve the exact HDF5 artifact through `HuggingFaceReader`.
5. Validate source metadata and shapes.
6. Convert each output to a process-unique temporary path.
7. Validate output row counts, schema, and ground-truth alignment.
8. Atomically publish the output files.
9. Write `manifest.json` last using an atomic replacement.

The manifest is the completion marker. Partial files without a valid manifest
are never reused. Concurrent preparations may duplicate conversion work, but
they must not expose partial output or corrupt a complete cache entry.

The manifest records:

- conversion schema version;
- repository ID, requested revision, and resolved commit;
- source filename and provider-reported ETag or content identifier when
  available;
- source HDF5 attributes;
- catalog name, lifecycle, distribution, size, dimension, and metric;
- output filenames, row counts, schemas, and file sizes; and
- preparation timestamp and VectorDBBench version.

Hugging Face owns raw-download integrity and retry behavior. VectorDBBench owns
source-schema validation and prepared-output atomicity. Errors retain enough
context to identify the repository, revision, filename, and preparation stage,
but never include credentials.

## Case Model and User Experience

### Case type

Add one `CaseType.VibePerformance` and one parameterized
`VibePerformanceCase`. Do not create one Python case class per dataset.

The case accepts one required public parameter:

```text
vibe_dataset=<catalog name>
```

It constructs `VibeDatasetManager` from the catalog and inherits the existing
performance-case search controls such as K, concurrency, load behavior, and
timeouts. Dataset metric and dimension come from the catalog, not user input.

The case always uses `NonFilter`. Supplying a VDBBench filter option with
`VibePerformance` fails validation with a message explaining that canonical
VIBE artifacts do not contain scalar fields or filtered ground truth.

### CLI

The intended CLI shape is one additional dataset selector:

```bash
vectordbbench milvushnsw \
  --case-type VibePerformance \
  --vibe-dataset glove-200-cosine
```

The CLI help lists active datasets first and deprecated datasets afterward.
Selecting a deprecated entry prints a warning but does not require a special
override. Selecting a large or very-large entry prints an estimated-resource
warning but remains allowed.

Users do not provide the Hugging Face repository, filename, dimension, metric,
or revision for a canonical catalog case. That keeps the public interface short
and prevents accidental workload drift.

### Frontend

Add a `VIBE Search Performance` case cluster generated from the catalog,
following the dynamic item pattern already used for FTS in
[`dbCaseConfigs.py`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/frontend/config/dbCaseConfigs.py#L195-L215).

Each item displays:

- dataset name;
- ID or OOD;
- metric and dimension;
- corpus size;
- active or deprecated status; and
- a resource warning for large entries.

Active and deprecated entries are separate visual groups. The existing
S3/Aliyun download selector does not apply to VIBE items because the dataset
declares Hugging Face as its preferred source.

### REST and serialized configuration

The existing task envelope remains unchanged. A VIBE case serializes:

```json
{
  "case_id": "VibePerformance",
  "vibe_dataset": "glove-200-cosine"
}
```

No Hugging Face token, repository, filename, or mutable `main` revision appears
in task configuration. Old task and result files remain readable because no
existing field changes meaning and no existing enum value is renamed.

### Result provenance

Add an optional `dataset_metadata` field to `CaseResult`, which currently owns
the metrics and task configuration for one result. See
[`CaseResult`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/models.py#L351-L360).

```python
class DatasetMetadata(BaseModel):
    name: str
    distribution: Literal["id", "ood"]
    lifecycle: Literal["active", "deprecated"]
    source: DatasetSource
    repository: str
    filename: str
    revision: str
    source_distance: str
    metric_type: MetricType
    point_type: str


class CaseResult(BaseModel):
    metrics: Metric
    task_config: TaskConfig
    label: ResultLabel = ResultLabel.NORMAL
    dataset_metadata: DatasetMetadata | None = None
```

For a successful VIBE preparation, `dataset_metadata` is constructed from the
validated conversion manifest, not reconstructed later from the current
catalog. It records the dataset name, ID/OOD classification, lifecycle, source
metric, VDBBench metric, point type, repository, source filename, and resolved
Hugging Face revision used by that run.

If preparation fails before the source is resolved and validated,
`dataset_metadata` remains absent; the requested dataset name is still present
in the task's VIBE case configuration. VectorDBBench must not emit resolved
provenance based only on an unvalidated catalog expectation.

The field is optional so result JSON written before this feature continues to
load with `dataset_metadata=None`. Result serialization, result collection, and
frontend readers preserve the object when present and tolerate it when absent.
The frontend may display it, but it must not require it to render an older
result.

## Filtering Semantics

The Hugging Face source provider is format- and workload-neutral. It can fetch
filtered datasets when a repository publishes them.

Canonical VIBE is unfiltered because its published files contain vectors and
unfiltered nearest-neighbor ground truth, but no scalar label domain and no
ground truth conditioned on a filter expression. VDBBench must not generate
arbitrary labels and report the resulting workload as VIBE.

A later derived filtering feature is possible, but it must:

- define a deterministic scalar-label generation rule;
- define filter selectivities and expressions;
- recompute exact ground truth under each filter;
- version those derivation rules and outputs; and
- label resulting cases and reports as `VDBBench-derived from VIBE`.

## Backend and Index Compatibility

Full dataset support means VectorDBBench can discover, download, validate,
prepare, and represent every advertised VIBE dataset. It does not mean every
database/index pair can execute every entry.

Before load, the assembled database case receives the catalog metric using the
existing metric assignment path in
[`Assembler`](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/assembler.py#L33-L38).
Each backend remains responsible for translating or rejecting the metric and
dimension. A backend must not silently substitute cosine for IP, IP for cosine,
or another distance to make a case run.

The first implementation must include focused validation for the target smoke
backend. A universal metric/dimension capability matrix for every existing
VectorDBBench adapter is outside this feature and must not block catalog and
dataset preparation support.

Unsupported examples fail before loading when they are knowable from static
configuration. Provider or server limits only discoverable at runtime must be
reported as backend errors, not dataset-conversion failures.

## Dependencies

Add direct runtime dependencies for:

- `huggingface_hub` for single-file resolution, revision pinning, caching, and
  downloads;
- `h5py` for bounded HDF5 reads;
- `pyarrow` if it remains only transitively installed, because the converter
  and existing iterator directly import it; and
- `filelock` for cross-process conversion locking so concurrent preparations
  cannot publish different Parquet files under one valid manifest.

Dependency versions must remain compatible with VectorDBBench's supported
Python versions. Optionalizing these dependencies is not recommended for the
first implementation: selecting a built-in VIBE case should not fail because a
hidden dataset extra was omitted during installation.

## Security, Privacy, and Licensing

- The VIBE Hugging Face repository is public; anonymous download is the normal
  path.
- Standard Hugging Face credentials may be used by the SDK when configured,
  but VectorDBBench does not solicit, serialize, or log them.
- Repository ID and revision are allowlisted by the checked-in catalog for
  canonical VIBE cases.
- Filenames come from the catalog, not unsanitized user input.
- Conversion never executes code from the dataset repository.
- VectorDBBench downloads directly from the source and does not bundle or
  republish the data.
- Documentation must retain the VIBE dataset citation and the source card's
  dataset-specific credit notices. The repository is marked CC BY 4.0 on its
  [Hugging Face page](https://huggingface.co/datasets/vector-index-bench/vibe).

## Compatibility

### Preserved behavior

- S3 remains the default for existing vector datasets.
- Aliyun OSS selection continues to affect existing mirrored datasets.
- FTS continues using `ir_datasets` automatically.
- Existing case IDs, CLI flags, REST payloads, custom datasets, and stored
  results keep their current meaning.
- Existing result files without `dataset_metadata` load with a null default;
  VIBE results include validated provenance at the `CaseResult` level.
- The existing Parquet iterator and search runners remain the only path that
  inserts and queries prepared VIBE vectors.

### New validation failures

- Unknown VIBE catalog name.
- Deprecated dataset warning, not failure.
- Catalog metadata does not match HDF5 metadata or shapes.
- Unsupported point type or distance.
- Neighbor IDs outside the corpus range.
- Requested K is outside `1..100`.
- Filtered `VibePerformance` request.
- Backend/index cannot preserve the selected dataset metric.
- Incomplete or inconsistent prepared cache.

## Verification Plan

The implementation uses VDBBench development verification Level 1: narrow,
deterministic tests for each changed contract. A real download is a separately
authorized functional probe, not a unit test.

### Dataset source tests

- `DatasetSource.HuggingFace` resolves `HuggingFaceReader`.
- Mocked `hf_hub_download()` receives the exact repository, filename,
  `repo_type="dataset"`, revision, and cache options.
- Reader returns the resolved local path.
- Download/provider exceptions retain dataset context and do not expose a
  token.
- Existing S3, OSS, and `ir_datasets` dispatch remains unchanged.

### Converter tests

Use tiny synthetic HDF5 fixtures; do not download VIBE in unit tests.

- Valid L2, cosine/normalized, and IP fixtures convert successfully.
- Output IDs, vectors, neighbor IDs, schemas, dimensions, and row counts match
  the source fixture.
- Normalized vectors are not normalized a second time.
- IP vectors are not normalized or remapped to cosine.
- Training conversion creates the expected partition names and row coverage.
- Missing attributes/arrays, mismatched dimensions, invalid shapes, and
  out-of-range neighbor IDs fail clearly.
- Output is reusable only after a valid manifest is present.
- An interrupted temporary output is ignored and safely rebuilt.
- Matching revision/schema cache is reused without reconversion.
- Revision or schema change selects a new prepared path.

### Catalog and case tests

- Catalog contains exactly 19 active and five deprecated entries.
- Catalog contains 15 ID and nine OOD entries when historical deprecated
  distribution is included.
- All names are unique and filenames are `<name>.hdf5`.
- All metric mappings are explicit; the four active IP entries map to IP.
- `VibePerformanceCase` constructs the correct dataset manager from a name.
- Unknown names, filters, and K above 100 fail before database initialization.
- Deprecated and resource-tier entries remain selectable and emit warnings.
- Dataset-preferred source wins for VIBE and FTS; run-level S3/OSS selection
  still controls existing vector datasets.
- Case serialization and deserialization preserve the dataset name.
- CLI option mapping and generated frontend case items cover the full catalog.

### Result provenance tests

- A successfully prepared VIBE case writes the exact metadata captured by its
  validated conversion manifest.
- The result records name, distribution, lifecycle, source, repository,
  filename, resolved revision, source distance, VDBBench metric, and point type.
- A preparation failure does not claim resolved metadata.
- An older result without `dataset_metadata` loads with a null value and
  round-trips without failure.
- Result output, collection, and frontend transformation preserve VIBE metadata
  and tolerate mixed old and new results.

### Documentation checks

- CLI examples match the implemented command and option names.
- The pinned Hugging Face revision contains every catalog filename.
- Source and credit links resolve.
- Counts in the catalog table and tests agree.

### Functional probe

After the unit implementation passes Level 1, run one separately authorized
functional probe using a comparatively small active dataset. The probe verifies
download, cache reuse, conversion, loading, serial Recall@K, and concurrent
search against one target backend. It is functional validation, not performance
evidence.

An IP functional probe is also required before declaring the full feature
benchmark-ready, because a cosine/L2 probe does not validate IP translation.

## Implementation Sequence

1. Add dependencies, source enum, and mocked Hugging Face reader tests.
2. Add catalog model/data and catalog consistency tests.
3. Add `VibeDatasetManager`, synthetic HDF5 conversion, manifests, and recovery
   tests.
4. Generalize preferred-source selection while preserving FTS and legacy
   behavior.
5. Add `VibePerformanceCase`, CLI mapping, and validation.
6. Add generated frontend items and REST serialization coverage.
7. Add optional result provenance and compatibility coverage.
8. Update user documentation, attribution, and troubleshooting guidance.
9. Run Level 1 checks.
10. With separate authorization, run one standard-metric and one IP functional
   probe on the client machine.

## Alternatives Considered

### Standalone conversion script plus custom dataset

This is the smallest implementation. A user manually downloads and converts an
HDF5 file, then fills in every custom-dataset field. It does not create a
Hugging Face source, does not provide a reviewed catalog, and makes provenance
and metric mistakes easy. It remains useful as a debugging tool but is not the
public feature.

### Convert HDF5 inside `HuggingFaceReader`

This keeps the VIBE manager smaller but incorrectly couples a provider to one
repository's format. A generic Hugging Face reader must also be able to fetch
non-HDF5 artifacts without assuming VIBE semantics.

### Download the entire repository snapshot

This simplifies filename resolution but downloads unrelated, deprecated, and
derived datasets. Single-file download is required because users select one
workload at a time and the repository is large.

### Upload preconverted Parquet mirrors

This would simplify runtime conversion, but creates a second distribution that
must be hosted, versioned, attributed, and kept consistent with VIBE. Direct
download plus deterministic local conversion avoids that operational burden.

### One case class per VIBE dataset

This follows some existing fixed performance cases but would duplicate source,
metric, timeout, and UI configuration 24 times. A catalog-backed parameterized
case is shorter for callers and makes completeness testable.

## Acceptance Criteria

The feature is complete when:

- all 24 advertised VIBE entries are selectable;
- selecting one downloads only its pinned HDF5 artifact;
- prepared files satisfy the existing VectorDBBench Parquet and ground-truth
  contracts;
- L2, cosine/normalized, and IP semantics are preserved;
- active/deprecated and ID/OOD metadata is visible;
- exported VIBE results contain validated dataset source provenance and older
  results remain readable;
- interrupted preparation cannot be mistaken for success;
- existing dataset sources and cases pass their focused compatibility tests;
- Level 1 tests pass; and
- authorized standard-metric and IP functional probes pass before benchmark
  readiness is declared.

## References

- [VIBE Hugging Face dataset card](https://huggingface.co/datasets/vector-index-bench/vibe)
- [VIBE source repository](https://github.com/vector-index-bench/vibe/tree/6b81f95ae572f87df049b7cdb7fff97537325f7f)
- [Hugging Face Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
- [VectorDBBench dataset source implementations](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/data_source.py)
- [VectorDBBench dataset preparation and iteration](https://github.com/zilliztech/VectorDBBench/blob/24863376eceab8e0fdaa2c39c5cae5e78c969ed4/vectordb_bench/backend/dataset.py)
