from typing import Literal

from pydantic import ConfigDict

from vectordb_bench.base import BaseModel

from .clients.api import MetricType

VIBE_REPO_ID = "vector-index-bench/vibe"
VIBE_REVISION = "07b387891a221b7b073b83d2f752b76462e5fa03"


class VibeDatasetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    distribution: Literal["id", "ood"]
    lifecycle: Literal["active", "deprecated"]
    modality: str
    size: int
    dimension: int
    source_distance: Literal["euclidean", "cosine", "normalized", "ip"]
    metric_type: MetricType
    point_type: str = "float"
    filename: str
    resource_tier: Literal["standard", "large", "very_large"] = "standard"


def _spec(
    name: str,
    distribution: Literal["id", "ood"],
    lifecycle: Literal["active", "deprecated"],
    modality: str,
    size: int,
    dimension: int,
    source_distance: Literal["euclidean", "cosine", "normalized", "ip"],
    resource_tier: Literal["standard", "large", "very_large"] = "standard",
) -> VibeDatasetSpec:
    metric_type = (
        MetricType.L2
        if source_distance == "euclidean"
        else MetricType.IP if source_distance == "ip" else MetricType.COSINE
    )
    return VibeDatasetSpec(
        name=name,
        distribution=distribution,
        lifecycle=lifecycle,
        modality=modality,
        size=size,
        dimension=dimension,
        source_distance=source_distance,
        metric_type=metric_type,
        filename=f"{name}.hdf5",
        resource_tier=resource_tier,
    )


VIBE_DATASETS = (
    _spec("agnews-mxbai-1024-euclidean", "id", "active", "Text", 769_382, 1024, "euclidean"),
    _spec("arxiv-nomic-768-normalized", "id", "active", "Text", 1_344_643, 768, "normalized"),
    _spec("dpr-jina-768-normalized", "id", "active", "Text", 20_969_760, 768, "normalized", "very_large"),
    _spec("glove-200-cosine", "id", "active", "Word", 1_192_514, 200, "cosine"),
    _spec("gooaq-distilroberta-768-normalized", "id", "active", "Text", 1_475_024, 768, "normalized"),
    _spec("imagenet-clip-512-normalized", "id", "active", "Image", 1_281_167, 512, "normalized"),
    _spec("inaturalist-resnet-2048-cosine", "id", "active", "Image", 499_000, 2048, "cosine"),
    _spec("landmark-dino-768-cosine", "id", "active", "Image", 760_757, 768, "cosine"),
    _spec("landmark-nomic-768-normalized", "id", "active", "Image", 760_757, 768, "normalized"),
    _spec("msmarco-qwen-1024-normalized", "id", "active", "Text", 8_840_823, 1024, "normalized", "very_large"),
    _spec("yahoo-minilm-384-normalized", "id", "active", "Text", 677_305, 384, "normalized"),
    _spec("hotpotqa-harrier-640-normalized", "ood", "active", "Text", 5_233_329, 640, "normalized", "large"),
    _spec("imagenet-align-640-normalized", "ood", "active", "Text-to-Image", 1_281_167, 640, "normalized"),
    _spec("laion-clip-512-normalized", "ood", "active", "Text-to-Image", 1_000_448, 512, "normalized"),
    _spec("yandex-200-cosine", "ood", "active", "Text-to-Image", 1_000_000, 200, "cosine"),
    _spec("cqadupstack-lemur-2048-ip", "ood", "active", "Multi-vector encoding", 457_149, 2048, "ip"),
    _spec("cqadupstack-muvera-5120-ip", "ood", "active", "Multi-vector encoding", 457_149, 5120, "ip", "large"),
    _spec("yi-128-ip", "ood", "active", "Attention", 187_843, 128, "ip"),
    _spec("llama-128-ip", "ood", "active", "Attention", 256_921, 128, "ip"),
    _spec("ccnews-nomic-768-normalized", "id", "deprecated", "Text", 495_328, 768, "normalized"),
    _spec("celeba-resnet-2048-cosine", "id", "deprecated", "Image", 201_599, 2048, "cosine"),
    _spec("coco-nomic-768-normalized", "ood", "deprecated", "Text-to-Image", 282_360, 768, "normalized"),
    _spec("codesearchnet-jina-768-cosine", "id", "deprecated", "Code", 1_374_067, 768, "cosine"),
    _spec("simplewiki-openai-3072-normalized", "id", "deprecated", "Text", 260_372, 3072, "normalized"),
)
VIBE_DATASET_BY_NAME = {dataset.name: dataset for dataset in VIBE_DATASETS}


def get_vibe_dataset(name: str) -> VibeDatasetSpec:
    try:
        return VIBE_DATASET_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(VIBE_DATASET_BY_NAME)
        msg = f"Unknown VIBE dataset {name!r}; supported datasets: {supported}"
        raise ValueError(msg) from exc
