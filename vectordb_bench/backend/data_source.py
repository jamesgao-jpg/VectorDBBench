import logging
import os
import pathlib
import time
import typing
from abc import ABC, abstractmethod
from enum import Enum

import ir_datasets
from fsspec.callbacks import Callback
from tqdm import tqdm

from vectordb_bench import config

# Set ir_datasets to use tmp directory for both home and temp
# This ensures all downloaded files and temporary data are stored in /tmp
ir_datasets_home = pathlib.Path(config.DATASET_LOCAL_DIR) / "ir_datasets"
ir_datasets_tmp = pathlib.Path(config.DATASET_LOCAL_DIR) / "ir_datasets_tmp"
os.environ.setdefault("IR_DATASETS_HOME", str(ir_datasets_home))
os.environ.setdefault("IR_DATASETS_TMP", str(ir_datasets_tmp))


logging.getLogger("s3fs").setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

DatasetReader = typing.TypeVar("DatasetReader")
ProgressCallback = typing.Callable[[int, int, str], None]


class _ThrottledProgress:
    def __init__(
        self,
        callback: ProgressCallback | None,
        total: int,
        interval_seconds: float = 0.25,
    ):
        self.callback = callback
        self.total = total
        self.interval_seconds = interval_seconds
        self._last_emit = 0.0

    def emit(self, current: int, message: str, *, force: bool = False):
        if self.callback is None:
            return

        now = time.monotonic()
        if force or now - self._last_emit >= self.interval_seconds:
            self.callback(current, self.total, message)
            self._last_emit = now


class DatasetSource(Enum):
    S3 = "S3"
    AliyunOSS = "AliyunOSS"
    IR_DATASETS = "IR_DATASETS"

    def reader(self) -> DatasetReader:
        if self == DatasetSource.S3:
            return AwsS3Reader()

        if self == DatasetSource.AliyunOSS:
            return AliyunOSSReader()

        if self == DatasetSource.IR_DATASETS:
            return IRDatasetsReader()

        return None


class DatasetReader(ABC):
    source: DatasetSource
    remote_root: str

    @abstractmethod
    def read(
        self,
        dataset: str,
        files: list[str],
        local_ds_root: pathlib.Path,
        progress_callback: ProgressCallback | None = None,
    ):
        """read dataset files from remote_root to local_ds_root,

        Args:
            dataset(str): for instance "sift_small_500k"
            files(list[str]):  all filenames of the dataset
            local_ds_root(pathlib.Path): whether to write the remote data.
        """

    @abstractmethod
    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        pass


class AliyunOSSReader(DatasetReader):
    source: DatasetSource = DatasetSource.AliyunOSS
    remote_root: str = config.ALIYUN_OSS_URL

    def __init__(self):
        import oss2

        self.bucket = oss2.Bucket(oss2.AnonymousAuth(), self.remote_root, "benchmark", True)

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        info = self.bucket.get_object_meta(remote.as_posix())

        # check size equal
        remote_size, local_size = info.content_length, local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True

    def read(
        self,
        dataset: str,
        files: list[str],
        local_ds_root: pathlib.Path,
        progress_callback: ProgressCallback | None = None,
    ):
        downloads = []
        completed_bytes = 0
        total_bytes = 0
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)

        for file in files:
            remote_file = pathlib.PurePosixPath("benchmark", dataset, file)
            local_file = local_ds_root.joinpath(file)
            remote_size = self.bucket.get_object_meta(remote_file.as_posix()).content_length
            total_bytes += remote_size

            if local_file.exists() and local_file.stat().st_size == remote_size:
                completed_bytes += remote_size
                continue

            log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
            downloads.append((remote_file, local_file, remote_size))

        reporter = _ThrottledProgress(progress_callback, total_bytes)
        reporter.emit(completed_bytes, "Checking cached dataset files", force=True)

        if len(downloads) == 0:
            reporter.emit(total_bytes, "Dataset files are ready", force=True)
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for remote_file, local_file, remote_size in tqdm(downloads):
            log.debug(f"downloading file {remote_file} to {local_file}")
            self.bucket.get_object_to_file(remote_file.as_posix(), local_file.absolute())
            completed_bytes += remote_size
            reporter.emit(completed_bytes, f"Downloaded {local_file.name}", force=True)

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")


class AwsS3Reader(DatasetReader):
    source: DatasetSource = DatasetSource.S3
    remote_root: str = config.AWS_S3_URL

    def __init__(self):
        import s3fs

        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})

    def ls_all(self, dataset: str):
        dataset_root_dir = pathlib.Path(self.remote_root, dataset)
        log.info(f"listing dataset: {dataset_root_dir}")
        names = self.fs.ls(dataset_root_dir)
        for n in names:
            log.info(n)
        return names

    def read(
        self,
        dataset: str,
        files: list[str],
        local_ds_root: pathlib.Path,
        progress_callback: ProgressCallback | None = None,
    ):
        downloads: list[tuple[pathlib.PurePosixPath, pathlib.Path, int]] = []
        completed_bytes = 0
        total_bytes = 0
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)

        for file in files:
            remote_file = pathlib.PurePosixPath(self.remote_root, dataset, file)
            local_file = local_ds_root.joinpath(file)
            remote_size = self.fs.info(remote_file).get("size")
            if not isinstance(remote_size, int):
                msg = f"Unable to determine remote file size: {remote_file}"
                raise OSError(msg)
            total_bytes += remote_size

            if local_file.exists() and local_file.stat().st_size == remote_size:
                completed_bytes += remote_size
                continue

            log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
            downloads.append((remote_file, local_file, remote_size))

        reporter = _ThrottledProgress(progress_callback, total_bytes)
        reporter.emit(completed_bytes, "Checking cached dataset files", force=True)

        if len(downloads) == 0:
            reporter.emit(total_bytes, "Dataset files are ready", force=True)
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for s3_file, local_file, remote_size in tqdm(downloads):
            partial_file = local_file.with_name(f"{local_file.name}.part")
            local_file.unlink(missing_ok=True)
            partial_file.unlink(missing_ok=True)
            log.debug(f"downloading file {s3_file} to {partial_file}")

            file_start = completed_bytes

            def report_file_progress(
                _size: int | None,
                value: int,
                _file_start: int = file_start,
                _remote_size: int = remote_size,
                _file_name: str = local_file.name,
                **_kwargs,
            ):
                current = min(_file_start + value, _file_start + _remote_size)
                reporter.emit(current, f"Downloading {_file_name}")

            callback = (
                Callback(hooks={"progress": report_file_progress})
                if progress_callback is not None
                else Callback()
            )
            try:
                self.fs.get_file(s3_file.as_posix(), partial_file.as_posix(), callback=callback)
            except Exception:
                partial_file.unlink(missing_ok=True)
                raise

            partial_size = partial_file.stat().st_size
            if partial_size != remote_size:
                partial_file.unlink(missing_ok=True)
                msg = (
                    f"downloaded file: {partial_file} size[{partial_size}] "
                    f"not match with remote size[{remote_size}]"
                )
                raise OSError(msg)
            partial_file.replace(local_file)

            completed_bytes += remote_size
            reporter.emit(completed_bytes, f"Downloaded {local_file.name}", force=True)

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        # info() uses ls() inside, maybe we only need to ls once
        info = self.fs.info(remote)

        # check size equal
        remote_size, local_size = info.get("size"), local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True


class IRDatasetsReader(DatasetReader):
    """Reader for ir_datasets based datasets"""

    source: DatasetSource = DatasetSource.IR_DATASETS
    remote_root: str = ""  # Not used for ir_datasets

    def __init__(self):
        self.ir_datasets = ir_datasets

    def read(
        self,
        dataset: str,
        files: list[str],
        local_ds_root: pathlib.Path,
        progress_callback: ProgressCallback | None = None,
    ):
        """
        Download FTS dataset using ir_datasets API

        Args:
            dataset: ir_datasets dataset name
            files: Expected output files (ignored, not used)
            local_ds_root: Local directory (not used, ir_datasets handles its own cache)
        """
        log.info(f"Downloading FTS dataset '{dataset}' using ir_datasets")
        if progress_callback is not None:
            progress_callback(0, 0, f"Loading dataset metadata for {dataset}")

        try:
            # Load dataset using ir_datasets - this will download if needed
            # ir_datasets handles caching automatically
            # Actual data download happens lazily when iterating
            self.ir_datasets.load(dataset)
            log.info(f"Successfully loaded dataset: {dataset}")
            if progress_callback is not None:
                progress_callback(0, 0, f"Dataset metadata ready for {dataset}")

        except Exception:
            log.exception(f"Failed to download FTS dataset '{dataset}'")
            raise

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        """For ir_datasets, we don't validate against remote files"""
        # ir_datasets handles its own caching and validation
        return local.exists() and local.stat().st_size > 0
