# %%

import json
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import img2dataset
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import Pipeline, pipeline_def
from nvidia.dali.data_node import DataNode
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from utils.clogging import getColoredLogger

from ..magicvalues import MagicNumbers
from .dalidataset import DaliDataset
from .downloader import Img2DatasetKwargs

logger = getColoredLogger(__name__)

__all__ = ["CC3MDataset"]


class CC3MDataset(DaliDataset):
    """CC3M Dataset.

    Usage:
    >>> CC3M_DATA_DIR = Path.home() / "datasets" / "WebDataset" / "CC3M"
    >>> dataset = CC3MDataset(CC3M_DATA_DIR, split="train")
    >>> dataset.download(enable_wandb=True)  # need to download the dataset first, which will take some time
    >>> dataloader = dataset.build_dataloader(
    ...     batch_size=64,
    ...     num_threads=4,
    ...     shuffle=False,
    ... )
    >>> for batch, *_ in dataloader:
    ...     images, metadata = dataset.batch_extract(batch)
    ...     captions = [datum["caption"] for datum in metadata]
    ...     print(f"{images.shape=} {captions[0]=}")
    ...     break
    """

    def __init__(
        self, dataset_dir: str | Path, split: Literal["train", "val"] = "train"
    ):
        super().__init__(dataset_dir)
        self.split = split
        self.split_dataset_dir = self.dataset_dir / split
        self.split_dataset_dir.mkdir(parents=True, exist_ok=True)

    def download(self, enable_wandb: bool = False):
        assert (
            self.dataset_dir / "train.tsv"
        ).exists(), f"{str(self.dataset_dir / 'train.tsv')} does not exist!"
        assert (
            self.dataset_dir / "val.tsv"
        ).exists(), f"{str(self.dataset_dir / 'val.tsv')} does not exist!"

        def _download(tsv_path: Path, _output_folder: Path):
            with open(tsv_path, "r+") as f:
                f.seek(0)
                if f.read(7) != "caption":
                    f.write("caption\turl\n")
            _output_folder.mkdir(parents=True, exist_ok=True)

            kwargs = Img2DatasetKwargs(
                url_list=str(tsv_path),
                input_format="tsv",
                url_col="url",
                caption_col="caption",
                output_format="webdataset",
                output_folder=str(_output_folder),
                processes_count=16,
                thread_count=64,
                image_size=256,
                enable_wandb=enable_wandb,
            )
            img2dataset.download(**kwargs.as_dict())

        if self.split == "val" and len(tuple(self.split_dataset_dir.iterdir())) == 0:
            _download(self.dataset_dir / "val.tsv", self.dataset_dir / "val")

        elif (
            self.split == "train" and len(tuple(self.split_dataset_dir.iterdir())) == 0
        ):
            _download(self.dataset_dir / "train.tsv", self.dataset_dir / "train")

        self.wds2idx()

    def build_dataloader(
        self,
        ext: list[str] = ["jpg", "json"],
        shuffle: bool = False,
        image_size: int | tuple[int, int] = 224,
        device: Literal["cpu", "gpu", "mixed"] = "gpu",
        name: str = "CC3MReader",
        normalize: bool = False,
        *,
        enable_conditionals: bool = False,
        batch_size: int = 1,
        num_threads: int = 1,
        num_shards: int = 1,
        device_id: int = 0,
        shard_id: int = 0,
        seed: Optional[int] = None,
        exec_pipelined: bool = True,
        prefetch_queue_depth: int | tuple[int, int] = 1,
        exec_async: bool = True,
        bytes_per_sample: int = 0,
        set_affinity: bool = False,
        max_streams: int = -1,
        default_cuda_stream_priority: int = 0,
        enable_memory_stats: bool = False,
        enable_checkpointing: bool = False,
        checkpoint: Optional[Any] = None,
        py_num_workers: int = 1,
        py_start_method: str = "fork",
        py_callback_pickler: Optional[Any] = None,
        output_dtype: types.DALIDataType | tuple[types.DALIDataType, ...] | None = None,
        output_ndim: int | tuple[int, ...] | None = None,
        reader_name: str = "CC3MReader",
        auto_reset: bool = True,
        fill_last_batch: Optional[bool] = None,
        dynamic_shape: bool = False,
        last_batch_padded: bool = False,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.DROP,
        prepare_first_batch: bool = False,
    ) -> DALIGenericIterator:
        assert self.tarfiles, "No tarfiles found!"
        assert self.idxfiles, "No idxfiles found!"
        self.pipeline = self.build_pipeline(
            ext=ext,
            random_shuffle=shuffle,
            image_size=image_size,
            device=device,
            name=name,
            normalize=normalize,
            enable_conditionals=enable_conditionals,
            batch_size=batch_size,
            num_threads=num_threads,
            num_shards=num_shards,
            shard_id=shard_id,
            device_id=device_id,
            seed=seed,
            exec_pipelined=exec_pipelined,
            prefetch_queue_depth=prefetch_queue_depth,
            exec_async=exec_async,
            bytes_per_sample=bytes_per_sample,
            set_affinity=set_affinity,
            max_streams=max_streams,
            default_cuda_stream_priority=default_cuda_stream_priority,
            enable_memory_stats=enable_memory_stats,
            enable_checkpointing=enable_checkpointing,
            checkpoint=checkpoint,
            py_num_workers=py_num_workers,
            py_start_method=py_start_method,
            py_callback_pickler=py_callback_pickler,
            output_dtype=output_dtype,
            output_ndim=output_ndim,
        )
        return DALIGenericIterator(
            self.pipeline,
            ext,
            auto_reset=auto_reset,
            fill_last_batch=fill_last_batch,
            dynamic_shape=dynamic_shape,
            last_batch_padded=last_batch_padded,
            last_batch_policy=last_batch_policy,
            prepare_first_batch=prepare_first_batch,
            reader_name=reader_name,
        )

    def batch_extract(self, batch: tuple[Any, ...]):
        if isinstance(batch, list):
            batch = batch[0]
        images, metadata = batch["jpg"], batch["json"].numpy()
        metadata = [
            json.loads("".join([chr(o) for o in row.tolist() if o != 0]))
            for row in metadata
        ]
        return images, metadata

    def build_pipeline(
        self,
        normalize: bool = False,
        *,
        ext: list[str] = ["jpg", "json"],
        random_shuffle: bool = True,
        image_size: int | tuple[int, int] = 224,
        device: Literal["cpu", "gpu", "mixed"] = "gpu",
        name: str = "CC3MReader",
        num_shards: int = 1,
        shard_id: int = 0,
        seed: int = -1,
        **kwargs,
    ) -> Callable[
        [Callable[..., DataNode | tuple[DataNode, ...]]], Callable[..., Pipeline]
    ]:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        @pipeline_def
        def _build():
            image, info = fn.readers.webdataset(
                ext=ext,
                paths=self.tarfiles,
                index_paths=self.idxfiles,
                random_shuffle=random_shuffle,
                missing_component_behavior="error",
                name=name,
                num_shards=num_shards,
                shard_id=shard_id,
                seed=seed,
            )
            image = fn.decoders.image(image, device="mixed", output_type=types.RGB)
            image = (
                fn.resize(
                    image,
                    device=device,
                    resize_x=image_size[0],
                    resize_y=image_size[1],
                )
                / 255.0
            )
            if normalize:
                image = fn.crop_mirror_normalize(
                    image,
                    dtype=types.FLOAT,
                    device=device,
                    mean=MagicNumbers.RGB_CLIP_IMAGE_MEAN,
                    std=MagicNumbers.RGB_CLIP_IMAGE_STD,
                )
            else:
                image = fn.transpose(image, device=device, perm=[2, 0, 1])
            return image, fn.pad(info, device="cpu")

        return _build(**kwargs)
