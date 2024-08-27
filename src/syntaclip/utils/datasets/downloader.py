import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import img2dataset
import requests
from utils.clogging import getColoredLogger

global_logger = getColoredLogger(__name__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass(frozen=True)
class Img2DatasetKwargs:
    url_list: str
    image_size: int = 256
    output_folder: str = "images"
    processes_count: int = 1
    resize_mode: str = "border"
    resize_only_if_bigger: bool = False
    upscale_interpolation: str = "lanczos"
    downscale_interpolation: str = "area"
    encode_quality: int = 95
    encode_format: str = "jpg"
    skip_reencode: bool = False
    output_format: str = "files"
    input_format: str = "txt"
    url_col: str = "url"
    caption_col: Optional[str] = None
    bbox_col: Optional[str] = None
    thread_count: int = 256
    number_sample_per_shard: int = 10000
    extract_exif: bool = True
    save_additional_columns: Optional[list[str]] = None
    timeout: int = 10
    enable_wandb: bool = False
    wandb_project: str = "img2dataset"
    oom_shard_count: int = 5
    compute_hash: Optional[str] = "sha256"
    verify_hash: Optional[list[str]] = None
    distributor: str = "multiprocessing"
    subjob_size: int = 1000
    retries: int = 0
    disable_all_reencoding: bool = False
    min_image_size: int = 0
    max_image_area: float = float("inf")
    max_aspect_ratio: float = float("inf")
    incremental_mode: str = "incremental"
    max_shard_retry: int = 1
    user_agent_token: Optional[str] = None
    disallowed_header_directives: Optional[list[str]] = None

    def as_dict(self):
        return asdict(self)


def download_file(url: str, output_path: Path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return output_path


def download_img2dataset(kwargs: Img2DatasetKwargs):
    img2dataset.download(**kwargs.as_dict())


def download_cc3m(
    output_folder: Path,
    train_tsv_path: Path,
    val_tsv_path: Path,
    logger: Optional[logging.Logger] = None,
):
    """
    Download Conceptual Captions 3M dataset to output_folder
    Args:
        output_folder: Path to save the dataset
        train_tsv_path: Path to training tsv file
        val_tsv_path: Path to validation tsv file
        logger: Logger to use

    Usage:
    >>> # Already downloaded the tsv files as Training.tsv and Validation.tsv
    >>> CC3M_DATA_DIR = Path.home() / "datasets" / "CC3M"
    >>> download_cc3m(
    >>>    output_folder=CC3M_DATA_DIR,
    >>>    train_tsv_path=CC3M_DATA_DIR / "tsv" / "Training.tsv",
    >>>    val_tsv_path=CC3M_DATA_DIR / "tsv" / "Validation.tsv",
    >>>    logger=logger,
    >>> )
    """
    if logger is None:
        global global_logger
        logger = global_logger
    logger.info(f"{output_folder=}")
    url = "https://ai.google.com/research/ConceptualCaptions/download"
    assert (
        train_tsv_path.exists()
    ), f"{train_tsv_path=} does not exist! please download it from {url=}"
    assert (
        val_tsv_path.exists()
    ), f"{val_tsv_path=} does not exist! please download it from {url=}"

    def _download(tsv_path: Path, _output_folder: Path):
        with open(tsv_path, "r+") as f:
            f.seek(0)
            f.write("caption\turl\n")
        (_output_folder / "images").mkdir(parents=True, exist_ok=True)
        kwargs = Img2DatasetKwargs(
            url_list=str(tsv_path),
            input_format="tsv",
            url_col="url",
            caption_col="caption",
            output_format="webdataset",
            output_folder=str(_output_folder / "images"),
            processes_count=16,
            thread_count=64,
            image_size=256,
            enable_wandb=True,
        )
        download_img2dataset(kwargs)

    logger.info(f"Downloading Training Data to {output_folder=}")
    _download(train_tsv_path, output_folder / "train")
    logger.info(f"Downloading Validation Data to {output_folder=}")
    _download(val_tsv_path, output_folder / "val")
    logger.info(f"Downloaded CC3M to {output_folder=}, Done!")
