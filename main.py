# %%
import gc
import gzip
import html
import json
import os
import random
import string
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import ftfy
import img2dataset
import matplotlib.pyplot as plt
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import regex as re
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from nvidia.dali import Pipeline, pipeline_def
from nvidia.dali.data_node import DataNode
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from PIL import Image
from tap import Tap
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.clogging import getColoredLogger
from utils.initialize import initializer
from utils.jupyter import is_jupyter
from wds2idx import IndexCreator

import wandb

logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")
PROJECT_ROOT = initializer(globals(), logger=logger)
HUGGINGFACE_HUB_CACHE = os.environ.get("HUGGINGFACE_HUB_CACHE", None)
logger.info(f"{PROJECT_ROOT=}")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_CUDA_AVAILABLE = torch.cuda.is_available()
if IS_DISTRIBUTED:
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    WORLD_SIZE = torch.cuda.device_count()
    torch.cuda.set_device(LOCAL_RANK)
    logger.info(f"Parallel INFO: {LOCAL_RANK=}, {WORLD_SIZE=}")

DEVICE = torch.device("cuda" if IS_CUDA_AVAILABLE else "cpu")

torch._dynamo.reset()
torch.set_float32_matmul_precision("high")

# fix seed
INT32_MAX = 2**32 - 1
SEED = None
if SEED is None:
    SEED = torch.seed()

# seedで使える値がnumpyではint32, Daliではint64までっぽい
SEED = (SEED + LOCAL_RANK) % INT32_MAX

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
logger.info(f"{SEED=}, {IS_CUDA_AVAILABLE=}, {DEVICE=}")


# %%
# 定数/初期値定義
@dataclass
class _Constants:
    DEFAULT_CONTEXT_LENGTH = 77
    DEFAULT_BPE_PATH = "default_bpe.txt.gz"
    DEFAULT_VOCAB_SIZE = 49408
    DEFAULT_CC3M_DIR = Path.home() / "datasets" / "WebDataset" / "CC3M"
    DEFAULT_CC12M_DIR = Path.home() / "datasets" / "WebDataset" / "CC12M"
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_NUM_THREADS = 16
    DEFAULT_DEVICE_ID = LOCAL_RANK
    DEFAULT_NUM_SHARDS = WORLD_SIZE
    DEFAULT_SHARD_ID = LOCAL_RANK
    DEFAULT_BATCHSIZE = 64
    DEFAULT_NUMTHREADS = 4
    DEFAULT_IMAGE_DIR = PROJECT_ROOT / "ignores" / "images"
    DEFAULT_TOKENIZER_NAME = "o200k_base"

    RGB_CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    RGB_CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


const = _Constants()


def str2dtype(str_dtype: str):
    dtype_map = {
        "fp32": "float32",
        "fp16": "float16",
        "fp64": "float64",
        "bf16": "bfloat16",
    }
    dtype = getattr(torch, dtype_map.get(str_dtype, str_dtype), None)
    if dtype is None:
        raise ValueError(f"Invalid dtype: {str_dtype}")
    return dtype


def str2dataset_path(dataset_name: str):
    dataset_path = {
        "CC3M": const.DEFAULT_CC3M_DIR,
        "CC12M": const.DEFAULT_CC12M_DIR,
    }.get(dataset_name, None)
    if dataset_path is None:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return dataset_path


class ModelConfigArgs(Tap):
    model_arch: Literal["SyntaCLIP", "CLIP"] = "CLIP"
    attn_dropout_p: float = 0.25
    gate_dropout_p: float = 0.25
    pretrained: Optional[str] = None
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
    # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia
    amp_device_type: Literal["cpu", "cuda"] = "cuda"
    amp_dtype: torch.dtype = torch.float32
    use_wandb: bool = False
    project_name: Optional[str] = None
    save_dir: Path = (
        Path(globals().get("__file__", Path.cwd() / "main.py")).parent
        / "outputs"
        / "models"
    )

    def configure(self):
        self.add_argument(
            "--amp_dtype", type=str2dtype, help="AMP dtype", default=torch.bfloat16
        )


class TrainArgs(ModelConfigArgs):
    dataset_name: Literal["CC3M", "CC12M"] = "CC3M"
    dataset_path: Path = const.DEFAULT_CC3M_DIR
    batch_size: int = const.DEFAULT_BATCH_SIZE
    num_threads: int = const.DEFAULT_NUM_THREADS
    num_epochs: int = 1
    break_num_epoch: Optional[int] = None
    break_batch_index: Optional[int] = None
    seed: int = SEED
    shuffle: bool = True
    grad_norm: float = 1.0
    compile_mode: Optional[
        Literal[
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ]
    ] = None
    save_interval: int = 10


class InferArgs(ModelConfigArgs):
    save_filename: str = "inference_results.png"


class EvalArgs(ModelConfigArgs): ...


class TypedArgs(Tap):
    def configure(self):
        self.add_subparsers(dest="mode", help="Mode of operation")
        self.add_subparser("train", TrainArgs, help="Training arguments")
        self.add_subparser("inference", InferArgs, help="Inference arguments")
        self.add_subparser("eval", EvalArgs, help="Evaluation arguments")


# %%


# Tokenizer定義
class SimpleTokenizer:
    def __init__(
        self,
        bpe_path: str = const.DEFAULT_BPE_PATH,
        additional_special_tokens: Optional[list[str]] = None,
        context_length: Optional[int] = const.DEFAULT_CONTEXT_LENGTH,
        clean: str = "lower",
        reduction_mask: str = "",
    ):
        self.byte_encoder = SimpleTokenizer.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = self.get_clean_fn(clean)
        self.reduction_fn = (
            self.get_reduction_mask_fn(reduction_mask) if reduction_mask else None
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = SimpleTokenizer.get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = SimpleTokenizer.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def __call__(
        self, texts: str | list[str], context_length: Optional[int] = None
    ) -> torch.LongTensor:
        """Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length"

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
            )

        all_tokens = [
            [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result

    @lru_cache()
    @staticmethod
    def bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a significant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    @staticmethod
    def canonicalize_text(
        text,
        *,
        keep_punctuation_exact_string=None,
        trans_punctuation: dict = str.maketrans("", "", string.punctuation),
    ):
        """Returns canonicalized `text` (lowercase and punctuation removed).

        From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

        Args:
        text: string to be canonicalized.
        keep_punctuation_exact_string: If provided, then this exact string kept.
            For example providing '{}' will keep any occurrences of '{}' (but will
            still remove '{' and '}' that appear separately).
        """
        text = text.replace("_", " ")
        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                part.translate(trans_punctuation)
                for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = text.translate(trans_punctuation)
        text = text.lower()
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def get_pairs(word):
        """Return set of symbol pairs in a word.
        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    @staticmethod
    def whitespace_clean(text):
        text = " ".join(text.split())
        text = text.strip()
        return text

    @staticmethod
    def _clean_canonicalize(x):
        # basic, remove whitespace, remove punctuation, lower case
        return SimpleTokenizer.canonicalize_text(SimpleTokenizer.basic_clean(x))

    @staticmethod
    def _clean_lower(x):
        # basic, remove whitespace, lower case
        return SimpleTokenizer.whitespace_clean(SimpleTokenizer.basic_clean(x)).lower()

    @staticmethod
    def _clean_whitespace(x):
        # basic, remove whitespace
        return SimpleTokenizer.whitespace_clean(SimpleTokenizer.basic_clean(x))

    @staticmethod
    def get_clean_fn(type: str):
        if type == "canonicalize":
            return SimpleTokenizer._clean_canonicalize
        elif type == "lower":
            return SimpleTokenizer._clean_lower
        elif type == "whitespace":
            return SimpleTokenizer._clean_whitespace
        else:
            assert False, f"Invalid clean function ({type})."


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


class DaliDataset(ABC):
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir).absolute()
        self.split_dataset_dir = self.dataset_dir / "split"

    @property
    def tarfiles(self):
        return tuple(map(str, sorted(self.split_dataset_dir.glob("*.tar"))))

    @property
    def idxfiles(self):
        return tuple(map(str, sorted(self.split_dataset_dir.glob("*.idx"))))

    @abstractmethod
    def download(self):
        pass

    def wds2idx(self):
        assert self.tarfiles, "No tar files found. Is the dataset downloaded?"
        if self.idxfiles:
            return

        for tar in map(Path, self.tarfiles):
            with IndexCreator(str(tar), str(tar.with_suffix(".idx"))) as c:
                logger.info(f"Creating index for {tar} ...")
                c.create_index()

    @abstractmethod
    def build_dataloader(self) -> DALIGenericIterator:
        pass

    @abstractmethod
    def build_pipeline(self):
        pass


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
        device: Literal["cpu", "gpu", "mixed"] = "gpu"
        if torch.cuda.is_available()
        else "cpu",
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
        device: Literal["cpu", "gpu", "mixed"] = "gpu"
        if torch.cuda.is_available()
        else "cpu",
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
            image = fn.decoders.image(
                image,
                device="mixed" if torch.cuda.is_available() else "cpu",
                output_type=types.RGB,
            )
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
                    mean=const.RGB_CLIP_IMAGE_MEAN,
                    std=const.RGB_CLIP_IMAGE_STD,
                )
            else:
                image = fn.transpose(image, device=device, perm=[2, 0, 1])
            return image, fn.pad(info, device="cpu")

        return _build(**kwargs)


class CC12MDataset(DaliDataset):
    r"""CC12M Dataset. instructions: https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md
    1. Download the dataset.
    2. Run `sed -i '1s/^/url\tcaption\n/' cc12m.tsv` to add the header.

    """

    def __init__(self, dataset_dir: str | Path):
        super().__init__(dataset_dir)
        self.split_dataset_dir = self.dataset_dir / "data"
        self.split_dataset_dir.mkdir(parents=True, exist_ok=True)

    def download(self, enable_wandb: bool = False, tsv: Optional[Path] = None):
        if tsv is None:
            tsv = self.dataset_dir / "cc12m.tsv"
        assert (tsv).exists(), f"{str(tsv)} does not exist!"

        def _download(tsv_path: Path, _output_folder: Path):
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

        if len(tuple(self.split_dataset_dir.iterdir())) == 0:
            _download(tsv, self.dataset_dir / "data")

        self.wds2idx()

    def build_dataloader(
        self,
        ext: list[str] = ["jpg", "json"],
        shuffle: bool = False,
        image_size: int | tuple[int, int] = 224,
        device: Literal["cpu", "gpu", "mixed"] = "gpu"
        if torch.cuda.is_available()
        else "cpu",
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
        device: Literal["cpu", "gpu", "mixed"] = "gpu"
        if torch.cuda.is_available()
        else "cpu",
        name: str = "CC12MReader",
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
            image = fn.decoders.image(
                image,
                device="mixed" if torch.cuda.is_available() else "cpu",
                output_type=types.RGB,
            )
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
                    mean=const.RGB_CLIP_IMAGE_MEAN,
                    std=const.RGB_CLIP_IMAGE_STD,
                )
            else:
                image = fn.transpose(image, device=device, perm=[2, 0, 1])
            return image, fn.pad(info, device="cpu")

        return _build(**kwargs)


# 関数定義
@torch.jit.script
def merge_masks(
    num_heads: int,
    attention_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    query: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[int]]:
    r"""Determine mask type and combine masks if necessary.

    If only one mask is provided, that mask
    and the corresponding mask type will be returned. If both masks are provided, they will be both
    expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
    and mask type 2 will be returned
    Args:
        attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
        key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
        query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
    Returns:
        merged_mask: merged mask
        mask_type: merged mask type (0, 1, or 2)
    """
    merged_mask: Optional[torch.Tensor] = None
    # mask_type = 1: key_padding_mask, 2: attn_mask, 3: key_padding_mask + attn_mask
    mask_type: Optional[int] = None

    if key_padding_mask is not None:
        mask_type = 1
        merged_mask = key_padding_mask

    if attention_mask is not None:
        batch_size, seq_len, _ = query.shape
        mask_type = 2

        if attention_mask.dim() == 3:
            attention_mask_expanded = attention_mask.view(
                batch_size, -1, seq_len, seq_len
            )
        else:
            attention_mask_expanded = attention_mask.view(
                1, 1, seq_len, seq_len
            ).expand(batch_size, num_heads, -1, -1)
        merged_mask = attention_mask_expanded

        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.view(
                batch_size, 1, 1, seq_len
            ).expand(-1, num_heads, -1, -1)
            merged_mask = attention_mask_expanded + key_padding_mask_expanded

    return merged_mask, mask_type


def get_tensor_eps_for_jit(
    x: torch.Tensor,
    epsf16: float = torch.finfo(torch.float16).eps,
    epsbf16: float = torch.finfo(torch.bfloat16).eps,
    epsf32: float = torch.finfo(torch.float32).eps,
    epsf64: float = torch.finfo(torch.float64).eps,
) -> float:
    # Match aren't supported in jit
    if x.dtype == torch.float16:
        return epsf16
    elif torch.bfloat16:
        return epsbf16
    elif torch.float32:
        return epsf32
    elif torch.float64:
        return epsf64
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")


@torch.jit.script
def syntactic_multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_attn_heads: int,
    num_gate_heads: int,
    in_proj_weight: Optional[torch.Tensor],
    in_proj_bias: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    attn_gate: Optional[torch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    query_proj_weight: Optional[torch.Tensor] = None,
    key_proj_weight: Optional[torch.Tensor] = None,
    value_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Forward method for SyntacticMultiHeadAttention.
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        attn_gate: gate tensor for attention. # ADDED

        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if attn_gate is None:
        return F.multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim_to_check,
            num_attn_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=query_proj_weight,
            k_proj_weight=key_proj_weight,
            v_proj_weight=value_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    if F.has_torch_function(tens_ops):
        return F.handle_torch_function(
            syntactic_multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_attn_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=query_proj_weight,
            k_proj_weight=key_proj_weight,
            v_proj_weight=value_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = F._mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_attn_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query.unsqueeze_(1)
        key.unsqueeze_(1)
        value.unsqueeze_(1)
        if key_padding_mask is not None:
            key_padding_mask.unsqueeze_(0)

    # set up shape vars
    target_length, batch_size, embed_dim = query.shape
    source_length, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_attn_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_attn_heads
    assert (
        head_dim * num_attn_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_attn_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    # compute in-projection
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        query, key, value = F._in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias
        )
    else:
        assert (
            query_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            key_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            value_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            query_bias = key_bias = value_bias = None
        else:
            query_bias, key_bias, value_bias = in_proj_bias.chunk(3)
        query, key, value = F._in_projection(
            query,
            key,
            value,
            query_proj_weight,
            key_proj_weight,
            value_proj_weight,
            query_bias,
            key_bias,
            value_bias,
        )

    # prep attention mask
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (target_length, source_length)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (
                batch_size * num_attn_heads,
                target_length,
                source_length,
            )
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        key = torch.cat([key, bias_k.repeat(1, batch_size, 1)])
        value = torch.cat([value, bias_v.repeat(1, batch_size, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    # reshape q, k, v for multihead attention and make them batch first
    query = query.view(target_length, batch_size * num_attn_heads, head_dim).transpose(
        0, 1
    )
    if static_k is None:
        key = key.view(key.shape[0], batch_size * num_attn_heads, head_dim).transpose(
            0, 1
        )
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        key = static_k
    if static_v is None:
        value = value.view(
            value.shape[0], batch_size * num_attn_heads, head_dim
        ).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        value = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (batch_size * num_attn_heads, 1, head_dim)
        key = torch.cat(
            [key, torch.zeros(zero_attn_shape, dtype=key.dtype, device=key.device)],
            dim=1,
        )
        value = torch.cat(
            [
                value,
                torch.zeros(zero_attn_shape, dtype=value.dtype, device=value.device),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    source_length = key.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = (
            key_padding_mask.view(batch_size, 1, 1, source_length)
            .expand(-1, num_attn_heads, -1, -1)
            .reshape(batch_size * num_attn_heads, 1, source_length)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # (deep breath) calculate attention and out projection
    batch_attn_heads, target_length, head_dim = query.shape
    scaled_query = query * ((1.0 / float(head_dim)) ** 0.5)

    assert not (
        is_causal and attn_mask is None
    ), "FIXME: is_causal not implemented for need_weights"

    # ADDED: Compute the attention weights
    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_miltihead_attention.py#L329
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(
            attn_mask, scaled_query, key.transpose(-2, -1)
        )
    else:
        attn_output_weights = torch.bmm(scaled_query, key.transpose(-2, -1))

    # attn_output_weights: (batch_size * num_attn_heads, target_length, source_length)

    attn_output_weights_dtype = attn_output_weights.dtype
    attn_output_weights = F.softmax(attn_output_weights, dim=-1).type(torch.float32)

    # ADDED: Adjust attn_gates
    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L86
    num_heads_diff = num_attn_heads - num_gate_heads

    if num_heads_diff > 0:
        attn_gate = (
            torch.cat(
                [
                    attn_gate.view(
                        batch_size, num_gate_heads, target_length, source_length
                    ),
                    attn_gate.new_ones(
                        (
                            batch_size,
                            num_heads_diff,
                            target_length,
                            source_length,
                        )
                    ),
                ],
                dim=1,
            )
            .view(batch_size * num_attn_heads, target_length, source_length)
            .contiguous()
        )

    # Apply attn_gate
    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_miltihead_attention.py#L361
    attn_output_weights = attn_output_weights * attn_gate.type_as(attn_output_weights)
    attn_output_weights = attn_output_weights / (
        attn_output_weights.sum(dim=-1, keepdim=True)
        + get_tensor_eps_for_jit(attn_output_weights)
    )
    attn_output_weights = attn_output_weights.type(attn_output_weights_dtype)

    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_miltihead_attention.py#L377
    attn_output = torch.bmm(attn_output_weights, value)

    attn_output = (
        attn_output.transpose(0, 1)
        .contiguous()
        .view(target_length * batch_size, embed_dim)
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(target_length, batch_size, attn_output.size(1))

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(
        batch_size, num_attn_heads, target_length, source_length
    )
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights


# モデル定義
class PatchDropout(nn.Module):
    """Patch Dropout
    https://arxiv.org/abs/2212.00794
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L49
    Args:
        p (float): Probability of an element to be zeroed
        exclude_first_token (bool): Exclude first token
    """

    def __init__(self, p: float = 0.0, exclude_first_token=True):
        super().__init__()
        assert 0 <= p < 1.0
        self.prob = p
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class CastLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype).
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L24

    from https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L157
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        x = super().forward(x.type(torch.float32))
        return x.type(orig_type)


class LayerScale(nn.Module):
    """Layer scale
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L39
    Args:
        embed_dim (int): Embedding dimension
        init_scale_ratio (float): Initial scale ratio
        inplace (bool): Inplace operation
    """

    def __init__(
        self, embed_dim: int, init_scale_ratio: float = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_scale_ratio * torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SyntacticDistanceGate1D(nn.Module):
    """
    Syntactic Distance Gate
    - https://aclanthology.org/2021.acl-srw.33/
    """

    def __init__(
        self,
        embed_dim: int,
        num_lookback_range: int = 3,
        num_gate_heads: int = 2,
        *,
        tau: float = 1.0,
        dropout_p: float = 0.0,
        distance_activation_module: nn.Module = nn.Tanh,
        mask_triu: bool = False,
    ):
        super().__init__()
        self.lookback_range = num_lookback_range
        self.tau = tau
        self.num_gate_heads = num_gate_heads
        self.conv = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(
                embed_dim,
                num_gate_heads,
                num_lookback_range,
                padding=num_lookback_range,
            ),
        )
        self.distance_activation = distance_activation_module()
        self.mask_triu = mask_triu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)
        # x: (batch_size, embed_dim, seq_len)
        batch_size, embed_dim, seq_len = x.size()

        # distance: Syntactic Distance [d_i, ...]: i番目の単語の構文距離 (構文高？)
        # distance := distance  (batch_size, seq_len, 1)
        # distance[i] = \tanh(W_D [k_{i-M}, k_{i-M+1}, ..., K_{i}]^{\top} + b_D)
        # conv_input: (batch_size, embed_dim, seq_len)
        distance = self.conv(x)
        # disttance : (batch_size, distance_dim, seq_len + lookback_range)
        distance = distance[:, :, 1 : -self.lookback_range]
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L60
        distance = self.distance_activation(distance)
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L106
        # distance: (batch_size, num_gate_heads, seq_len)
        distance = distance.view(batch_size * self.num_gate_heads, -1, 1).contiguous()
        # distance: (batch_size * num_gates_heads, seq_len, 1)
        # Compute Span Logits
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L41
        alpha = (F.hardtanh((distance - distance.transpose(2, 1)) * self.tau) + 1) / 2
        # alpha: (batch_size, seq_len, seq_len), 0 <= alpha <= 1
        lower_tri = (
            (alpha.tril(diagonal=-1) + torch.ones_like(alpha).triu(diagonal=0))
            .flip([-1])
            .cumprod(dim=-1)
            .flip([-1])
        )
        if self.mask_triu:
            return lower_tri, distance

        upper_tri = (
            torch.ones_like(alpha).tril(diagonal=0) + alpha.triu(diagonal=1)
        ).cumprod(dim=-1)

        gate = lower_tri * upper_tri

        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L105
        distance = (
            distance.contiguous()
            .view(batch_size, self.num_gate_heads, seq_len, 1)
            .mean(dim=1)
        )
        # gate := gate  (batch_size * num_gate_heads, seq_len, seq_len), 0 <= gate <= 1
        # distance := distance  (batch_size, seq_len, 1), -1 <= distance <= 1
        return gate, distance


class SyntacticDistanceGate2D(nn.Module):
    """
    Syntactic Distance Gate
    - https://aclanthology.org/2021.acl-srw.33/
    """

    def __init__(
        self,
        embed_dim: int,
        num_lookback_range: int = 3,
        num_gate_heads: int = 2,
        *,
        tau: float = 1.0,
        dropout_p: float = 0.0,
        distance_activation_module: nn.Module = nn.Tanh,
        mask_triu: bool = False,
    ):
        super().__init__()
        self.lookback_range = num_lookback_range
        self.tau = tau
        self.num_gate_heads = num_gate_heads
        self.conv = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(
                embed_dim,
                num_gate_heads,
                num_lookback_range,
                padding=num_lookback_range,
            ),
        )
        self.distance_activation = distance_activation_module()
        self.mask_triu = mask_triu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)
        # x: (batch_size, embed_dim, seq_len)
        batch_size, embed_dim, seq_len = x.size()

        # distance: Syntactic Distance [d_i, ...]: i番目の単語の構文距離 (構文高？)
        # distance := distance  (batch_size, seq_len, 1)
        # distance[i] = \tanh(W_D [k_{i-M}, k_{i-M+1}, ..., K_{i}]^{\top} + b_D)
        # conv_input: (batch_size, embed_dim, seq_len)
        distance = self.conv(x)
        # disttance : (batch_size, distance_dim, seq_len + lookback_range)
        distance = distance[:, :, 1 : -self.lookback_range]
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L60
        distance = self.distance_activation(distance)
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L106
        # distance: (batch_size, num_gate_heads, seq_len)
        distance = distance.view(batch_size * self.num_gate_heads, -1, 1).contiguous()
        # distance: (batch_size * num_gates_heads, seq_len, 1)
        # Compute Span Logits
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L41
        alpha = (F.hardtanh((distance - distance.transpose(2, 1)) * self.tau) + 1) / 2
        # alpha: (batch_size, seq_len, seq_len), 0 <= alpha <= 1
        lower_tri = (
            (alpha.tril(diagonal=-1) + torch.ones_like(alpha).triu(diagonal=0))
            .flip([-1])
            .cumprod(dim=-1)
            .flip([-1])
        )
        if self.mask_triu:
            return lower_tri, distance

        upper_tri = (
            torch.ones_like(alpha).tril(diagonal=0) + alpha.triu(diagonal=1)
        ).cumprod(dim=-1)

        gate = lower_tri * upper_tri

        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L105
        distance = (
            distance.contiguous()
            .view(batch_size, self.num_gate_heads, seq_len, 1)
            .mean(dim=1)
        )
        # gate := gate  (batch_size * num_gate_heads, seq_len, seq_len), 0 <= gate <= 1
        # distance := distance  (batch_size, seq_len, 1), -1 <= distance <= 1
        return gate, distance


class MultiheadAttention(nn.Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout_p: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = True,
        device: torch.device | str = None,
        dtype: torch.dtype = None,
    ):
        assert embed_dim > 0, f"embed_dim must be greater than 0, got {embed_dim}"
        assert num_heads > 0, f"num_heads must be greater than 0, got {num_heads}"
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.batch_first: bool = batch_first

        self.kdim: int = kdim if kdim is not None else embed_dim
        self.vdim: int = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim: bool = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )

        self.num_heads: int = num_heads
        self.dropout_p: float = dropout_p
        self.head_dim: int = embed_dim // num_heads

        self.in_proj_weight: Optional[nn.Parameter]
        self.q_proj_weight: Optional[nn.Parameter]
        self.k_proj_weight: Optional[nn.Parameter]
        self.v_proj_weight: Optional[nn.Parameter]

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, embed_dim), device=device, dtype=dtype)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), device=device, dtype=dtype)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), device=device, dtype=dtype)
            )
            self.register_buffer("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), device=device, dtype=dtype)
            )
            self.register_buffer("p_proj_weight", None)
            self.register_buffer("k_proj_weight", None)
            self.register_buffer("v_proj_weight", None)

        self.in_proj_bias: Optional[nn.Parameter]
        if bias:
            self.in_proj_bias = nn.Parameter(
                torch.empty(3 * embed_dim, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj: nn.Linear = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self.add_bias_kv: bool = add_bias_kv
        if add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.empty(1, 1, embed_dim), device=device, dtype=dtype
            )
            self.bias_v = nn.Parameter(
                torch.empty(1, 1, embed_dim), device=device, dtype=dtype
            )
        else:
            self.register_buffer("bias_k", None)
            self.register_buffer("bias_v", None)

        self.add_zero_attn: bool = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(
        self, weight_gain: float = 1.0 / (2.0**0.5), bias_constant: float = 0.0
    ):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight, gain=weight_gain)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight, gain=weight_gain)
            nn.init.xavier_uniform_(self.v_proj_weight, gain=weight_gain)
            nn.init.xavier_uniform_(self.q_proj_weight, gain=weight_gain)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, bias_constant)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=weight_gain)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, bias_constant)
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_batched_query = query.dim() == 3
        key = key if key is not None else query
        value = value if value is not None else query
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        merged_mask, mask_type = merge_masks(
            num_heads=self.num_heads,
            attention_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            query=query,
        )
        use_fast_path = any(
            (
                not torch.backends.mha.get_fastpath_enabled(),
                not is_batched_query,
                query is not key or key is not value,
                self.in_proj_weight is None,
                self.in_proj_bias is not None
                and query.dtype != self.in_proj_bias.dtype,
                query.dtype != self.in_proj_weight.dtype,
                self.training,
                self.num_heads % 2 != 0,
                not self.batch_first,
                self.bias_k is not None or self.bias_v is not None,
                self.add_zero_attn,
                not self._qkv_same_embed_dim,
                query.is_nested
                and (key_padding_mask is not None or attn_mask is not None),
                torch.is_autocast_enabled(),
            )
        )

        if (
            not use_fast_path
            and self._qkv_same_embed_dim
            and self.in_proj_bias is not None
        ):
            return torch._native_multi_head_attention(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
                merged_mask,
                need_weights,
                average_attn_weights,
                mask_type,
            )

        assert not (
            query.is_nested or key.is_nested or value.is_nested
        ), "MultiheadAttention does not support NestedTensor."
        if self.batch_first and is_batched_query:
            # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
            assert (
                key.dim() == 3
            ), f"key must have 3 dimensions (batch_size, seq_len, embed_dim), got {key.dim()}"
            assert (
                value.dim() == 3
            ), f"value must have 3 dimensions (batch_size, seq_len, embed_dim), got {value.dim()}"
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

        multi_head_attention_forward_kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout_p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if not self._qkv_same_embed_dim:
            multi_head_attention_forward_kwargs.update(
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            **multi_head_attention_forward_kwargs
        )
        if self.batch_first and is_batched_query:
            # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
            attn_output = attn_output.transpose(1, 0)
        return attn_output, attn_output_weights


class SyntacticMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        num_gate_heads: int = 1,
        *,
        dropout_p: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = True,
        device: torch.device | str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_attn_heads,
            dropout_p=dropout_p,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.num_attn_heads: int = num_attn_heads
        self.num_gate_heads: int = num_gate_heads

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        attn_gate: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        attn_weight_div_delta: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attn_gate is None:
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        is_batched_query = query.dim() == 3
        key = key if key is not None else query
        value = value if value is not None else query

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        merged_mask, mask_type = merge_masks(
            num_heads=self.num_attn_heads,
            attention_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            query=query,
        )
        use_fast_path = any(
            (
                not torch.backends.mha.get_fastpath_enabled(),
                not is_batched_query,
                query is not key or key is not value,
                self.in_proj_weight is None,
                self.in_proj_bias is not None
                and query.dtype != self.in_proj_bias.dtype,
                query.dtype != self.in_proj_weight.dtype,
                self.training,
                self.num_attn_heads % 2 != 0,
                not self.batch_first,
                self.bias_k is not None or self.bias_v is not None,
                self.add_zero_attn,
                not self._qkv_same_embed_dim,
                query.is_nested
                and (key_padding_mask is not None or attn_mask is not None),
                torch.is_autocast_enabled(),
            )
        )

        if (
            not use_fast_path
            and self._qkv_same_embed_dim
            and self.in_proj_bias is not None
        ):
            return torch._native_multi_head_attention(
                query,
                key,
                value,
                self.embed_dim,
                self.num_attn_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
                merged_mask,
                need_weights,
                average_attn_weights,
                mask_type,
            )

        assert not (
            query.is_nested or key.is_nested or value.is_nested
        ), "MultiheadAttention does not support NestedTensor."
        if self.batch_first and is_batched_query:
            # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
            assert (
                key.dim() == 3
            ), f"key must have 3 dimensions (batch_size, seq_len, embed_dim), got {key.dim()}"
            assert (
                value.dim() == 3
            ), f"value must have 3 dimensions (batch_size, seq_len, embed_dim), got {value.dim()}"
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

        syntactic_multi_head_attention_forward_kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_attn_heads=self.num_attn_heads,
            num_gate_heads=self.num_gate_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout_p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if not self._qkv_same_embed_dim:
            syntactic_multi_head_attention_forward_kwargs.update(
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        attn_output, attn_output_weights = syntactic_multi_head_attention_forward(
            **syntactic_multi_head_attention_forward_kwargs
        )
        if self.batch_first and is_batched_query:
            # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
            attn_output = attn_output.transpose(1, 0)
        return attn_output, attn_output_weights


class ResidualAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_p: float = 0.0,
        *,
        res_mlp_dim: Optional[int] = None,
        res_mlp_activation_module: nn.Module = nn.GELU,
        init_layerscale_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        if res_mlp_dim is None:
            res_mlp_dim = embed_dim * 4

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layernorm_1 = CastLayerNorm(normalized_shape=embed_dim)
        self.attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout_p=dropout_p,
        )
        self.layerscale_1 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layerscale_ratio)
            if init_layerscale_ratio is not None
            else nn.Identity()
        )
        self.layernorm_2 = CastLayerNorm(normalized_shape=embed_dim)
        self.res_mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=res_mlp_dim),
            res_mlp_activation_module(),
            nn.Linear(in_features=res_mlp_dim, out_features=embed_dim),
        )
        self.layerscale_2 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layerscale_ratio)
            if init_layerscale_ratio is not None
            else nn.Identity()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_mask = attn_mask.to(query.dtype) if attn_mask is not None else None
        _normed_query = self.layernorm_1(query)
        key = key if key is not None else _normed_query
        value = value if value is not None else _normed_query
        attn_out, attn_weights = self.attention(
            _normed_query,
            key,
            value,
            need_weights=True,
            attn_mask=attn_mask,
        )
        x = query + self.layerscale_1(attn_out)
        x = x + self.layerscale_2(self.res_mlp(self.layernorm_2(x)))
        return x, attn_weights


class ResidualSyntacticAttentionEncoderLayer(ResidualAttentionEncoderLayer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_attn_heads: int = 8,
        num_gate_heads: int = 2,
        num_lookback_range: int = 3,
        tau: float = 10.0,
        attn_dropout_p: float = 0.0,
        gate_dropout_p: float = 0.0,
        *,
        res_mlp_dim: Optional[int] = None,
        res_mlp_activation_module: nn.Module = nn.GELU,
        init_layerscale_ratio: Optional[float] = None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_attn_heads,
            res_mlp_dim=res_mlp_dim,
            res_mlp_activation_module=res_mlp_activation_module,
            init_layerscale_ratio=init_layerscale_ratio,
        )
        self.attention = SyntacticMultiheadAttention(
            embed_dim=embed_dim,
            num_attn_heads=num_attn_heads,
            num_gate_heads=num_gate_heads,
            dropout_p=attn_dropout_p,
        )
        self.gate = SyntacticDistanceGate1D(
            embed_dim=embed_dim,
            num_lookback_range=num_lookback_range,
            num_gate_heads=num_gate_heads,
            tau=tau,
            dropout_p=gate_dropout_p,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        attn_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_mask = attn_mask.to(query.dtype) if attn_mask is not None else None
        _normed_query = self.layernorm_1(query)
        key = key if key is not None else _normed_query
        value = value if value is not None else _normed_query
        attn_out, attn_weights = self.attention(
            _normed_query,
            key,
            value,
            need_weights=True,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
        )
        x = query + self.layerscale_1(attn_out)
        x = x + self.layerscale_2(self.res_mlp(self.layernorm_2(x)))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_p: float = 0.0,
        *,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualAttentionEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if return_all_weights:
            ret_weights = []

        for res_attn_block in self.res_attn_blocks:
            x, attn_weight = res_attn_block(x, attn_mask=attn_mask)
            if return_all_weights:
                ret_weights.append(attn_weight)

        if return_all_weights:
            return x, torch.stack(ret_weights, dim=1)
        return x, attn_weight.unsqueeze(0)


class TextTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_p: float = 0.0,
        *,
        num_layers: int = 12,
        vocab_size: int = 49408,
        vocab_embed_dim: int = 512,
        max_context_length: int = 77,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.vocab_embed_dim = vocab_embed_dim
        self.max_context_length = max_context_length
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, vocab_embed_dim, padding_idx=pad_token_id
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(max_context_length, vocab_embed_dim)
        )

        self.layernorm_post = CastLayerNorm(normalized_shape=vocab_embed_dim)

        self.attn_mask: torch.Tensor
        self.register_buffer(
            "attn_mask",
            torch.zeros(max_context_length, max_context_length)
            .fill_(float("-inf"))
            .triu_(1),
            persistent=False,
        )

        self.head_weight = nn.Parameter(torch.randn(vocab_embed_dim, embed_dim))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, sequence_length]
        """
        x_ = x
        batch_size, sequence_length = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding[:sequence_length]
        x, w = self.transformer(
            x,
            attn_mask=attn_mask or self.attn_mask,
            return_all_weights=return_all_weights,
        )
        x = self.layernorm_post(x)

        # _tokens: unused
        pooled, _tokens = x[torch.arange(batch_size), x_.argmax(dim=-1)], x
        pooled = pooled @ self.head_weight

        return pooled, w


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_p: float = 0.0,
        *,
        input_image_size: int | tuple[int, int] | tuple[int, int, int] = 224,
        patch_embed_dim: int = 768,
        patch_size: tuple[int, int] = (32, 32),
        patch_stride: Optional[tuple[int, int]] = None,
        patch_dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.transformer = TransformerEncoder(
            embed_dim=patch_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )
        self.embed_dim = embed_dim
        self.patch_embed_dim = patch_embed_dim

        # image size adjustment
        if isinstance(input_image_size, int):
            input_image_size = (3, input_image_size, input_image_size)
        elif len(input_image_size) == 2:
            input_image_size = (3, *input_image_size)
        elif len(input_image_size) > 3:
            raise ValueError(
                f"input_image_size must be an integer or a tuple of 2 or 3, got {input_image_size}"
            )

        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size

        self.scale = patch_embed_dim ** (-0.5)
        self.input_image_size = input_image_size

        # check if the input image size is divisible by the patch size
        assert (
            input_image_size[1] % patch_size[0] == 0
        ), f"{input_image_size=} {patch_size=} {patch_stride=}"
        assert (
            input_image_size[2] % patch_size[1] == 0
        ), f"{input_image_size=} {patch_size=} {patch_stride=}"

        self.class_embedding = nn.Parameter(self.scale * torch.randn(patch_embed_dim))
        self.positional_grid_size = (
            input_image_size[1] // patch_size[0],
            input_image_size[2] // patch_size[1],
        )
        self.positional_embedding = nn.Parameter(
            self.scale
            * torch.randn(
                self.positional_grid_size[0] * self.positional_grid_size[1] + 1,
                patch_embed_dim,
            )
        )

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=patch_embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            bias=False,
        )

        self.patchdropout_pre = (
            PatchDropout(p=patch_dropout_prob)
            if patch_dropout_prob > 0
            else nn.Identity()
        )
        self.layernorm_pre = CastLayerNorm(normalized_shape=patch_embed_dim)
        self.layernorm_post = CastLayerNorm(normalized_shape=patch_embed_dim)

        self.head_weight = nn.Parameter(
            self.scale * torch.randn(patch_embed_dim, embed_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape
        # [batch, channels, height, width] -> [batch, self.patch_embed_dim, *self.positional_grid_size]
        x = self.conv(x)

        # num_patches := self.positional_grid_size[0] * self.positional_grid_size[1]
        # [batch, self.patch_embed_dim, *self.positional_grid_size] -> [batch, num_patches, self.patch_embed_dim]
        x = x.reshape(batch_size, self.patch_embed_dim, -1).permute(0, 2, 1)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = torch.cat(
            [self.class_embedding.view(1, 1, -1).expand(batch_size, -1, -1), x], dim=1
        )
        x = x + self.positional_embedding

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = self.patchdropout_pre(x)
        x = self.layernorm_pre(x)
        x, w = self.transformer(
            x, attn_mask=attn_mask, return_all_weights=return_all_weights
        )
        x = self.layernorm_post(x)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, self.patch_embed_dim], [batch, num_patches, self.patch_embed_dim]
        # _tokens: unused
        pooled, _tokens = x[:, 0], x[:, 1:]

        # [batch, self.patch_embed_dim] -> [batch, self.embed_dim]
        pooled = pooled @ self.head_weight
        return pooled, w


class SyntacticTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_attn_heads: int = 8,
        num_gate_heads: int = 8,
        num_induction_layers: int = 2,
        *,
        num_layers: int = 12,
        attn_dropout_p: float = 0.0,
        gate_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_attn_heads
        self.num_induction_layers = num_induction_layers
        self.induction_head = SyntacticDistanceGate1D(
            embed_dim=embed_dim,
            num_lookback_range=3,
            num_gate_heads=num_gate_heads,
            tau=10.0,
            dropout_p=gate_dropout_p,
        )
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualSyntacticAttentionEncoderLayer(
                    embed_dim=embed_dim,
                    num_attn_heads=num_attn_heads,
                    num_gate_heads=num_gate_heads,
                    attn_dropout_p=attn_dropout_p,
                    gate_dropout_p=gate_dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = False,
        return_distance: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        ret_weights = []

        for induction_layer in self.res_attn_blocks[: self.num_induction_layers]:
            x, attn_weight = induction_layer(x, attn_mask=attn_mask)
            if return_all_weights:
                ret_weights.append(attn_weight)

        attn_gate, distance = self.induction_head(x)

        for i, res_attn_block in enumerate(
            self.res_attn_blocks[self.num_induction_layers :]
        ):
            x, attn_weight = res_attn_block(
                x, attn_mask=attn_mask, attn_gate=attn_gate if i == 0 else None
            )
            if return_all_weights:
                ret_weights.append(attn_weight)

        if return_all_weights and return_distance:
            return x, torch.stack(ret_weights, dim=1), distance
        if return_all_weights:
            return x, torch.stack(ret_weights, dim=1)
        if return_distance:
            return x, attn_weight.unsqueeze(0), distance
        else:
            return x, attn_weight.unsqueeze(0)


class SyntacticTextTransformerEncoder(TextTransformerEncoder):
    def __init__(
        self,
        embed_dim: int = 512,
        num_attn_heads: int = 8,
        num_gate_heads: int = 8,
        attn_dropout_p: float = 0.0,
        gate_dropout_p: float = 0.0,
        *,
        num_layers: int = 12,
        vocab_size: int = 49408,
        vocab_embed_dim: int = 512,
        max_context_length: int = 77,
        pad_token_id: int = 0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_attn_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            vocab_embed_dim=vocab_embed_dim,
            max_context_length=max_context_length,
            pad_token_id=pad_token_id,
        )
        self.transformer = SyntacticTransformerEncoder(
            embed_dim=embed_dim,
            num_attn_heads=num_attn_heads,
            num_gate_heads=num_gate_heads,
            num_layers=num_layers,
            attn_dropout_p=attn_dropout_p,
            gate_dropout_p=gate_dropout_p,
        )

    def get_distance(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, sequence_length]
        """
        batch_size, sequence_length = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding[:sequence_length]
        x, w, d = self.transformer(
            x,
            attn_mask=attn_mask or self.attn_mask,
            return_all_weights=return_all_weights,
            return_distance=True,
        )
        return x, w, d


class SyntacticVisionTransformerEncoder(VisionTransformerEncoder):
    def __init__(
        self,
        embed_dim: int = 512,
        num_attn_heads: int = 12,
        num_gate_heads: int = 12,
        num_layers: int = 12,
        attn_dropout_p: float = 0.0,
        gate_dropout_p: float = 0.0,
        *,
        input_image_size: int | tuple[int, int] | tuple[int, int, int] = 224,
        patch_embed_dim: int = 768,
        patch_size: tuple[int, int] = (32, 32),
        patch_stride: Optional[tuple[int, int]] = None,
        patch_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_attn_heads,
            num_layers=num_layers,
            input_image_size=input_image_size,
            patch_embed_dim=patch_embed_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
            patch_dropout_prob=patch_dropout_prob,
        )
        self.transformer = SyntacticTransformerEncoder(
            embed_dim=patch_embed_dim,
            num_attn_heads=num_attn_heads,
            num_gate_heads=num_gate_heads,
            num_layers=num_layers,
            attn_dropout_p=attn_dropout_p,
            gate_dropout_p=gate_dropout_p,
        )

    def get_distance(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape
        # [batch, channels, height, width] -> [batch, self.patch_embed_dim, *self.positional_grid_size]
        x = self.conv(x)

        # num_patches := self.positional_grid_size[0] * self.positional_grid_size[1]
        # [batch, self.patch_embed_dim, *self.positional_grid_size] -> [batch, num_patches, self.patch_embed_dim]
        x = x.reshape(batch_size, self.patch_embed_dim, -1).permute(0, 2, 1)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = torch.cat(
            [self.class_embedding.view(1, 1, -1).expand(batch_size, -1, -1), x], dim=1
        )
        x = x + self.positional_embedding

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = self.patchdropout_pre(x)
        x = self.layernorm_pre(x)
        x, w, d = self.transformer(
            x,
            attn_mask=attn_mask,
            return_all_weights=return_all_weights,
            return_distance=True,
        )
        return x, w, d


class CLIP(nn.Module):
    def __init__(
        self,
        visual_backbone: nn.Module,
        textual_backbone: nn.Module,
    ):
        super().__init__()
        self.visual = visual_backbone
        self.textual = textual_backbone
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )
        self.logit_bias = nn.Parameter(torch.zeros([]))

    @property
    def dtype(self):
        if hasattr(self.visual, "dtype"):
            return self.visual.dtype
        elif hasattr(self.textual, "dtype"):
            return self.textual.dtype
        else:
            return self.logit_scale.dtype

    def encode_image(self, images: torch.Tensor, normalize: bool = True):
        feats, *_ = self.visual(images)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True):
        feats, *_ = self.textual(tokens)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def get_features(
        self, images: torch.Tensor, tokens: torch.Tensor, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images, normalize=normalize)
        text_features = self.encode_text(tokens, normalize=normalize)
        return image_features, text_features

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        *,
        normalize: bool = True,
        softmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features, text_features = self.get_features(
            images, tokens, normalize=normalize
        )
        logits_per_image = (
            self.logit_scale.exp() * image_features @ text_features.t()
            + self.logit_bias
        )
        if softmax:
            logits_per_image = logits_per_image.softmax(dim=1)
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# 損失定義
class ContrastiveLoss(nn.Module):
    # from https://github.com/zer0int/CLIP-fine-tune
    def __init__(self, temperature: float = 0.07, smoothing: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(
        self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # Apply label smoothing
        N = logits.size(0)
        smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
        with torch.autocast(device_type=logits.device.type, enabled=False):
            smoothed_labels = smoothed_labels.scatter(
                1,  # dim
                labels.unsqueeze(1),  # index
                1 - self.smoothing,  # src
            )

        # Calculate loss manually using log-softmax and smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs).sum(dim=1).mean()

        return (loss_img + loss_txt) / 2


def get_model(args: TypedArgs) -> nn.Module:
    match args.model_arch:
        case "CLIP":
            model = CLIP(
                visual_backbone=VisionTransformerEncoder(
                    dropout_p=args.attn_dropout_p,
                ),
                textual_backbone=TextTransformerEncoder(
                    dropout_p=args.attn_dropout_p,
                ),
            )
        case "SyntaCLIP":
            model = CLIP(
                visual_backbone=SyntacticVisionTransformerEncoder(
                    attn_dropout_p=args.attn_dropout_p,
                    gate_dropout_p=args.gate_dropout_p,
                ),
                textual_backbone=SyntacticTextTransformerEncoder(
                    attn_dropout_p=args.attn_dropout_p,
                    gate_dropout_p=args.gate_dropout_p,
                ),
            )
        case _:
            raise ValueError(f"Unknown model name {args.model_arch}")
    return model


def load_pretrained_model(model: nn.Module, pretrained_path: str) -> nn.Module:
    if pretrained_path:
        logger.info(f"Loading pretrained model from {pretrained_path}")
        state = torch.load(pretrained_path, map_location=DEVICE)
        state_dict = dict()
        if state.get("model", None):
            for k in list(state["model"].keys()):
                if k.startswith("module."):
                    kk = k.replace("module.", "")
                    state_dict[kk] = state["model"][k]
                else:
                    state_dict[k] = state["model"][k]
        else:
            state_dict = state

        model.load_state_dict(state_dict)

    return model


def parallelize_model(model: nn.Module) -> nn.Module:
    if IS_DISTRIBUTED and IS_CUDA_AVAILABLE:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            model = model.to(LOCAL_RANK)
            model = DistributedDataParallel(
                model,
                device_ids=[LOCAL_RANK],
                output_device=LOCAL_RANK,
                find_unused_parameters=True,
            )
            torch.cuda.current_stream().wait_stream(stream)

        logger.info(f"Model parallelized to device {LOCAL_RANK}")

    return model


def build_model(args: TypedArgs) -> nn.Module:
    model = get_model(args)
    pretrained_path = args.pretrained
    if pretrained_path:
        model = load_pretrained_model(model, pretrained_path)
    model = parallelize_model(model)

    return model


def build_dataloader(args: TypedArgs) -> tuple[Callable, DALIGenericIterator]:
    dataset: DaliDataset
    match args.dataset_name:
        case "CC3M":
            dataset = CC3MDataset(args.dataset_path)
        case "CC12M":
            dataset = CC12MDataset(args.dataset_path)
        case _:
            raise ValueError(f"Unknown dataset name {args.dataset_name}")

    def extract_fn(batch, *args):
        return dataset.batch_extract(batch)

    return extract_fn, dataset.build_dataloader(
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=LOCAL_RANK,
        num_shards=WORLD_SIZE,
        shard_id=LOCAL_RANK,
        seed=args.seed,
        shuffle=args.shuffle,
    )


def train(
    *,
    model: nn.Module,
    transform: Callable,
    tokenizer: SimpleTokenizer,
    dataloader: DALIGenericIterator,
    extract_fn: Callable,
    loss_fn: nn.Module,
    num_epochs: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    break_num_epoch: Optional[int] = None,
    break_batch_index: Optional[int] = None,
    compile_mode: Optional[
        Literal["default", "reduce-overhead", "max-autotune"]
    ] = None,
    args: Optional[TypedArgs] = None,
):
    model.train()
    if compile_mode is not None:
        logger.info(f"Compiling model in {compile_mode} mode")
        compiled_model = torch.compile(model, mode=compile_mode)
        logger.info("Compiled !")

    if scaler is None:
        scaler = torch.cuda.amp.GradScaler()
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    start_timestamp = datetime.now().isoformat()
    save_dir = args.save_dir / args.model_arch / args.dataset_name / start_timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    loader_length = len(dataloader)
    best_loss_epoch = 0
    epoch_losses = []
    epoch_end_timestamps = []

    with logging_redirect_tqdm(loggers=[logger]), tqdm(
        range(num_epochs), desc="Epochs"
    ) as pbar:
        for epoch in pbar:
            model.train()
            logger.info(
                f"Start training for {epoch + 1}/{num_epochs} epochs, {scheduler.get_last_lr()[0]=}"
            )
            epoch_loss = 0.0
            with tqdm(
                dataloader,
                desc="Batches",
                leave=False,
                total=min(break_batch_index, loader_length)
                if break_batch_index is not None
                else loader_length,
            ) as epoch_pbar:
                for batch_index, batch in enumerate(epoch_pbar):
                    images, metadata = extract_fn(*batch)
                    captions = [datum["caption"] for datum in metadata]
                    tokens = tokenizer(captions)

                    images, tokens = images.to(LOCAL_RANK), tokens.to(LOCAL_RANK)
                    # images.shape: [batch, 3, 224, 224]
                    # tokens.shape: [batch, 77]

                    optimizer.zero_grad()
                    with torch.autocast(
                        device_type=args.amp_device_type, dtype=args.amp_dtype
                    ):
                        if compile_mode is not None:
                            logits_per_image, logits_per_text = compiled_model(
                                images, tokens
                            )
                        else:
                            logits_per_image, logits_per_text = model(images, tokens)
                        loss = loss_fn(logits_per_image, logits_per_text)

                    if loss.isnan().any():
                        logger.warning(f"Loss is NaN at {epoch=}, {batch_index=}")
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    epoch_pbar.set_postfix(
                        loss=f"{loss.item():.8f}",
                        lr=f"{scheduler.get_last_lr()[0]:.8f}",
                    )
                    epoch_pbar.set_description(
                        f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_index + 1}/{loader_length} | Loss: {loss.item():.8f}"
                    )

                    wandb.log(
                        {
                            "batch_loss": loss.item(),
                        }
                    )

                    if (
                        break_batch_index is not None
                        and batch_index >= break_batch_index
                    ):
                        break

            epoch_loss /= batch_index + 1
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.8f}"
            )
            logger.info(
                f" + End Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.8f}"
            )
            if break_num_epoch is not None and epoch >= break_num_epoch:
                break

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

            epoch_losses.append(epoch_loss)
            epoch_end_timestamps.append(datetime.now().isoformat())

            logits_per_image_inference = inference(
                model,
                transform,
                tokenizer,
                args,
                save_dirname=start_timestamp,
                save_filename=f"{epoch}.png",
            )

            is_best_epoch = epoch_loss == min(epoch_losses)
            if is_best_epoch:
                best_loss_epoch = epoch

            wandb.log(
                {
                    "epoch_loss": epoch_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            if epoch % args.save_interval == 0 or is_best_epoch:
                info = {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": epoch_end_timestamps[-1],
                    "epoch_timestamps": epoch_end_timestamps,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr(),
                    "best_loss_epoch": best_loss_epoch,
                    "epoch_loss": epoch_loss,
                    "epoch_losses": epoch_losses,
                    "logits_per_image_inference": logits_per_image_inference.tolist(),
                }

                with open(save_dir / "info.json", "w") as f:
                    for k, v in sorted(args.as_dict().items(), key=lambda x: x[0]):
                        info[k] = str(v)
                    json.dump(info, f, indent=4)

                info["model"] = model.state_dict()
                info["optimizer"] = optimizer.state_dict()
                info["scaler"] = scaler.state_dict()
                info["scheduler"] = scheduler.state_dict()

                torch.save(
                    info,
                    save_dir / f"epoch_{epoch + 1}.pth",
                )
                if is_best_epoch:
                    torch.save(
                        info,
                        save_dir / "best.pth",
                    )


def inference(
    model: nn.Module,
    transform: Callable,
    tokenizer: SimpleTokenizer,
    args: TypedArgs,
    sentences: list[str] = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a human",
        "a photo of a car",
        "a photo of a plane",
        "a photo of a train",
        "a photo of a boat",
        "a photo of a bicycle",
        "a photo of a mountain",
        "a photo of a river",
        "a photo of a lake",
        "a photo of a sea",
        "a photo of a forest",
        "a photo of a desert",
        "a photo of a beach",
        "a photo of a city",
        "a photo of a town",
        "a photo of a village",
        "a photo of a house",
        "a photo of a building",
        "a photo of a bridge",
        "a photo of a tunnel",
        "a photo of a road",
        "a photo of a sky",
    ],
    image_dir: Path = const.DEFAULT_IMAGE_DIR,
    save_dirname: str = "default",
    save_filename: Optional[str] = None,
) -> torch.Tensor:
    model.eval()
    images = torch.stack(
        [transform(Image.open(path).convert("RGB")) for path in image_dir.glob("*.jpg")]
    )

    tokens = tokenizer(sentences)
    images, tokens = images.to(LOCAL_RANK), tokens.to(LOCAL_RANK)
    images, tokens = images.to(DEVICE), tokens.to(DEVICE)

    with torch.inference_mode():
        logits_per_image, logits_per_text = model(images, tokens)

    if save_filename:
        fig, axes = plt.subplots(len(images), 2, figsize=(12, 6 * len(images)), dpi=200)
        for image_index in range(len(images)):
            bar = logits_per_image[image_index].cpu().numpy()
            axes[image_index, 0].imshow(
                images[image_index].permute(1, 2, 0).cpu().numpy()
            )
            axes[image_index, 0].set_title(f"Image {image_index}")
            axes[image_index, 0].axis("off")
            axes[image_index, 1].bar(range(len(sentences)), bar, color="gray")
            axes[image_index, 1].bar(bar.argmax(), bar[bar.argmax()], color="blue")
            axes[image_index, 1].set_title("Logits per image")
            axes[image_index, 1].set_xticks(range(len(sentences)))
            axes[image_index, 1].set_xticklabels(sentences, rotation=90)
            axes[image_index, 1].set_ylim(0, 1)
        fig.tight_layout()
        (args.save_dir / "inferences" / save_dirname / save_filename).parent.mkdir(
            parents=True, exist_ok=True
        )
        wandb.log({"inference": fig})

        fig.savefig(args.save_dir / "inferences" / save_dirname / save_filename)

        plt.close(fig)

    return logits_per_image


def main(args: TypedArgs):
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("==== Arguments: ====")
    for k, v in args.as_dict().items():
        logger.info(f"| {k}: {v}")
    logger.info("====================")

    if args.use_wandb and LOCAL_RANK == 0:
        wandb.init(project=args.project_name or args.model_arch, save_code=True)
    else:
        wandb.init(project=args.project_name or args.model_arch, mode="disabled")

    wandb.config.update(args.as_dict())

    tokenizer = SimpleTokenizer(
        context_length=const.DEFAULT_CONTEXT_LENGTH,
        bpe_path=const.DEFAULT_BPE_PATH,
    )
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # inference(model, transform, tokenizer, args)
    model = build_model(args)
    model.to(DEVICE)

    try:
        match args.mode:
            case "train":
                extract_fn, dataloader = build_dataloader(args)
                scaler = torch.amp.GradScaler()
                criterion = ContrastiveLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.8, patience=3
                )
                train(
                    model=model,
                    transform=transform,
                    tokenizer=tokenizer,
                    dataloader=dataloader,
                    extract_fn=extract_fn,
                    loss_fn=criterion,
                    num_epochs=args.num_epochs,
                    scaler=scaler,
                    scheduler=scheduler,
                    optimizer=optimizer,
                    break_num_epoch=args.break_num_epoch,
                    break_batch_index=args.break_batch_index,
                    compile_mode=args.compile_mode,
                    args=args,
                )

            case "inference":
                inference(
                    model, transform, tokenizer, args, save_filename=args.save_filename
                )

            case "eval":
                pass

            case _:
                raise ValueError(f"Unknown mode {args.mode}")
    except Exception as e:
        logger.error(e)
    finally:
        if IS_DISTRIBUTED:
            torch.distributed.destroy_process_group()
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    if is_jupyter(globals()):
        sys.argv = [
            __file__,
            "train",
            "--model_arch",
            "SyntaCLIP",
        ]
        print(sys.argv)
    parser = TypedArgs()
    args = parser.parse_args()
    main(args)
# %%
