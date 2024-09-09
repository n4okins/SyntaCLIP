# %%
import argparse
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import open_clip
import requests
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchinfo
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from tqdm.auto import tqdm
from utils.clogging import getColoredLogger
from utils.dummy import DummyObject
from utils.initialize import initializer
from utils.jupyter import is_jupyter

import syntaclip.nn as synn
from syntaclip.utils.converter import convert_model_params_laion2b_s34b_b79k_to_CLIP512
from syntaclip.utils.datasets import CC3MDataset, CC12MDataset

# Logger Settings
logger = getColoredLogger(__name__)
PROJECT_ROOT = initializer(globals(), logger=logger)


@dataclass
class TypedArgs:
    arch_name: Literal["CLIP512", "SyntaCLIP512"] = "CLIP512"
    dataset_name: Literal["CC3M", "CC12M"] = "CC3M"
    use_wandb_log: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    training_layer_indexes: list[int] = field(default_factory=lambda: [0, 1, 10, 11])
    attn_dropout_p: float = 0.25
    gate_dropout_p: float = 0.25
    learning_rate: float = 1e-4
    scheduler: Literal["none", "cosine", "plateau"] = "plateau"
    epochs: int = 30
    seed: int = 42
    batch_size: int = 512
    use_pretrained: bool = True
    insert_induction_layer_num_visual: list[int] = field(default_factory=lambda: [2])
    insert_induction_layer_num_textual: list[int] = field(default_factory=lambda: [2])

    @staticmethod
    def parse() -> "TypedArgs":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--arch_name",
            type=str,
            default="CLIP512",
            choices=["CLIP512", "SyntaCLIP512"],
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="CC3M",
            choices=["CC3M", "CC12M"],
        )
        parser.add_argument(
            "--use_wandb_log",
            action="store_true",
        )
        parser.add_argument(
            "--log_level",
            type=str,
            default="DEBUG",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        parser.add_argument(
            "--training_layer_indexes",
            type=lambda x: list(map(int, x.split(","))),
            default=[0, 1, 2, 9, 10, 11],
        )
        parser.add_argument(
            "--attn_dropout_p",
            type=float,
            default=0.25,
        )
        parser.add_argument(
            "--gate_dropout_p",
            type=float,
            default=0.25,
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="plateau",
            choices=["none", "cosine", "plateau"],
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=30,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=512,
        )
        parser.add_argument(
            "--use_pretrained",
            action="store_true",
        )
        parser.add_argument(
            "--insert_induction_layer_num_visual",
            type=lambda x: list(map(int, x.split(","))),
            default=[2],
        )
        parser.add_argument(
            "--insert_induction_layer_num_textual",
            type=lambda x: list(map(int, x.split(","))),
            default=[2],
        )

        return TypedArgs(**vars(parser.parse_args()))

    def asdict(self):
        return asdict(self)


# parse args
args = TypedArgs.parse() if not is_jupyter(globals()) else TypedArgs()
logger.setLevel(args.log_level)
logger.info(f"{args=}")

training_data_dir = Path.home() / "datasets" / "WebDataset" / args.dataset_name

match args.dataset_name:
    case "CC12M":
        dataset = CC12MDataset(training_data_dir)
    case "CC3M":
        dataset = CC3MDataset(training_data_dir, split="train")
    case _:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

PROJECT_NAME = f"{args.arch_name}-{args.dataset_name}"
USE_WANDB_LOG = args.use_wandb_log

# Torch distributed settings
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_CUDA_AVAILABLE = torch.cuda.is_available()

if IS_DISTRIBUTED:
    assert IS_CUDA_AVAILABLE, "Distributed training requires CUDA available"
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    WORLD_SIZE = torch.cuda.device_count()
    torch.cuda.set_device(LOCAL_RANK)
    logger.info(f"LOCAL_RANK={LOCAL_RANK}, WORLD_SIZE={WORLD_SIZE}")

FIRST_RANK = LOCAL_RANK == 0
if USE_WANDB_LOG and FIRST_RANK:
    import wandb

    wandb.init(project=PROJECT_NAME, save_code=True)
else:
    wandb = DummyObject()

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PROJECT_INFOMATION_DICT = dict(
    TIMESTAMP=TIMESTAMP,
    PROJECT_ROOT=PROJECT_ROOT,
    PROJECT_NAME=PROJECT_NAME,
    WORLD_SIZE=WORLD_SIZE,
    LOCAL_RANK=LOCAL_RANK,
    IS_DISTRIBUTED=IS_DISTRIBUTED,
    USE_WANDB_LOG=USE_WANDB_LOG,
    TORCH_VERSION=torch.__version__,
    IS_CUDA_AVAILABLE=IS_CUDA_AVAILABLE,
    TORCH_CUDA_VERSION=torch.version.cuda,
    TORCH_CUDNN_VERSION=torch.backends.cudnn.version(),
    TORCH_DEVICE_COUNT=torch.cuda.device_count(),
    TORCH_DEVICES_INFO=[
        torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())
    ],
)


def once_logger_info(message: str):
    if FIRST_RANK:
        logger.info(message)


# Print Project Information
once_logger_info("=" * 16 + " Project Information Begin " + "=" * 16)
for k, v in PROJECT_INFOMATION_DICT.items():
    tab = 3 - len(k) // 6
    if tab == 0:
        tab += int(len(k) % 6 == 0)
    tab += 1
    once_logger_info(f" | {k}" + "\t" * tab + f"{v}")
wandb.config.update({"project_information": PROJECT_INFOMATION_DICT})
once_logger_info("=" * 16 + " Project Information End " + "=" * 16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_device_type = device.type
amp_dtype = torch.float32 if device.type == "cuda" else torch.float16
tokenizer = open_clip.get_tokenizer("ViT-B-32")

change_visual_layers = change_textual_layers = args.training_layer_indexes

model_path = (
    PROJECT_ROOT
    / "models"
    / args.arch_name
    / PROJECT_NAME
    / TIMESTAMP
    / "before_training.pth"
)
model_path.parent.mkdir(parents=True, exist_ok=True)

match args.arch_name:
    case "CLIP512":
        model = synn.CLIP(
            visual_backbone=synn.VisionTransformerEncoder(
                dropout_p=args.attn_dropout_p,
            ),
            textual_backbone=synn.TextTransformerEncoder(
                dropout_p=args.attn_dropout_p,
            ),
        )

    case "SyntaCLIP512":
        model = synn.CLIP(
            visual_backbone=synn.SyntacticVisionTransformerEncoder(
                attn_dropout_p=args.attn_dropout_p,
                gate_dropout_p=args.gate_dropout_p,
                insert_induction_layer_num=args.insert_induction_layer_num_visual,
            ),
            textual_backbone=synn.SyntacticTextTransformerEncoder(
                attn_dropout_p=args.attn_dropout_p,
                gate_dropout_p=args.gate_dropout_p,
                insert_induction_layer_num=args.insert_induction_layer_num_textual,
            ),
        )

openclip_model, _, transform = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None),
)

wandb.config.update(
    {
        "model_path": str(model_path),
        "args": args.asdict(),
    }
)

if model_path.exists():
    state = torch.load(model_path, weights_only=True)
    once_logger_info(f"Loaded model from {model_path}")
    model.load_state_dict(state["model"])

    for name, param in model.named_parameters():
        if "transformer.res_attn_blocks" in name:
            param.requires_grad_(False)

    for visual_layer_index in change_visual_layers:
        model.visual.transformer.res_attn_blocks[visual_layer_index].requires_grad_(
            True
        )
    for textual_layer_index in change_textual_layers:
        model.textual.transformer.res_attn_blocks[textual_layer_index].requires_grad_(
            True
        )

else:
    # load pretrained model
    state = convert_model_params_laion2b_s34b_b79k_to_CLIP512(model, openclip_model)
    once_logger_info("Loaded pretrained model from open_clip")
    model.load_state_dict(state)

    # freeze the transformer layers
    for name, param in model.named_parameters():
        if "transformer.res_attn_blocks" in name:
            param.requires_grad_(False)

    for visual_layer_index in change_visual_layers:
        model.visual.transformer.res_attn_blocks[visual_layer_index].requires_grad_(
            True
        )
    for textual_layer_index in change_textual_layers:
        model.textual.transformer.res_attn_blocks[textual_layer_index].requires_grad_(
            True
        )

    once_logger_info(f"Model is loaded from open_clip and saved to {model_path}")
    torch.save(
        {
            "model": model.state_dict(),
            "train_layers": {
                "visual": change_visual_layers,
                "textual": change_textual_layers,
            },
        },
        model_path,
    )


# check "Trainable params" and "Non-trainable params"
if not IS_DISTRIBUTED or FIRST_RANK:
    print(
        torchinfo.summary(
            model,
            input_size=[(1, 3, 224, 224), (1, 77)],
            dtypes=[torch.float32, torch.long],
            device="cpu",
            depth=5,
        )
    )


def inference(model, epoch):
    fig_dir = (
        PROJECT_ROOT / "ignores" / "figs" / args.arch_name / PROJECT_NAME / f"{epoch}"
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    urls = (
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000294350.jpg",
    )
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    images = torch.stack(
        list(
            map(
                transform,
                [
                    Image.open(requests.get(url, stream=True).raw).convert("RGB")
                    for url in urls
                ],
            )
        )
    )
    sentences = [
        "a photo of a cat",
        "a photo of two cats",
        "a photo of three cats",
        "a photo of a dog",
        "a photo of two dogs",
        "a photo of three dogs",
        "a photo of a bird",
        "a photo of a person",
    ]
    logger.info(f"{sentences=}")
    tokens = tokenizer(sentences)

    images = images.to(device)
    tokens = tokens.to(device)
    model = model.to(device)
    if IS_DISTRIBUTED:
        images, tokens = images.to(LOCAL_RANK), tokens.to(LOCAL_RANK)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=amp_device_type, dtype=amp_dtype
        ):
            images_feats, images_weights = model.module.visual(
                images, return_all_weights=True
            )
            tokens_feats, tokens_weights = model.module.textual(
                tokens, return_all_weights=True
            )
            logits_per_image, logits_per_text = model(images, tokens)
    else:
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float32 if device.type == "cuda" else torch.float16,
        ):
            images_feats, images_weights = model.visual(images, return_all_weights=True)
            tokens_feats, tokens_weights = model.textual(
                tokens, return_all_weights=True
            )
            logits_per_image, logits_per_text = model(images, tokens)

    fig, axes = plt.subplots(4, 3, figsize=(24, 32))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(tokens_weights[0, i].detach().cpu().numpy())
        ax.axis("off")
        ax.set_title(f"token_attn_weight_{i}")
    fig.tight_layout()
    fig.savefig(fig_dir / "token_attn_weights.png")
    fig.clear()
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(24, 32))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images_weights[0, i].detach().cpu().numpy())
        ax.axis("off")
        ax.set_title(f"image_attn_weight_{i}")
    fig.tight_layout()
    fig.savefig(fig_dir / "image_attn_weights.png")
    fig.clear()
    plt.close(fig)
    del fig, axes
    return logits_per_image, logits_per_text


if model is not None and IS_DISTRIBUTED and IS_CUDA_AVAILABLE:
    TORCH_STRAEM = torch.cuda.Stream()
    TORCH_STRAEM.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(TORCH_STRAEM):
        model = model.to(LOCAL_RANK)
        model = DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
        )
        torch.cuda.current_stream().wait_stream(TORCH_STRAEM)

    logger.info(f"[{LOCAL_RANK=}] Model is DistributedDataParallel")

if FIRST_RANK:
    dataset.download(enable_wandb=USE_WANDB_LOG)

dataloader = dataset.build_dataloader(
    batch_size=args.batch_size,
    num_threads=16,
    device_id=LOCAL_RANK if IS_CUDA_AVAILABLE else None,
    num_shards=WORLD_SIZE,
    shard_id=LOCAL_RANK,
    seed=args.seed + LOCAL_RANK,
    shuffle=True,
)

# Training
scaler = torch.amp.GradScaler()
criterion = synn.ContrastiveLoss()

criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
match args.scheduler:
    case "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=5e-8
        )
    case "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-8
        )
    case "none":
        scheduler = DummyObject()
    case _:
        raise ValueError(f"Unknown scheduler name: {args.scheduler}")
model.to(device)

total_pbar = tqdm(range(args.epochs))
losses = []


logger.info("Start Training")
logger.info(f"{args.epochs=}")
logger.info(f"{args.batch_size=}")
logger.info(f"{args.learning_rate=}")
logger.info(f"{args.training_layer_indexes=}")
logger.info(f"{scheduler=}")

per100 = len(dataloader) // 100

for epoch in total_pbar:
    model.train()
    epoch_pbar = tqdm(dataloader, leave=False)
    epoch_loss = 0

    if FIRST_RANK:
        inference_logits_per_image, _ = inference(model, epoch)
        logger.info(f"{inference_logits_per_image=}")
        wandb.log(
            {
                "inference_logits_per_image": inference_logits_per_image.detach()
                .cpu()
                .numpy(),
            }
        )

    for i, (batch, *_) in enumerate(epoch_pbar):
        images, metadata = dataset.batch_extract(batch)
        captions = [datum["caption"] for datum in metadata]
        tokens = tokenizer(captions)

        if IS_DISTRIBUTED:
            images, tokens = images.to(LOCAL_RANK), tokens.to(LOCAL_RANK)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype):
            logits_per_image, logits_per_text = model(images, tokens)
            loss_img, loss_txt = criterion(logits_per_image, logits_per_text)
            loss = (loss_img + loss_txt) / 2

        if loss.isnan().any():
            logger.warning(f"Loss is NaN at epoch {args.epochs+1}, iteration {i+1}")
            del loss, loss_img, loss_txt
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        epoch_pbar.set_description(
            f"Epoch {epoch+1}/{args.epochs} | Loss {loss.item():.8f}"
        )
        epoch_pbar.set_postfix(
            {
                "loss_img": f"{loss_img.item():.8f}",
                "loss_txt": f"{loss_txt.item():.8f}",
            }
        )

        if i % per100 == 0:
            wandb.log(
                {
                    "step_loss": loss.item(),
                    "step_loss_img": loss_img.item(),
                    "step_loss_txt": loss_txt.item(),
                }
            )

    epoch_loss /= len(dataloader)
    wandb.log(
        {
            "epoch_loss": epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )
    losses.append(epoch_loss)
    torch.save(
        {
            "model": model.state_dict()
            if not IS_DISTRIBUTED
            else model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss": losses,
            "train_layers": {
                "visual": change_visual_layers,
                "textual": change_textual_layers,
            },
        },
        model_path.with_name(f"checkpoint_{epoch:02d}.pth"),
    )
    if args.scheduler != "none":
        if args.scheduler == "plateau":
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

wandb.finish()
# %%
