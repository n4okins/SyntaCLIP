# %%
import os
from datetime import datetime
from pathlib import Path

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

import syntaclip.nn as synn
from syntaclip.utils.converter import convert_model_params_laion2b_s34b_b79k_to_CLIP512
from syntaclip.utils.datasets.cc3m import CC3MDataset

# Logger Settings
logger = getColoredLogger(__name__)
# Project Init
PROJECT_ROOT = initializer(globals(), logger=logger)
logger.setLevel("DEBUG")

PROJECT_NAME = "SyntaCLIP512-Finetune"
USE_WANDB_LOG = True

# Torch distributed settings
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_CUDA_AVAILABLE = torch.cuda.is_available()

if IS_DISTRIBUTED:
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

PROJECT_INFOMATION_DICT = dict(
    TIMESTAMP=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

tokenizer = open_clip.get_tokenizer("ViT-B-32")

change_visual_layers = change_textual_layers = [0, 1, 10, 11]

model_path = (
    PROJECT_ROOT
    / "models"
    / "SyntaCLIP512"
    / f"Origin_V{'-'.join(map(str, change_visual_layers))}_T{'-'.join(map(str, change_textual_layers))}"
    / "first.pth"
)
model_path.parent.mkdir(parents=True, exist_ok=True)
model = synn.CLIP(
    visual_backbone=synn.SyntacticVisionTransformerEncoder(),
    textual_backbone=synn.SyntacticTextTransformerEncoder(),
)
openclip_model, _, transform = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None),
)

wandb.config.update(
    {
        "change_visual_layers": change_visual_layers,
        "change_textual_layers": change_textual_layers,
        "model_path": str(model_path),
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
            depth=4,
        )
    )


# %%
def inference(model, epoch):
    fig_dir = PROJECT_ROOT / "ignores" / "figs" / "SyntaCLIP" / f"{epoch}"
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
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a person",
    ]
    tokens = tokenizer(sentences)

    images = images.to(LOCAL_RANK).to(device)
    tokens = tokens.to(LOCAL_RANK).to(device)
    model = model.to(device)
    if IS_DISTRIBUTED:
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda", dtype=torch.float32
        ):
            images_feats, images_weights, images_distances = (
                model.module.visual.get_distance(images)
            )
            tokens_feats, tokens_weights, tokens_distances = (
                model.module.textual.get_distance(tokens)
            )
            logits_per_image, logits_per_text = model(images, tokens)
    else:
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda", dtype=torch.float32
        ):
            images_feats, images_weights, images_distances = model.visual.get_distance(
                images
            )
            tokens_feats, tokens_weights, tokens_distances = model.textual.get_distance(
                tokens
            )
            logits_per_image, logits_per_text = model(images, tokens)

    fig, axes = plt.subplots(4, 3, figsize=(24, 32))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(tokens_weights[0, i].detach().cpu().numpy())
        ax.axis("off")
        ax.set_title(f"token_attn_weight_{i}")

    fig.suptitle(f"Epoch {epoch} Token Attention Weights")
    fig.tight_layout()
    fig.savefig(fig_dir / "token_attn_weights.png")
    fig.clear()
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(24, 32))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images_weights[0, i].detach().cpu().numpy())
        ax.axis("off")
        ax.set_title(f"image_attn_weight_{i}")

    fig.suptitle(f"Epoch {epoch} Image Attention Weights")
    fig.tight_layout()
    fig.savefig(fig_dir / "image_attn_weights.png")
    fig.clear()
    plt.close(fig)
    del fig, axes
    return logits_per_image, logits_per_text


# %%
if model is not None and IS_DISTRIBUTED and IS_CUDA_AVAILABLE:
    TORCH_STRAEM = torch.cuda.Stream()
    TORCH_STRAEM.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(TORCH_STRAEM):
        model = model.to(LOCAL_RANK)
        model = DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )
        torch.cuda.current_stream().wait_stream(TORCH_STRAEM)

    logger.info(f"[{LOCAL_RANK=}] Model is DistributedDataParallel")


CC3M_DATA_DIR = Path.home() / "datasets" / "WebDataset" / "CC3M"
dataset = CC3MDataset(CC3M_DATA_DIR, split="train")
if FIRST_RANK:
    dataset.download(enable_wandb=USE_WANDB_LOG)

dataloader = dataset.build_dataloader(
    batch_size=512,
    num_threads=8,
    device_id=LOCAL_RANK,
    num_shards=WORLD_SIZE,
    shard_id=LOCAL_RANK,
    seed=42 + LOCAL_RANK,
    shuffle=True,
)

# Training
scaler = torch.amp.GradScaler()
criterion = synn.ContrastiveLoss()

criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-9
)
model.to(device)

epochs = 30
total_pbar = tqdm(range(epochs))
losses = []

for epoch in total_pbar:
    model.train()
    epoch_pbar = tqdm(dataloader, leave=False)
    epoch_loss = 0
    per100 = len(dataloader) // 100
    inference_logits_per_image, _ = inference(model, epoch)
    logger.info(f"{inference_logits_per_image=}")

    for i, (batch, *_) in enumerate(epoch_pbar):
        images, metadata = dataset.batch_extract(batch)
        captions = [datum["caption"] for datum in metadata]
        texts = tokenizer(captions)

        images = images.to(LOCAL_RANK)
        texts = texts.to(LOCAL_RANK)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            logits_per_image, logits_per_text = model(images, texts)
            loss_img, loss_txt = criterion(logits_per_image, logits_per_text)
            loss = (loss_img + loss_txt) / 2

        if loss.isnan().any():
            logger.warning(f"Loss is NaN at epoch {epoch+1}, iteration {i+1}")
            del loss, loss_img, loss_txt
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss {loss.item():.8f}")
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
            "model": model.state_dict(),
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
    scheduler.step()

wandb.finish()
# %%
