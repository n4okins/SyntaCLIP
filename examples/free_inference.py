# %%
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import open_clip
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchinfo
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from utils.clogging import getColoredLogger
from utils.dummy import DummyObject
from utils.initialize import initializer

import syntaclip.nn as synn

# Logger Settings
logger = getColoredLogger(__name__)
# Project Init
PROJECT_ROOT = initializer(globals(), logger=logger)
logger.setLevel("DEBUG")

ARCH_NAME = "SyntaCLIP512"
PROJECT_NAME = f"{ARCH_NAME}-Visualize"
USE_WANDB_LOG = False

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
fig_dir = PROJECT_ROOT / "ignores" / "figs" / f"{TIMESTAMP}-Pretrained-{ARCH_NAME}"
fig_dir.mkdir(parents=True, exist_ok=True)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model_path = (
    PROJECT_ROOT
    / "models"
    / ARCH_NAME
    / f"{ARCH_NAME}-Finetune-Dropout/checkpoint_29.pth"
)

# model_path = PROJECT_ROOT / "models" / "CLIP512" / "CLIP512-Finetune-Dropout/first.pth"
logger.info(f"{model_path=}")
assert model_path.exists(), f"Model not found: {model_path}"

if ARCH_NAME == "SyntaCLIP512":
    model = synn.CLIP(
        visual_backbone=synn.SyntacticVisionTransformerEncoder(
            attn_dropout_p=0.25,
            gate_dropout_p=0.5,
        ),
        textual_backbone=synn.SyntacticTextTransformerEncoder(
            attn_dropout_p=0.25,
            gate_dropout_p=0.5,
        ),
    )
elif ARCH_NAME == "CLIP512":
    model = synn.CLIP(
        visual_backbone=synn.VisionTransformerEncoder(dropout_p=0.25),
        textual_backbone=synn.TextTransformerEncoder(dropout_p=0.25),
    )
state = torch.load(model_path, map_location=device)
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
model.eval()

torchinfo.summary(
    model,
    input_size=[(1, 3, 224, 224), (1, 77)],
    dtypes=[torch.float32, torch.long],
    device=device,
)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
seed = 0
random.seed(seed)

# load and preprocess the images
samples: list[Image.Image] = [
    Image.open(PROJECT_ROOT / "examples" / "images" / "01.jpg"),
]
# load and preprocess the text
masks = ["red", "green", "yellow", "orange", "purple"]
sentences = [f"In this picture, the color of the lemon is {mask}." for mask in masks]

images = torch.stack([transform(sample) for sample in samples])
num_samples = len(images)
tokens = tokenizer(sentences)
images = images.to(device)
tokens = tokens.to(device)
if IS_CUDA_AVAILABLE:
    images, tokens = images.to(LOCAL_RANK), tokens.to(LOCAL_RANK)

model = model.to(device)
images_distances, tokens_distances = None, None
with torch.inference_mode(), torch.amp.autocast(
    device_type=device.type, dtype=torch.float32
):
    if hasattr(model.visual, "get_distance"):
        if IS_DISTRIBUTED:
            images_feats, images_weights, images_distances = (
                model.module.visual.get_distance(images)
            )
            tokens_feats, tokens_weights, tokens_distances = (
                model.module.textual.get_distance(tokens)
            )
            logits_per_image, logits_per_text = model(images, tokens)
        else:
            images_feats, images_weights, images_distances = model.visual.get_distance(
                images
            )
            tokens_feats, tokens_weights, tokens_distances = model.textual.get_distance(
                tokens
            )
            logits_per_image, logits_per_text = model(images, tokens)
    else:
        images_feats, images_weights = model.visual(images, return_all_weights=True)
        tokens_feats, tokens_weights = model.textual(tokens, return_all_weights=True)
        logits_per_image, logits_per_text = model(images, tokens)

print(ARCH_NAME)
print(samples)
print(sentences)
print(logits_per_image)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.imshow(samples[0])
# ax.axis("off")
# ax.set_title("In this picture, the color of the lemon is [mask].")
ax.bar(range(len(sentences)), logits_per_image[0].cpu().detach().numpy(), color=masks)
ax.set_xticks(range(len(sentences)))
ax.set_ylim(0, 1)
ax.set_xticklabels(masks)
ax.grid(True)
ax.set_title("[mask] probability")
fig.tight_layout()
plt.savefig(fig_dir / f"in_this_picture_{ARCH_NAME}.jpg")

# %%
