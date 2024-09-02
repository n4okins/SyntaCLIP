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

ARCH_NAME = "CLIP512"
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


split = "val2017"
mscoco_root = Path.home() / "datasets" / "MSCOCO"
image_dir = mscoco_root / split


@dataclass
class COCOImageAndCaption:
    image_id: int
    caption: str
    id: int  # caption_id

    def load_image(self) -> Image.Image:
        return Image.open(image_dir / f"{self.image_id:012d}.jpg")

    @property
    def caption_id(self) -> int:
        return self.id


annotations_dir = mscoco_root / "annotations" / f"captions_{split}.json"
annotations_json = json.load(annotations_dir.open("r"))
image_and_captions: tuple[COCOImageAndCaption, ...] = tuple(
    sorted(
        map(lambda x: COCOImageAndCaption(**x), annotations_json["annotations"]),
        key=lambda x: x.image_id,
    ),
)

# # %%
# index = 105
# plt.imshow(annotations[index].load_image())
# plt.title(annotations[index].caption)
# plt.axis("off")
# plt.tight_layout()
# plt.show()
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model_path = (
    PROJECT_ROOT
    / "models"
    / ARCH_NAME
    / f"{ARCH_NAME}-Finetune-Dropout/checkpoint_29.pth"
)
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
# %%
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
num_samples = 10
seed = 0
random.seed(seed)
samples: list[COCOImageAndCaption] = random.sample(image_and_captions, num_samples)
images = torch.stack([transform(sample.load_image()) for sample in samples])
sentences = [sample.caption for sample in samples]
tokens = tokenizer(sentences)
images = images.to(LOCAL_RANK).to(device)
tokens = tokens.to(LOCAL_RANK).to(device)
model = model.to(device)
images_distances, tokens_distances = None, None
with torch.inference_mode(), torch.amp.autocast(
    device_type="cuda", dtype=torch.float32
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

print(samples)
print(sentences)
# %%
for i in tqdm(range(num_samples)):  # [batch_size, 77, 1]
    caption_dir = fig_dir / f"{samples[i].id:012d}"
    caption_dir.mkdir(parents=True, exist_ok=True)
    _tokens = tokens[i].tolist()
    end_index = _tokens.index(tokenizer.all_special_ids[1])
    _tokens = _tokens[: end_index + 1]
    decoded_tokens = [
        tokenizer.decoder[token].replace("</w>", " ") for token in _tokens
    ]

    if images_distances is not None and tokens_distances is not None:
        # TOKEN DISTANCE
        token_distance = tokens_distances[i].squeeze(1)  # [77]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_ylim(-1, 1)
        ax.bar(
            range(len(_tokens)), token_distance[: end_index + 1].cpu().detach().numpy()
        )
        ax.set_xticks(range(len(_tokens)))
        ax.set_xticklabels(
            [tokenizer.decoder[token].replace("</w>", " ") for token in _tokens],
            rotation=90,
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(caption_dir / "token_distance.png")
        fig.clear()
        plt.close(fig)

        # TOKEN DISTANCE
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_ylim(-1, 1)
        ax.bar(
            range(77), token_distance.cpu().detach().numpy()
        )
        ax.set_xticks(range(77))
        ax.set_xticklabels(
            [tokenizer.decoder[token].replace("</w>", " ") for token in _tokens]
            + [" "] * (77 - len(_tokens)),
            rotation=90,
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(caption_dir / "all_token_distance.png")
        fig.clear()
        plt.close(fig)

        # ABS TOKEN DISTANCE
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_ylim(-0.05, 1)
        ax.bar(
            range(len(_tokens)),
            token_distance[: end_index + 1].abs().cpu().detach().numpy(),
        )
        ax.set_xticks(range(len(_tokens)))
        ax.set_xticklabels(
            [tokenizer.decoder[token].replace("</w>", " ") for token in _tokens],
            rotation=90,
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(caption_dir / "token_distance_abs.png")
        fig.clear()
        plt.close(fig)

        # TOKEN DISTANCE
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_ylim(-1, 1)
        ax.bar(
            range(77), token_distance.abs().cpu().detach().numpy()
        )
        ax.set_xticks(range(len(_tokens)))
        ax.set_xticklabels(
            [tokenizer.decoder[token].replace("</w>", " ") for token in _tokens],
            rotation=90,
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(caption_dir / "all_token_distance_abs.png")
        fig.clear()
        plt.close(fig)


        # IMAGE DISTANCE
        image_distance = images_distances[i].squeeze(1)  # [50]
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.set_ylim(-1, 1)
        ax.set_xticks(range(len(image_distance)))
        ax.set_xticklabels(
            ["CLS_TOKEN"] + [f"Patch-{i}" for i in range(len(image_distance) - 1)],
            rotation=90,
        )
        ax.bar(range(len(image_distance)), image_distance.cpu().detach().numpy())
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(caption_dir / "image_distance.png")
        fig.clear()
        plt.close(fig)

        # ABS IMAGE DISTANCE
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.set_ylim(-0.05, 1)
        ax.set_xticks(range(len(image_distance)))
        ax.set_xticklabels(
            ["CLS_TOKEN"] + [f"Patch-{i}" for i in range(len(image_distance) - 1)],
            rotation=90,
        )
        ax.bar(range(len(image_distance)), image_distance.abs().cpu().detach().numpy())
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(caption_dir / "image_distance_abs.png")
        fig.clear()
        plt.close(fig)

    # TOKEN WEIGHTS
    for j in range(12):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        token_weights = tokens_weights[i]
        ax.imshow(token_weights[j].cpu().detach().numpy(), cmap="viridis", alpha=0.8)
        ax.set_yticks(range(len(decoded_tokens)))
        ax.set_yticklabels(
            decoded_tokens,
        )
        ax.set_xticks(range(len(decoded_tokens)))
        ax.set_xticklabels(
            decoded_tokens,
            rotation=90,
        )
        ax.set_title(f"Token Attention-{j}")
        ax.grid(linewidth=0.5, linestyle="--")
        ax.xaxis.tick_top()
        fig.tight_layout()
        fig.savefig(caption_dir / f"all_token_weights_{j:02d}.png")

        fig.clear()
        plt.close(fig)

    for j in range(12):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        token_weights = tokens_weights[i]
        ax.imshow(
            token_weights[j][: end_index + 1, : end_index + 1].cpu().detach().numpy(),
            cmap="viridis",
            alpha=0.8,
        )
        ax.set_yticks(range(end_index + 1))
        ax.set_yticklabels(
            decoded_tokens[: end_index + 1],
        )
        ax.set_xticks(range(end_index + 1))
        ax.set_xticklabels(
            decoded_tokens[: end_index + 1],
            rotation=90,
        )
        ax.set_title(f"Token Attention-{j}")
        ax.grid(linewidth=0.5, linestyle="--")
        ax.xaxis.tick_top()
        fig.tight_layout()
        fig.savefig(caption_dir / f"token_weights_{j:02d}.png")

        fig.clear()
        plt.close(fig)

    # IMAGE WEIGHTS
    for j in range(12):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        image_weights = images_weights[i]
        ax.imshow(image_weights[j].cpu().detach().numpy(), cmap="viridis")
        ax.set_xticks(range(50))
        ax.set_xticklabels(
            ["CLS_TOKEN"] + [f"Patch-{i}" for i in range(50 - 1)],
            rotation=90,
        )
        ax.set_yticks(range(50))
        ax.set_yticklabels(
            ["CLS_TOKEN"] + [f"Patch-{i}" for i in range(50 - 1)],
        )
        ax.set_title(f"Image Attention-{j}")
        ax.grid(linewidth=0.5, linestyle="--")
        ax.xaxis.tick_top()
        fig.tight_layout()
        fig.savefig(caption_dir / f"image_weights_{j:02d}.png")
        fig.clear()
        plt.close(fig)

    # IMAGE
    patch = (
        images[i]
        .reshape(3, 7, 32, 7, 32)
        .permute(0, 1, 3, 2, 4)
        .reshape(3, -1, 32, 32)
        .permute(1, 0, 2, 3)
    )
    recon = torchvision.utils.make_grid(patch, nrow=7)
    arr = (recon * 255).type(torch.uint8).detach().cpu().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    canvas = ImageDraw.Draw(img)
    for pr in range(7):
        for pl in range(7):
            canvas.text(
                (8 + pl * 34, 8 + pr * 34), str(pr * 7 + pl), fill=(255, 20, 20)
            )

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(caption_dir / "image.png")
    fig.clear()
    plt.close(fig)

    dump_dict = {
        "i": i,
        "image_id": samples[i].image_id,
        "caption_id": samples[i].caption_id,
        "caption": samples[i].caption,
        "tokens": [tokenizer.decoder[token].replace("</w>", " ") for token in _tokens],
        "model_path": str(model_path),
        "logits_per_image": logits_per_image[i].cpu().detach().numpy().tolist(),
        "logits_per_image_argmax": logits_per_image[i].argmax().cpu().detach().item(),
    }

    if images_distances is not None and tokens_distances is not None:
        dump_dict.update(
            {
                "token_distance": token_distance.cpu().detach().numpy().tolist(),
                "image_distance": image_distance.cpu().detach().numpy().tolist(),
            }
        )

    json.dump(
        dump_dict,
        (caption_dir / "info.json").open("w"),
        indent=4,
    )


# %%
