# %%
import os

import open_clip
import requests
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchinfo
from PIL import Image
from torchvision import transforms
from utils.clogging import getColoredLogger
from utils.initialize import initializer

import syntaclip.nn as synn

# Logger Settings
logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")

# Project Init
PROJECT_ROOT = initializer(globals(), logger=logger)
PROJECT_NAME = "misc"


def convert_model_params_laion2b_s34b_b79k_to_CLIP512(
    target_model: synn.CLIP, model_laion2b_s34b_b79k: nn.Module, verbose: bool = False
) -> synn.CLIP:
    model_laion2b_s34b_b79k_state_dict = model_laion2b_s34b_b79k.state_dict()
    weight_mapping = {
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.head_weight",
        "visual.class_embedding": "visual.class_embedding",
        "visual.positional_embedding": "visual.positional_embedding",
        "visual.proj": "visual.head_weight",
        "visual.conv1.weight": "visual.conv.weight",
        "visual.ln_pre.weight": "visual.layernorm_pre.weight",
        "visual.ln_pre.bias": "visual.layernorm_pre.bias",
        "visual.ln_post.weight": "visual.layernorm_post.weight",
        "visual.ln_post.bias": "visual.layernorm_post.bias",
        "token_embedding.weight": "textual.embedding.weight",
        "logit_scale": "logit_scale",
        "ln_final.weight": "textual.layernorm_post.weight",
        "ln_final.bias": "textual.layernorm_post.bias",
    }
    for i in range(12):
        weight_mapping.update(
            {
                f"visual.transformer.resblocks.{i}.ln_1.weight": f"visual.transformer.res_attn_blocks.{i}.layernorm_1.weight",
                f"visual.transformer.resblocks.{i}.ln_1.bias": f"visual.transformer.res_attn_blocks.{i}.layernorm_1.bias",
                f"visual.transformer.resblocks.{i}.attn.in_proj_weight": f"visual.transformer.res_attn_blocks.{i}.attention.in_proj_weight",
                f"visual.transformer.resblocks.{i}.attn.in_proj_bias": f"visual.transformer.res_attn_blocks.{i}.attention.in_proj_bias",
                f"visual.transformer.resblocks.{i}.attn.out_proj.weight": f"visual.transformer.res_attn_blocks.{i}.attention.out_proj.weight",
                f"visual.transformer.resblocks.{i}.attn.out_proj.bias": f"visual.transformer.res_attn_blocks.{i}.attention.out_proj.bias",
                f"visual.transformer.resblocks.{i}.ln_2.weight": f"visual.transformer.res_attn_blocks.{i}.layernorm_2.weight",
                f"visual.transformer.resblocks.{i}.ln_2.bias": f"visual.transformer.res_attn_blocks.{i}.layernorm_2.bias",
                f"visual.transformer.resblocks.{i}.mlp.c_fc.weight": f"visual.transformer.res_attn_blocks.{i}.res_mlp.0.weight",
                f"visual.transformer.resblocks.{i}.mlp.c_fc.bias": f"visual.transformer.res_attn_blocks.{i}.res_mlp.0.bias",
                f"visual.transformer.resblocks.{i}.mlp.c_proj.weight": f"visual.transformer.res_attn_blocks.{i}.res_mlp.2.weight",
                f"visual.transformer.resblocks.{i}.mlp.c_proj.bias": f"visual.transformer.res_attn_blocks.{i}.res_mlp.2.bias",
                f"transformer.resblocks.{i}.ln_1.weight": f"textual.transformer.res_attn_blocks.{i}.layernorm_1.weight",
                f"transformer.resblocks.{i}.ln_1.bias": f"textual.transformer.res_attn_blocks.{i}.layernorm_1.bias",
                f"transformer.resblocks.{i}.attn.in_proj_weight": f"textual.transformer.res_attn_blocks.{i}.attention.in_proj_weight",
                f"transformer.resblocks.{i}.attn.in_proj_bias": f"textual.transformer.res_attn_blocks.{i}.attention.in_proj_bias",
                f"transformer.resblocks.{i}.attn.out_proj.weight": f"textual.transformer.res_attn_blocks.{i}.attention.out_proj.weight",
                f"transformer.resblocks.{i}.attn.out_proj.bias": f"textual.transformer.res_attn_blocks.{i}.attention.out_proj.bias",
                f"transformer.resblocks.{i}.ln_2.weight": f"textual.transformer.res_attn_blocks.{i}.layernorm_2.weight",
                f"transformer.resblocks.{i}.ln_2.bias": f"textual.transformer.res_attn_blocks.{i}.layernorm_2.bias",
                f"transformer.resblocks.{i}.mlp.c_fc.weight": f"textual.transformer.res_attn_blocks.{i}.res_mlp.0.weight",
                f"transformer.resblocks.{i}.mlp.c_fc.bias": f"textual.transformer.res_attn_blocks.{i}.res_mlp.0.bias",
                f"transformer.resblocks.{i}.mlp.c_proj.weight": f"textual.transformer.res_attn_blocks.{i}.res_mlp.2.weight",
                f"transformer.resblocks.{i}.mlp.c_proj.bias": f"textual.transformer.res_attn_blocks.{i}.res_mlp.2.bias",
            }
        )
    target_model_state_dict = target_model.state_dict()
    for k, v in weight_mapping.items():
        assert (
            model_laion2b_s34b_b79k_state_dict[k].shape
            == target_model_state_dict[v].shape
        ), f"{k=}, {v=}, {model_laion2b_s34b_b79k_state_dict[k].shape=}, {target_model_state_dict[v].shape=}"
        if verbose:
            print(k, "->", v)
        target_model_state_dict[v] = model_laion2b_s34b_b79k_state_dict[k]
    return target_model_state_dict


model = synn.CLIP(
    visual_backbone=synn.SyntacticVisionTransformerEncoder(),
    textual_backbone=synn.SyntacticTextTransformerEncoder(),
)
model.eval()

print(
    torchinfo.summary(
        model,
        input_size=[(1, 3, 224, 224), (1, 77)],
        dtypes=[torch.float32, torch.long],
        device="cpu",
    )
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
openclip_model, _, transform_openclip = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None),
)
openclip_model.eval()
print(
    torchinfo.summary(
        openclip_model,
        input_size=[(1, 3, 224, 224), (1, 77)],
        dtypes=[torch.float32, torch.long],
        device="cpu",
    )
)
state = convert_model_params_laion2b_s34b_b79k_to_CLIP512(model, openclip_model)
model.load_state_dict(state)
openclip_model.to("cpu")
model.to("cpu")


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

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    image_features_openclip = openclip_model.encode_image(images)
    text_features_openclip = openclip_model.encode_text(tokens)
    image_features_openclip /= image_features_openclip.norm(dim=-1, keepdim=True)
    text_features_openclip /= text_features_openclip.norm(dim=-1, keepdim=True)
    probs_openclip = (100 * image_features_openclip @ text_features_openclip.T).softmax(
        dim=-1
    )
    logits_per_image_openclip, logits_per_text_openclip = openclip_model.get_logits(
        images, tokens
    )
    logits_per_image_openclip = logits_per_image_openclip.softmax(dim=-1)
    logits_per_text_openclip = logits_per_image_openclip.t()

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    images_feats, images_weights, images_distances = model.visual.get_distance(images)
    tokens_feats, tokens_weights, tokens_distances = model.textual.get_distance(tokens)
    logits_per_image, logits_per_text = model(images, tokens)

print(f"{images_feats.size()=} {images_weights.size()=} {images_distances.size()=}")
print(f"{tokens_feats.size()=} {tokens_weights.size()=} {tokens_distances.size()=}")

# Check if the features are close
print(f"{logits_per_image_openclip=}")
print(f"{logits_per_image=}")

# Check if the logits are close (False)
print(f"{torch.allclose(logits_per_image_openclip, logits_per_image)=}")
print(f"{torch.allclose(logits_per_text_openclip, logits_per_text)=}")

model_dir = PROJECT_ROOT / "ignores" / "models" / "openclip_convert"
model_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    model.state_dict(),
    model_dir / "syntaclip512.pth",
)
# %%
