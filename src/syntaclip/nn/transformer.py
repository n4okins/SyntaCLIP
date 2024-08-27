from typing import Optional

import torch
import torch.nn as nn

from .attention_layer import ResidualAttentionEncoderLayer
from .dropout import PatchDropout
from .layernorm import CastLayerNorm

__all__ = [
    "TransformerEncoder",
    "TextTransformerEncoder",
    "VisionTransformerEncoder",
]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
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
        *,
        input_image_size: int | tuple[int, int] | tuple[int, int, int] = 224,
        patch_embed_dim: int = 768,
        patch_size: tuple[int, int] = (32, 32),
        patch_stride: Optional[tuple[int, int]] = None,
        patch_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.transformer = TransformerEncoder(
            embed_dim=patch_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
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
