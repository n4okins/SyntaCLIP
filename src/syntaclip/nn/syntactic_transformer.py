from typing import Optional

import torch
import torch.nn as nn

from .syntactic_attention_layer import ResidualSyntacticAttentionEncoderLayer
from .syntactic_distance_gate import SyntacticDistanceGate
from .transformer import TextTransformerEncoder, VisionTransformerEncoder

__all__ = [
    "SyntacticTransformerEncoder",
    "SyntacticTextTransformerEncoder",
    "SyntacticVisionTransformerEncoder",
]


class SyntacticTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_attn_heads: int = 8,
        num_gate_heads: int = 8,
        num_induction_layers: int = 2,
        *,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_attn_heads
        self.num_induction_layers = num_induction_layers
        self.induction_head = SyntacticDistanceGate(
            embed_dim=embed_dim,
            num_lookback_range=3,
            num_gate_heads=num_gate_heads,
            tau=10.0,
            dropout_p=0.0,
        )
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualSyntacticAttentionEncoderLayer(
                    embed_dim=embed_dim,
                    num_attn_heads=num_attn_heads,
                    num_gate_heads=num_gate_heads,
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
