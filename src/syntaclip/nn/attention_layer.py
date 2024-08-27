from typing import Optional

import torch
import torch.nn as nn

from .layernorm import CastLayerNorm
from .layerscale import LayerScale
from .multihead_attention import MultiheadAttention

__all__ = ["ResidualAttentionEncoderLayer"]

class ResidualAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
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
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
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
