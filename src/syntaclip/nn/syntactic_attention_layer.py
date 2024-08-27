from typing import Optional

import torch
import torch.nn as nn

from .attention_layer import ResidualAttentionEncoderLayer
from .syntactic_distance_gate import SyntacticDistanceGate
from .syntactic_multihead_attention import SyntacticMultiheadAttention

__all__ = ["ResidualSyntacticAttentionEncoderLayer"]


class ResidualSyntacticAttentionEncoderLayer(ResidualAttentionEncoderLayer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_attn_heads: int = 8,
        num_gate_heads: int = 2,
        num_lookback_range: int = 3,
        tau: float = 10.0,
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
        )
        self.gate = SyntacticDistanceGate(
            embed_dim=embed_dim,
            num_lookback_range=num_lookback_range,
            num_gate_heads=num_attn_heads,
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
        attn_out, attn_weight = self.attention(
            _normed_query,
            key,
            value,
            need_weights=True,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
        )
        x = query + self.layerscale_1(attn_out)
        x = x + self.layerscale_2(self.res_mlp(self.layernorm_2(x)))
        return x, attn_weight
