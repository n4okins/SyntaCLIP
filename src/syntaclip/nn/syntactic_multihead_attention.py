from typing import Optional

import torch
import torch.nn.functional as F

from .functional import merge_masks, syntactic_multi_head_attention_forward
from .multihead_attention import MultiheadAttention


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