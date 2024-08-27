from typing import Optional

import torch
import torch.nn.functional as F
from utils.clogging import getColoredLogger

logger = getColoredLogger(__name__)


@torch.jit.script
def merge_masks(
    num_heads: int,
    attention_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    query: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[int]]:
    r"""Determine mask type and combine masks if necessary.

    If only one mask is provided, that mask
    and the corresponding mask type will be returned. If both masks are provided, they will be both
    expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
    and mask type 2 will be returned
    Args:
        attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
        key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
        query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
    Returns:
        merged_mask: merged mask
        mask_type: merged mask type (0, 1, or 2)
    """
    merged_mask: Optional[torch.Tensor] = None
    # mask_type = 1: key_padding_mask, 2: attn_mask, 3: key_padding_mask + attn_mask
    mask_type: Optional[int] = None

    if key_padding_mask is not None:
        mask_type = 1
        merged_mask = key_padding_mask

    if attention_mask is not None:
        batch_size, seq_len, _ = query.shape
        mask_type = 2

        if attention_mask.dim() == 3:
            attention_mask_expanded = attention_mask.view(
                batch_size, -1, seq_len, seq_len
            )
        else:
            attention_mask_expanded = attention_mask.view(
                1, 1, seq_len, seq_len
            ).expand(batch_size, num_heads, -1, -1)
        merged_mask = attention_mask_expanded

        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.view(
                batch_size, 1, 1, seq_len
            ).expand(-1, num_heads, -1, -1)
            merged_mask = attention_mask_expanded + key_padding_mask_expanded

    return merged_mask, mask_type


def get_tensor_eps_for_jit(
    x: torch.Tensor,
    epsf16: float = torch.finfo(torch.float16).eps,
    epsbf16: float = torch.finfo(torch.bfloat16).eps,
    epsf32: float = torch.finfo(torch.float32).eps,
    epsf64: float = torch.finfo(torch.float64).eps,
) -> float:
    # Match aren't supported in jit
    if x.dtype == torch.float16:
        return epsf16
    elif torch.bfloat16:
        return epsbf16
    elif torch.float32:
        return epsf32
    elif torch.float64:
        return epsf64
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")


@torch.jit.script
def syntactic_multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_attn_heads: int,
    num_gate_heads: int,
    in_proj_weight: Optional[torch.Tensor],
    in_proj_bias: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    attn_gate: Optional[torch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    query_proj_weight: Optional[torch.Tensor] = None,
    key_proj_weight: Optional[torch.Tensor] = None,
    value_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Forward method for SyntacticMultiHeadAttention.
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if attn_gate is None:
        return F.multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim_to_check,
            num_attn_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=query_proj_weight,
            k_proj_weight=key_proj_weight,
            v_proj_weight=value_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    if F.has_torch_function(tens_ops):
        return F.handle_torch_function(
            syntactic_multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_attn_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=query_proj_weight,
            k_proj_weight=key_proj_weight,
            v_proj_weight=value_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = F._mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_attn_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query.unsqueeze_(1)
        key.unsqueeze_(1)
        value.unsqueeze_(1)
        if key_padding_mask is not None:
            key_padding_mask.unsqueeze_(0)

    # set up shape vars
    target_length, batch_size, embed_dim = query.shape
    source_length, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_attn_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_attn_heads
    assert (
        head_dim * num_attn_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_attn_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    # compute in-projection
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        query, key, value = F._in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias
        )
    else:
        assert (
            query_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            key_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            value_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            query_bias = key_bias = value_bias = None
        else:
            query_bias, key_bias, value_bias = in_proj_bias.chunk(3)
        query, key, value = F._in_projection(
            query,
            key,
            value,
            query_proj_weight,
            key_proj_weight,
            value_proj_weight,
            query_bias,
            key_bias,
            value_bias,
        )

    # prep attention mask
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (target_length, source_length)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (
                batch_size * num_attn_heads,
                target_length,
                source_length,
            )
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        key = torch.cat([key, bias_k.repeat(1, batch_size, 1)])
        value = torch.cat([value, bias_v.repeat(1, batch_size, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    # reshape q, k, v for multihead attention and make them batch first
    query = query.view(target_length, batch_size * num_attn_heads, head_dim).transpose(
        0, 1
    )
    if static_k is None:
        key = key.view(key.shape[0], batch_size * num_attn_heads, head_dim).transpose(
            0, 1
        )
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        key = static_k
    if static_v is None:
        value = value.view(
            value.shape[0], batch_size * num_attn_heads, head_dim
        ).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        value = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (batch_size * num_attn_heads, 1, head_dim)
        key = torch.cat(
            [key, torch.zeros(zero_attn_shape, dtype=key.dtype, device=key.device)],
            dim=1,
        )
        value = torch.cat(
            [
                value,
                torch.zeros(zero_attn_shape, dtype=value.dtype, device=value.device),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    source_length = key.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = (
            key_padding_mask.view(batch_size, 1, 1, source_length)
            .expand(-1, num_attn_heads, -1, -1)
            .reshape(batch_size * num_attn_heads, 1, source_length)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # (deep breath) calculate attention and out projection
    batch_attn_heads, target_length, head_dim = query.shape
    scaled_query = query * ((1.0 / float(head_dim)) ** 0.5)

    assert not (
        is_causal and attn_mask is None
    ), "FIXME: is_causal not implemented for need_weights"

    # ADDED: Compute the attention weights
    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_miltihead_attention.py#L329
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(
            attn_mask, scaled_query, key.transpose(-2, -1)
        )
    else:
        attn_output_weights = torch.bmm(scaled_query, key.transpose(-2, -1))

    # attn_output_weights: (batch_size * num_attn_heads, target_length, source_length)

    attn_output_weights_dtype = attn_output_weights.dtype
    attn_output_weights = F.softmax(attn_output_weights, dim=-1).type(torch.float32)

    # ADDED: Adjust attn_gates
    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L86
    num_heads_diff = num_attn_heads - num_gate_heads

    if num_heads_diff > 0:
        attn_gate = (
            torch.cat(
                [
                    attn_gate.view(
                        batch_size, num_gate_heads, target_length, source_length
                    ),
                    attn_gate.new_ones(
                        (
                            batch_size,
                            num_heads_diff,
                            target_length,
                            source_length,
                        )
                    ),
                ],
                dim=1,
            )
            .view(batch_size * num_attn_heads, target_length, source_length)
            .contiguous()
        )

    # Apply attn_gate
    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_miltihead_attention.py#L361
    attn_output_weights = attn_output_weights * attn_gate.type_as(attn_output_weights)
    attn_output_weights = attn_output_weights / (
        attn_output_weights.sum(dim=-1, keepdim=True)
        + get_tensor_eps_for_jit(attn_output_weights)
    )
    attn_output_weights = attn_output_weights.type(attn_output_weights_dtype)

    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_miltihead_attention.py#L377
    attn_output = torch.bmm(attn_output_weights, value)

    attn_output = (
        attn_output.transpose(0, 1)
        .contiguous()
        .view(target_length * batch_size, embed_dim)
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(target_length, batch_size, attn_output.size(1))

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(
        batch_size, num_attn_heads, target_length, source_length
    )
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights
