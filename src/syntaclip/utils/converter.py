import torch.nn as nn

from ..nn.clip import CLIP


def convert_model_params_laion2b_s34b_b79k_to_CLIP512(
    target_model: CLIP,
    model_laion2b_s34b_b79k: nn.Module,
    verbose: bool = False,
    num_freeze_layers: int = 12,
) -> CLIP:
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
    for i in range(num_freeze_layers):
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
