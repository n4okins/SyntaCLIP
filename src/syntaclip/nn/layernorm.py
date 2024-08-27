import torch
import torch.nn as nn


class CastLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype).
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L24

    from https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L157
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        x = super().forward(x.type(torch.float32))
        return x.type(orig_type)
