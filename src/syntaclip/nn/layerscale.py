import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Layer scale
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L39
    Args:
        embed_dim (int): Embedding dimension
        init_scale_ratio (float): Initial scale ratio
        inplace (bool): Inplace operation
    """

    def __init__(
        self, embed_dim: int, init_scale_ratio: float = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_scale_ratio * torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
