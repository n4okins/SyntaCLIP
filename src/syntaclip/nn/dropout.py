import torch
import torch.nn as nn


class PatchDropout(nn.Module):
    """Patch Dropout
    https://arxiv.org/abs/2212.00794
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L49
    Args:
        p (float): Probability of an element to be zeroed
        exclude_first_token (bool): Exclude first token
    """

    def __init__(self, p: float = 0.0, exclude_first_token=True):
        super().__init__()
        assert 0 <= p < 1.0
        self.prob = p
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x
