import torch
import torch.nn as nn


class CLIP(nn.Module):
    def __init__(
        self,
        visual_backbone: nn.Module,
        textual_backbone: nn.Module,
    ):
        super().__init__()
        self.visual = visual_backbone
        self.textual = textual_backbone
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )
        self.logit_bias = nn.Parameter(torch.zeros([]))

    @property
    def dtype(self):
        if hasattr(self.visual, "dtype"):
            return self.visual.dtype
        elif hasattr(self.textual, "dtype"):
            return self.textual.dtype
        else:
            return self.logit_scale.dtype

    def encode_image(self, images: torch.Tensor, normalize: bool = True):
        feats, *_ = self.visual(images)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True):
        feats, *_ = self.textual(tokens)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def get_features(
        self, images: torch.Tensor, tokens: torch.Tensor, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images, normalize=normalize)
        text_features = self.encode_text(tokens, normalize=normalize)
        return image_features, text_features

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        *,
        normalize: bool = True,
        softmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features, text_features = self.get_features(
            images, tokens, normalize=normalize
        )
        logits_per_image = (
            self.logit_scale.exp() * image_features @ text_features.t()
            + self.logit_bias
        )
        if softmax:
            logits_per_image = logits_per_image.softmax(dim=1)
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
