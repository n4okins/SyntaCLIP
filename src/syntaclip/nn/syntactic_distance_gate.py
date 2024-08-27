import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clogging import getColoredLogger

logger = getColoredLogger(__name__)


class SyntacticDistanceGate(nn.Module):
    """
    Syntactic Distance Gate
    - https://aclanthology.org/2021.acl-srw.33/
    """

    def __init__(
        self,
        embed_dim: int,
        num_lookback_range: int = 3,
        num_gate_heads: int = 2,
        *,
        tau: float = 1.0,
        dropout_p: float = 0.0,
        distance_activation_module: nn.Module = nn.Tanh,
        mask_triu: bool = False,
    ):
        super().__init__()
        self.lookback_range = num_lookback_range
        self.tau = tau
        self.num_gate_heads = num_gate_heads
        self.conv = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(
                embed_dim,
                num_gate_heads,
                num_lookback_range,
                padding=num_lookback_range,
            ),
        )
        self.distance_activation = distance_activation_module()
        self.mask_triu = mask_triu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)
        # x: (batch_size, embed_dim, seq_len)
        batch_size, embed_dim, seq_len = x.size()

        # distance: Syntactic Distance [d_i, ...]: i番目の単語の構文距離 (構文高？)
        # distance := distance  (batch_size, seq_len, 1)
        # distance[i] = \tanh(W_D [k_{i-M}, k_{i-M+1}, ..., K_{i}]^{\top} + b_D)
        # conv_input: (batch_size, embed_dim, seq_len)
        distance = self.conv(x)
        # disttance : (batch_size, distance_dim, seq_len + lookback_range)
        distance = distance[:, :, 1 : -self.lookback_range]
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L60
        distance = self.distance_activation(distance)
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L106
        # distance: (batch_size, num_gate_heads, seq_len)
        distance = distance.view(batch_size * self.num_gate_heads, -1, 1).contiguous()
        # distance: (batch_size * num_gates_heads, seq_len, 1)
        # Compute Span Logits
        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L41
        alpha = (F.hardtanh((distance - distance.transpose(2, 1)) * self.tau) + 1) / 2
        # alpha: (batch_size, seq_len, seq_len), 0 <= alpha <= 1
        lower_tri = (
            (alpha.tril(diagonal=-1) + torch.ones_like(alpha).triu(diagonal=0))
            .flip([-1])
            .cumprod(dim=-1)
            .flip([-1])
        )
        if self.mask_triu:
            return lower_tri, distance

        upper_tri = (
            torch.ones_like(alpha).tril(diagonal=0) + alpha.triu(diagonal=1)
        ).cumprod(dim=-1)

        gate = lower_tri * upper_tri

        # https://github.com/i13harada1s/nmt_sync_gram/blob/02d96a2f7e06f6394a549b6170e6a4cc8eb2f250/src/modules/structformer_layer.py#L105
        distance = (
            distance.contiguous()
            .view(batch_size, self.num_gate_heads, seq_len, 1)
            .mean(dim=1)
        )
        # gate := gate  (batch_size * num_gate_heads, seq_len, seq_len), 0 <= gate <= 1
        # distance := distance  (batch_size, seq_len, 1), -1 <= distance <= 1
        return gate, distance
