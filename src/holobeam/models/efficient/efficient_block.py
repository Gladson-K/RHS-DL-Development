import torch
from torch import nn


class EfficientBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden = dim * expansion

        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)