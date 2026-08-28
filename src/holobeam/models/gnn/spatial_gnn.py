import torch
from torch import nn


class SpatialGNNBlock(nn.Module):

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.message = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:

        messages = self.message(x)

        aggregated = torch.matmul(
            adjacency,
            messages,
        )

        update = self.update(
            torch.cat(
                [
                    x,
                    aggregated,
                ],
                dim=-1,
            )
        )

        return x + update