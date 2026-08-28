from torch import nn


class SpatialAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        y, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )

        return self.norm(
            residual + y
        )