import torch
from torch import nn

from .common.embeddings import ConfigurationEncoder
from .efficient.efficient_block import EfficientBlock
from .gnn.spatial_gnn import SpatialGNNBlock
from .gnn.attention import SpatialAttention


class HoloBeamGEN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 6,
        attention_heads: int = 8,
        RFno: int = 1,
        L: int = 4,
    ):
        super().__init__()

        self.RFno = RFno
        self.L = L

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.efficient_blocks = nn.ModuleList(
            [
                EfficientBlock(hidden_dim)
                for _ in range(2)
            ]
        )

        self.gnn_blocks = nn.ModuleList(
            [
                SpatialGNNBlock(hidden_dim)
                for _ in range(gnn_layers)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                SpatialAttention(
                    hidden_dim,
                    attention_heads,
                )
                for _ in range(2)
            ]
        )

        self.config_encoder = ConfigurationEncoder(
            input_dim=16,
            hidden_dim=hidden_dim,
        )

        self.M_head = nn.Linear(
            hidden_dim,
            2 * RFno,
        )

        self.w_head = nn.Linear(
            hidden_dim,
            2 * NrPlaceholder,
        )

    def forward(
        self,
        element_features,
        configuration,
        adjacency,
    ):

        x = self.input_projection(
            element_features
        )

        config = self.config_encoder(
            configuration
        )

        x = x + config.unsqueeze(1)

        for block in self.efficient_blocks:
            x = block(x)

        for block in self.gnn_blocks:
            x = block(
                x,
                adjacency,
            )

        for block in self.attention_blocks:
            x = block(x)

        M = self.M_head(x)

        w = self.w_head(
            x.mean(dim=1)
        )

        return {
            "M": M,
            "w": w,
        }