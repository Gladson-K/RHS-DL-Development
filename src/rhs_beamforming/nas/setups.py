"""Five architecture variants for the real 64x64 RHS beamforming task.

These are the candidate *setups* that combine the HaloBeam-GNN design ideas:

    1. ``mlp``          - tabular residual MLP (current best baseline)
    2. ``gnn``          - learned graph message passing over tokens
    3. ``transformer``  - Transformer encoder over tokens
    4. ``hybrid``       - GNN + Transformer interleaved
    5. ``hybrid_jev``   - GNN + Transformer + differentiable JEV decision head

Every model maps a 10-dim configuration vector ``x`` to the flat target
``[Re(M).ravel(), Im(M).ravel(), rate]`` of dimension 25601, so they are
drop-in compatible with ``nas.real_trainer`` / ``nas.real_search``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..models.gnn_block import GraphMessagePassingBlock
from ..models.jev_head import JEVDecisionHead
from ..models.transformer import TokenStem, TransformerEncoderBlock
from .real_space import (
    INPUT_DIM,
    OUTPUT_DIM,
    Architecture,
    build_model,
    count_parameters,
)


@dataclass
class SetupConfig:
    """Hyper-parameters shared by the token-based setups."""

    name: str = "gnn"
    family: str = "gnn"  # mlp | gnn | transformer | hybrid | hybrid_jev
    description: str = ""
    hidden_dim: int = 128
    num_tokens: int = 16
    num_blocks: int = 2
    heads: int = 4
    num_eig: int = 4
    dropout: float = 0.1
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    mlp_hidden: tuple[int, ...] = (256, 256, 128, 64)
    mlp_activation: str = "leaky_relu"


class TokenBackboneNet(nn.Module):
    """Token stem -> interleaved GNN/Transformer blocks -> pooled output head."""

    def __init__(self, cfg: SetupConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = TokenStem(cfg.input_dim, cfg.hidden_dim, cfg.num_tokens, cfg.dropout)
        blocks: list[nn.Module] = []
        for i in range(cfg.num_blocks):
            if cfg.family == "gnn":
                blocks.append(GraphMessagePassingBlock(cfg.hidden_dim, cfg.num_tokens, dropout=cfg.dropout))
            elif cfg.family == "transformer":
                blocks.append(TransformerEncoderBlock(cfg.hidden_dim, cfg.heads, dropout=cfg.dropout))
            else:  # hybrid / hybrid_jev alternate GNN and Transformer
                blocks.append(GraphMessagePassingBlock(cfg.hidden_dim, cfg.num_tokens, dropout=cfg.dropout))
                blocks.append(TransformerEncoderBlock(cfg.hidden_dim, cfg.heads, dropout=cfg.dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

        if cfg.family == "hybrid_jev":
            self.head = JEVDecisionHead(cfg.hidden_dim, cfg.output_dim, cfg.num_eig, cfg.dropout)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.output_dim),
            )

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        if isinstance(self.head, JEVDecisionHead):
            return self.head(h)
        return self.head(h.mean(dim=1))


def build_setup(cfg: SetupConfig) -> nn.Module:
    """Instantiate one of the five setups."""
    if cfg.family == "mlp":
        arch = Architecture(
            hidden_layers=tuple(cfg.mlp_hidden),
            activation=cfg.mlp_activation,
            dropout=0.1,
            batchnorm=False,
            residual=True,
            head="deep",
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
        )
        return build_model(arch)
    if cfg.family in {"gnn", "transformer", "hybrid", "hybrid_jev"}:
        return TokenBackboneNet(cfg)
    raise ValueError(f"Unknown setup family: {cfg.family}")


SETUP_VARIANTS: dict[str, SetupConfig] = {
    "mlp": SetupConfig(
        name="mlp",
        family="mlp",
        description="Tabular residual MLP (current best baseline, 256-256-128-64).",
    ),
    "gnn": SetupConfig(
        name="gnn",
        family="gnn",
        description="HaloBeam-GNN: learned graph message passing over 16 tokens.",
    ),
    "transformer": SetupConfig(
        name="transformer",
        family="transformer",
        description="Transformer encoder (2 blocks, 4 heads) over 16 tokens.",
    ),
    "hybrid": SetupConfig(
        name="hybrid",
        family="hybrid",
        description="GNN + Transformer interleaved (attachable Transformer).",
    ),
    "hybrid_jev": SetupConfig(
        name="hybrid_jev",
        family="hybrid_jev",
        description="GNN + Transformer + differentiable JEV eigen-decision head.",
    ),
}


def build_setup_by_name(name: str) -> nn.Module:
    return build_setup(SETUP_VARIANTS[name])


def setup_table() -> list[dict]:
    """Descriptive table (name, family, params, description) for docs/CLI."""
    rows = []
    for name, cfg in SETUP_VARIANTS.items():
        model = build_setup(cfg)
        rows.append(
            {
                "setup": name,
                "family": cfg.family,
                "hidden_dim": cfg.hidden_dim,
                "tokens": cfg.num_tokens,
                "blocks": cfg.num_blocks,
                "eig": cfg.num_eig if cfg.family == "hybrid_jev" else "-",
                "parameters": count_parameters(model),
                "description": cfg.description,
            }
        )
    return rows


if __name__ == "__main__":  # pragma: no cover - manual inspection
    for row in setup_table():
        print(row)
    x = torch.randn(2, INPUT_DIM)
    for name in SETUP_VARIANTS:
        out = build_setup_by_name(name)(x)
        assert out.shape == (2, OUTPUT_DIM), (name, out.shape)
    print("all setups OK")
