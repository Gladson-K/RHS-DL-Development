"""Graph message-passing block (GNN) over a small set of learned tokens.

Used as the "graph" path of the HaloBeam-GNN hybrid. A learned, row-softmaxed
adjacency mixes messages between tokens; a position-wise FFN follows.

Shapes: ``x: [B, T, D] -> [B, T, D]``.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphMessagePassingBlock(nn.Module):
    def __init__(self, dim: int, num_tokens: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.adj = nn.Parameter(torch.zeros(num_tokens, num_tokens))
        nn.init.xavier_uniform_(self.adj)
        self.msg = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        adj = torch.softmax(self.adj, dim=-1)               # [T,T]
        messages = self.msg(x)                              # [B,T,D]
        agg = torch.einsum("ts,bsd->btd", adj, messages)    # [B,T,D]
        x = x + self.drop(self.norm(agg))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
