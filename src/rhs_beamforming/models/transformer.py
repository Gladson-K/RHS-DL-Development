"""Transformer encoder block for the real 64x64 RHS beamforming task.

Lightweight, CPU-friendly pre-norm Transformer used as an attachable backbone
over a small set of learned tokens. Kept deliberately simple (batch-first
``nn.MultiheadAttention`` + position-wise FFN) so it can be dropped into the
HaloBeam-GNN hybrid without changing the I/O contract ``x -> [B, 25601]``.
"""

from __future__ import annotations

import torch
from torch import nn


class TokenStem(nn.Module):
    """Project a config vector to ``num_tokens`` learned tokens.

    ``x: [B, input_dim] -> [B, num_tokens, hidden_dim]``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.tokens = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.act(self.proj(x))).unsqueeze(1)  # [B,1,D]
        return self.norm(h + self.tokens)                    # [B,T,D]


class MultiHeadSelfAttention(nn.Module):
    """Thin wrapper around batch-first multi-head self-attention."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return x + a


class TransformerEncoderBlock(nn.Module):
    """Pre-norm self-attention + position-wise FFN with residuals.

    ``x: [B, T, D] -> [B, T, D]``.
    """

    def __init__(self, dim: int, heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.ffn(self.norm(x)))
        return x
