"""JEV / eigen-decomposition decision head.

The head makes an *eigen-based decision* on the token representation: it forms
the Gram matrix of the (normalised) tokens, eigen-decomposes it with
``torch.linalg.eigh`` (differentiable), and uses the dominant eigenvectors as a
spectral basis. The token features are projected into that basis and recombined
with learned, eigenvalue-weighted coefficients before the output projection.

This is the differentiable analogue of the classical EVD/eigenvector ("JEV")
beamformer: instead of taking a fixed dominant singular direction, the network
learns how to weight the eigen-components. The number of eigen-components
(``num_eig``) is a direct latency/accuracy knob.

Shapes: ``x: [B, T, D] -> [B, out_dim]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class JEVDecisionHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_eig: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_eig = num_eig
        self.norm = nn.LayerNorm(dim)
        # learns how to mix the top eigen-components (spectral decision)
        self.eig_mix = nn.Linear(num_eig, num_eig)
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.out = nn.Linear(dim, out_dim)
        self.last_eigenvalues: torch.Tensor | None = None

    def forward(self, x):
        h = self.norm(x)                                  # [B,T,D]
        gram = (h @ h.transpose(1, 2)) / math.sqrt(self.dim)  # [B,T,T], PSD
        evals, evecs = torch.linalg.eigh(gram)            # ascending
        k = min(self.num_eig, h.shape[1])
        evals_top = evals[..., -k:]                       # [B,k]
        vecs_top = evecs[..., -k:]                         # [B,T,k]
        self.last_eigenvalues = evals_top.detach()

        coeff = vecs_top.transpose(1, 2) @ h              # [B,k,D]
        weights = torch.softmax(evals_top, dim=-1)        # [B,k]
        weights = torch.softmax(self.eig_mix(weights), dim=-1)
        gated = (coeff * weights.unsqueeze(-1)).sum(dim=1)  # [B,D]

        base = h.mean(dim=1)                              # [B,D]
        feat = self.proj(torch.cat([base, gated], dim=-1))  # [B,D]
        return self.out(feat)                             # [B,out_dim]
