"""RHS Cell — three-path architecture cell for NAS.

The RHSCell implements the fundamental building block of the NAS
architecture. Each cell processes features through three parallel paths:

    RHS CELL
        │
    ┌───┼───────────────────┐
    │   │                   │
    ▼   ▼                   ▼
Complex  Spatial           Config
 path    path              path
    │       │                │
    ▼       ▼                ▼
Complex   GNN /         FiLM / Cross
  op     Attention       Attention
    │       │                │
    └───┬───┘                │
        ▼                    │
     Fusion  ←───────────────┘
        │
        ▼
     Residual

This design allows NAS to discover whether a particular layer benefits
more from complex-domain processing, spatial GNN/attention, or
configuration conditioning — rather than simply choosing generic
convolution sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rhs_beamform_nas.nas.search_space.primitives import (
    ComplexResidualBlock,
    SpatialGNNBlock,
    GraphAttentionBlock,
    RHSPrimitive,
)
from rhs_beamform_nas.nas.search_space.config_conditioning import (
    FiLMConditioning,
    AddConditioning,
)


class RHSCell(nn.Module):
    """Three-path RHS processing cell.

    Combines complex-domain processing, spatial graph processing, and
    system configuration conditioning into a single composable cell.

    Parameters
    ----------
    feat_dim:
        Real-valued feature dimension. Complex processing uses 2*feat_dim.
    config_dim:
        System configuration embedding dimension.
    complex_op:
        Complex-path operation module. Defaults to ComplexResidualBlock.
    spatial_op:
        Spatial-path operation module. Defaults to SpatialGNNBlock.
    conditioning:
        Config-path conditioning module. Defaults to FiLMConditioning.
    """

    def __init__(
        self,
        feat_dim: int,
        config_dim: int,
        complex_op: nn.Module | None = None,
        spatial_op: nn.Module | None = None,
        conditioning: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.config_dim = config_dim

        # --- Complex path ---
        # Works on 2*feat_dim (real+imag concatenated)
        self.complex_op: nn.Module = (
            complex_op if complex_op is not None else ComplexResidualBlock(feat_dim)
        )
        # Project complex features back to feat_dim for fusion
        self.complex_proj = nn.Linear(2 * feat_dim, feat_dim)

        # --- Spatial path ---
        self.spatial_op: nn.Module = (
            spatial_op if spatial_op is not None else SpatialGNNBlock(feat_dim)
        )

        # --- Config conditioning path ---
        self.conditioning: nn.Module = (
            conditioning if conditioning is not None
            else FiLMConditioning(feat_dim, config_dim)
        )

        # --- Fusion ---
        # Merge: complex_path(feat_dim) + spatial_path(feat_dim) + cond_path(feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * feat_dim, feat_dim),
            nn.GELU(),
        )
        self.residual_norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        x: torch.Tensor,
        config: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through three-path cell.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, feat_dim).
        config:
            System configuration embedding of shape (batch, config_dim).

        Returns
        -------
        torch.Tensor
            Output features of shape (batch, feat_dim).
        """
        # Complex path: duplicate to real+imag representation
        x_complex = torch.cat([x, x], dim=-1)          # (B, 2*D) — use x as real, zeros as imag start
        complex_out = self.complex_proj(self.complex_op(x_complex))  # (B, D)

        # Spatial path
        spatial_out = self.spatial_op(x)                # (B, D)

        # Config conditioning path
        cond_out = self.conditioning(x, config)          # (B, D)

        # Fusion: concatenate all three paths
        merged = torch.cat([complex_out, spatial_out, cond_out], dim=-1)  # (B, 3D)
        fused = self.fusion(merged)                      # (B, D)

        # Residual connection
        return self.residual_norm(x + fused)
