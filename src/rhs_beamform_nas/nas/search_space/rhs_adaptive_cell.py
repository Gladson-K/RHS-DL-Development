"""RHS Adaptive Cell — NNI LayerChoice-based searchable cell.

This module wraps the RHS NAS primitives in NNI's ModelSpace / LayerChoice
API so that the search strategy (DARTS, ENAS, random) can select which
primitive is active for each cell position.

NNI requirement: NNI ≥ 3.0 with the PyTorch NAS API enabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from nni.nas.nn.pytorch import ModelSpace, LayerChoice
    _NNI_AVAILABLE = True
except ImportError:
    _NNI_AVAILABLE = False
    # Fallback: plain nn.Module that picks the first op (for testing without NNI)
    ModelSpace = nn.Module  # type: ignore[misc,assignment]

from rhs_beamform_nas.nas.search_space.primitives import (
    ComplexResidualBlock,
    ComplexLinear,
    RealImagLinear,
    ComplexNorm,
    SpatialGNNBlock,
    GraphAttentionBlock,
    GlobalAttentionBlock,
    LocalRHSBlock,
    MBConvBlock,
    RHSBottleneck,
    DepthwiseBlock,
    ResidualBlock,
)

#: Default candidate primitives for the RHS operator LayerChoice.
#: Ordered as: complex → spatial → efficient (mirrors search space taxonomy).
DEFAULT_CANDIDATES: list[str] = [
    "complex_residual",
    "complex_linear",
    "real_imag_linear",
    "spatial_gnn",
    "graph_attention",
    "global_attention",
    "local_rhs",
    "mbconv",
    "bottleneck",
    "residual",
]


def _build_candidate_list(dim: int, candidates: list[str]) -> list[nn.Module]:
    """Build candidate module list for LayerChoice."""
    from rhs_beamform_nas.nas.search_space.primitives import build_primitive
    return [build_primitive(name, dim) for name in candidates]


class RHSAdaptiveCell(ModelSpace):
    """Searchable RHS cell with LayerChoice over RHS-specific primitives.

    NNI's search strategy selects one primitive per cell position during
    architecture search. After search, the selected op is frozen.

    Parameters
    ----------
    dim:
        Feature dimension. Complex primitives use 2*dim internally.
    candidates:
        List of primitive names to include in the LayerChoice.
        Defaults to DEFAULT_CANDIDATES.
    label:
        Unique identifier for this LayerChoice. Must be distinct across
        all cells in the model space if they should be searched independently.
    """

    def __init__(
        self,
        dim: int,
        candidates: list[str] | None = None,
        label: str = "rhs_operator",
    ) -> None:
        super().__init__()
        self.dim = dim
        _candidates = candidates if candidates is not None else DEFAULT_CANDIDATES

        if _NNI_AVAILABLE:
            self.op: nn.Module = LayerChoice(
                _build_candidate_list(dim, _candidates),
                label=label,
            )
        else:
            # Fallback without NNI: use the first candidate (ComplexResidualBlock)
            self.op = ComplexResidualBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selected (or default) primitive.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, dim) or (batch, 2*dim)
            depending on the selected primitive.

        Returns
        -------
        torch.Tensor
            Processed feature tensor of same shape as input.
        """
        return self.op(x)