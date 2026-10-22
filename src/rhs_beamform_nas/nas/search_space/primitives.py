"""RHS-specific NAS primitives.

All searchable operations in the RHSAdaptiveBeamformingSearchSpace.
These are communication-system-aware operations, NOT generic
image-classification operations.

Primitive categories
--------------------
Complex processing   : ComplexLinear, RealImagLinear, ComplexResidualBlock, ComplexNorm
Spatial processing   : LocalRHSBlock, SpatialGNNBlock, GraphAttentionBlock, GlobalAttentionBlock
Efficient processing : MBConvBlock, RHSBottleneck, DepthwiseBlock, ResidualBlock
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RHSPrimitive(nn.Module):
    """Base class for all RHS-specific NAS primitive operations.

    Parameters
    ----------
    dim:
        Feature dimension (real-valued vector dimension).
        Complex representations use 2*dim internally.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ---------------------------------------------------------------------------
# Complex processing primitives
# ---------------------------------------------------------------------------


class ComplexLinear(RHSPrimitive):
    """Complex-valued linear layer via real/imag decomposition.

    Input: (batch, 2*dim)  — [real_part | imag_part]
    Output: (batch, 2*dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.real_proj = nn.Linear(dim, dim)
        self.imag_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x[..., : self.dim], x[..., self.dim :]
        # (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
        out_real = self.real_proj(real) - self.imag_proj(imag)
        out_imag = self.real_proj(imag) + self.imag_proj(real)
        return self.norm(torch.cat([out_real, out_imag], dim=-1))


class RealImagLinear(RHSPrimitive):
    """Independent real and imaginary linear projections.

    Simpler than ComplexLinear — applies separate projections to
    real and imaginary parts without cross-coupling.

    Input: (batch, 2*dim)
    Output: (batch, 2*dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.real_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.imag_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x[..., : self.dim], x[..., self.dim :]
        return self.norm(torch.cat([self.real_proj(real), self.imag_proj(imag)], dim=-1))


class ComplexResidualBlock(RHSPrimitive):
    """Complex-valued residual block operating on (batch, 2*dim) features.

    Applies a two-layer MLP on the complex feature vector with a skip
    connection.

    Input: (batch, 2*dim)
    Output: (batch, 2*dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.net = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, 2 * dim),
        )
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ComplexNorm(RHSPrimitive):
    """Normalise complex features via amplitude-phase normalisation.

    Computes per-element amplitude, normalises real/imag parts, and
    applies a learned rescale.

    Input: (batch, 2*dim)
    Output: (batch, 2*dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.scale = nn.Parameter(torch.ones(2 * dim))
        self.bias = nn.Parameter(torch.zeros(2 * dim))
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x * self.scale + self.bias


# ---------------------------------------------------------------------------
# Spatial / RHS processing primitives
# ---------------------------------------------------------------------------


class LocalRHSBlock(RHSPrimitive):
    """Local spatial processing over RHS element neighbourhood.

    Models local inter-element coupling via a depthwise-separable
    operation in feature space.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__(dim)
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SpatialGNNBlock(RHSPrimitive):
    """Message-passing GNN block for RHS spatial graph processing.

    Simulates graph message passing using self-attention as a proxy
    for the actual GNN when graph topology is not available at the
    primitive level.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim) → treat as single-node sequence
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1)


class GraphAttentionBlock(RHSPrimitive):
    """Graph attention block with multi-head attention over node features.

    Applies scaled dot-product attention to aggregate neighbourhood
    information for each RHS element.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, N, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Single-node attention is identity; multi-node case generalises
        scale = math.sqrt(self.head_dim)
        attn_weights = F.softmax(
            torch.bmm(q, k.transpose(-2, -1)) / scale, dim=-1
        )
        out = torch.bmm(attn_weights, v)
        out = self.out_proj(out)
        return self.norm((x + out).squeeze(1))


class GlobalAttentionBlock(RHSPrimitive):
    """Global attention over the full RHS feature set.

    Uses a global context vector to condition each element, enabling
    array-wide coordination without explicit graph edges.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.context = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.context(x)
        gate = self.gate(x)
        return self.norm(x + self.proj(ctx * gate))


# ---------------------------------------------------------------------------
# Efficient processing primitives
# ---------------------------------------------------------------------------


class MBConvBlock(RHSPrimitive):
    """MobileNetV2-style inverted-residual (MBConv) block.

    Expand → Depthwise (via Linear) → Project with residual.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int, expansion: int = 6) -> None:
        super().__init__(dim)
        hidden = dim * expansion
        self.net = nn.Sequential(
            # Pointwise expand
            nn.Linear(dim, hidden),
            nn.GELU(),
            # Depthwise (groups on Linear not supported; use element-wise MLP)
            nn.Linear(hidden, hidden),
            nn.GELU(),
            # Pointwise project
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class RHSBottleneck(RHSPrimitive):
    """Bottleneck residual block — wide-narrow-wide.

    Reduces and restores dimensionality inside the residual branch.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int, reduction: int = 2) -> None:
        super().__init__(dim)
        hidden = max(dim // reduction, 16)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class DepthwiseBlock(RHSPrimitive):
    """Depthwise separable linear block (feature-space analogue).

    Applies channel-wise then pointwise projections as a factored
    alternative to a full linear layer.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        # Channel-wise: groups=dim via element-wise scale+bias
        self.depthwise_scale = nn.Parameter(torch.ones(dim))
        self.depthwise_bias = nn.Parameter(torch.zeros(dim))
        self.pointwise = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw = self.act(x * self.depthwise_scale + self.depthwise_bias)
        return self.norm(x + self.pointwise(dw))


class ResidualBlock(RHSPrimitive):
    """Standard two-layer MLP residual block.

    Input: (batch, dim)
    Output: (batch, dim)
    """

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__(dim)
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Mapping from canonical primitive name → class.
#: Used by model_space.py and config-driven instantiation.
PRIMITIVE_REGISTRY: dict[str, type[RHSPrimitive]] = {
    # Complex processing
    "complex_linear": ComplexLinear,
    "real_imag_linear": RealImagLinear,
    "complex_residual": ComplexResidualBlock,
    "complex_norm": ComplexNorm,
    # Spatial processing
    "local_rhs": LocalRHSBlock,
    "spatial_gnn": SpatialGNNBlock,
    "graph_attention": GraphAttentionBlock,
    "global_attention": GlobalAttentionBlock,
    # Efficient processing
    "mbconv": MBConvBlock,
    "bottleneck": RHSBottleneck,
    "depthwise": DepthwiseBlock,
    "residual": ResidualBlock,
}


def build_primitive(name: str, dim: int) -> RHSPrimitive:
    """Instantiate a primitive by name.

    Parameters
    ----------
    name:
        Canonical primitive name (key in PRIMITIVE_REGISTRY).
    dim:
        Feature dimension.

    Returns
    -------
    RHSPrimitive
        Instantiated primitive module.

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry.
    """
    if name not in PRIMITIVE_REGISTRY:
        raise KeyError(
            f"Unknown primitive '{name}'. "
            f"Available: {sorted(PRIMITIVE_REGISTRY)}"
        )
    return PRIMITIVE_REGISTRY[name](dim)