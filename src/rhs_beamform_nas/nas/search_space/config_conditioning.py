"""Config conditioning mechanisms for RHS NAS.

These modules condition the beamforming network on system configuration
parameters (Nr, Nx, Nt, frequency, SNR, users, etc.).

Conditioning strategies
-----------------------
AddConditioning          Simple additive conditioning
FiLMConditioning         Feature-wise Linear Modulation (scale + shift)
CrossAttentionConditioning  Cross-attention between features and config
ConcatConditioning       Concatenate config to features then project back
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddConditioning(nn.Module):
    """Additive conditioning: features += projection(config).

    The simplest conditioning mechanism — config is projected to the
    same dimension as features and added element-wise.

    Parameters
    ----------
    feat_dim:
        Feature vector dimension.
    config_dim:
        System configuration vector dimension.
    """

    def __init__(self, feat_dim: int, config_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(config_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        """Apply additive conditioning.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, feat_dim).
        config:
            Configuration tensor of shape (batch, config_dim).

        Returns
        -------
        torch.Tensor
            Conditioned features of shape (batch, feat_dim).
        """
        return self.norm(x + self.proj(config))


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning.

    Applies per-feature affine transform conditioned on the system
    configuration:  output = gamma(config) * x + beta(config)

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer", AAAI 2018.

    Parameters
    ----------
    feat_dim:
        Feature vector dimension.
    config_dim:
        System configuration vector dimension.
    """

    def __init__(self, feat_dim: int, config_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(config_dim, feat_dim)
        self.beta_proj = nn.Linear(config_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        # Initialise gamma near 1, beta near 0
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, feat_dim).
        config:
            Configuration tensor of shape (batch, config_dim).

        Returns
        -------
        torch.Tensor
            Conditioned features of shape (batch, feat_dim).
        """
        gamma = self.gamma_proj(config)
        beta = self.beta_proj(config)
        return self.norm(gamma * x + beta)


class CrossAttentionConditioning(nn.Module):
    """Cross-attention conditioning between features and config tokens.

    Features attend to the configuration embedding, allowing each
    feature dimension to selectively draw from relevant config factors.

    Parameters
    ----------
    feat_dim:
        Feature vector dimension (query dimension).
    config_dim:
        System configuration vector dimension (key/value dimension).
    num_heads:
        Number of attention heads.
    """

    def __init__(
        self,
        feat_dim: int,
        config_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(config_dim, feat_dim)
        self.v_proj = nn.Linear(config_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention conditioning.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, feat_dim).
        config:
            Configuration tensor of shape (batch, config_dim).

        Returns
        -------
        torch.Tensor
            Conditioned features of shape (batch, feat_dim).
        """
        # Expand to sequence dim=1
        q = self.q_proj(x).unsqueeze(1)          # (B, 1, D)
        k = self.k_proj(config).unsqueeze(1)      # (B, 1, D)
        v = self.v_proj(config).unsqueeze(1)      # (B, 1, D)

        attn_weights = F.softmax(
            torch.bmm(q, k.transpose(-2, -1)) / self.scale, dim=-1
        )  # (B, 1, 1)
        out = torch.bmm(attn_weights, v).squeeze(1)  # (B, D)
        return self.norm(x + self.out_proj(out))


class ConcatConditioning(nn.Module):
    """Concatenation-based conditioning.

    Concatenates the config vector to the feature vector then projects
    back to the original feature dimension.

    Parameters
    ----------
    feat_dim:
        Feature vector dimension.
    config_dim:
        System configuration vector dimension.
    """

    def __init__(self, feat_dim: int, config_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim + config_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        """Apply concatenation conditioning.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, feat_dim).
        config:
            Configuration tensor of shape (batch, config_dim).

        Returns
        -------
        torch.Tensor
            Conditioned features of shape (batch, feat_dim).
        """
        combined = torch.cat([x, config], dim=-1)
        return self.norm(self.proj(combined))


#: Registry of all conditioning mechanisms.
CONDITIONING_REGISTRY: dict[str, type[nn.Module]] = {
    "add": AddConditioning,
    "film": FiLMConditioning,
    "cross_attention": CrossAttentionConditioning,
    "concat": ConcatConditioning,
}


def build_conditioning(
    name: str,
    feat_dim: int,
    config_dim: int,
) -> nn.Module:
    """Instantiate a conditioning module by name.

    Parameters
    ----------
    name:
        Key from CONDITIONING_REGISTRY (e.g. 'film', 'add').
    feat_dim:
        Feature vector dimension.
    config_dim:
        Config vector dimension.

    Returns
    -------
    nn.Module
        Instantiated conditioning module.
    """
    if name not in CONDITIONING_REGISTRY:
        raise KeyError(
            f"Unknown conditioning '{name}'. "
            f"Available: {sorted(CONDITIONING_REGISTRY)}"
        )
    return CONDITIONING_REGISTRY[name](feat_dim, config_dim)
