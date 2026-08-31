"""Beamforming output cell.

Converts internal feature representations into the RHS beamforming
output tensors M (excitation / holographic weight) and w (combiner).

Output representations
----------------------
M (RHS excitation / holographic weight)
    - amplitude_phase: (Nr,) amplitude + (Nr,) phase → constrained to unit modulus
    - real_imag:       (2*Nr,) real and imaginary parts → normalised to unit modulus

w (combiner weight)
    - real_imag:       (2*Nt,) real and imaginary parts
    - complex_unit:    unit-norm combiner
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MOutputHead(nn.Module):
    """Output head for RHS excitation weight M.

    Predicts the holographic beamforming weight M with the chosen
    representation. Always normalises to satisfy the unit-modulus
    constraint |M_n| = 1 for all elements n.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    Nr:
        Number of RHS elements (output dimension of M).
    representation:
        'amplitude_phase' (default) or 'real_imag'.
    """

    def __init__(
        self,
        in_dim: int,
        Nr: int,
        representation: str = "amplitude_phase",
    ) -> None:
        super().__init__()
        self.Nr = Nr
        self.representation = representation

        if representation == "amplitude_phase":
            # Predict phase angles in [-π, π]; amplitude is 1 (unit modulus)
            self.proj = nn.Linear(in_dim, Nr)
        elif representation == "real_imag":
            self.proj = nn.Linear(in_dim, 2 * Nr)
        else:
            raise ValueError(
                f"Unknown M representation '{representation}'. "
                "Choose 'amplitude_phase' or 'real_imag'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict M and enforce unit-modulus constraint.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, in_dim).

        Returns
        -------
        torch.Tensor
            M tensor of shape (batch, 2*Nr) in real/imag format,
            normalised to unit modulus.
        """
        out = self.proj(x)

        if self.representation == "amplitude_phase":
            # Phase angles → unit-modulus complex: M = exp(j*theta)
            theta = torch.pi * torch.tanh(out)   # (batch, Nr) ∈ [-π, π]
            real = torch.cos(theta)
            imag = torch.sin(theta)
            return torch.cat([real, imag], dim=-1)  # (batch, 2*Nr)
        else:
            # Normalise each complex element to unit modulus
            real, imag = out[..., : self.Nr], out[..., self.Nr :]
            amp = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
            return torch.cat([real / amp, imag / amp], dim=-1)


class WOutputHead(nn.Module):
    """Output head for combiner weight w.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    Nt:
        Number of transmit antennas (output dimension of w).
    representation:
        'real_imag' (default) or 'complex_unit'.
    """

    def __init__(
        self,
        in_dim: int,
        Nt: int,
        representation: str = "real_imag",
    ) -> None:
        super().__init__()
        self.Nt = Nt
        self.representation = representation
        self.proj = nn.Linear(in_dim, 2 * Nt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict w.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, in_dim).

        Returns
        -------
        torch.Tensor
            w tensor of shape (batch, 2*Nt).
        """
        out = self.proj(x)

        if self.representation == "complex_unit":
            real, imag = out[..., : self.Nt], out[..., self.Nt :]
            amp = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
            return torch.cat([real / amp, imag / amp], dim=-1)

        # Default: raw real/imag projection
        return out


class RateOutputHead(nn.Module):
    """Auxiliary output head predicting achievable rate (scalar).

    Used for rate-aware training and multi-objective NAS scoring.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
            nn.Softplus(),   # rate is non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict achievable rate.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, in_dim).

        Returns
        -------
        torch.Tensor
            Rate prediction of shape (batch, 1).
        """
        return self.proj(x)


class BeamformingOutputCell(nn.Module):
    """Complete beamforming output cell producing M, w, and optionally rate.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    Nr:
        Number of RHS elements.
    Nt:
        Number of transmit antennas.
    M_representation:
        'amplitude_phase' or 'real_imag'.
    w_representation:
        'real_imag' or 'complex_unit'.
    predict_rate:
        If True, also predict achievable rate (auxiliary head).
    """

    def __init__(
        self,
        in_dim: int,
        Nr: int,
        Nt: int,
        M_representation: str = "amplitude_phase",
        w_representation: str = "real_imag",
        predict_rate: bool = True,
    ) -> None:
        super().__init__()
        self.M_head = MOutputHead(in_dim, Nr, M_representation)
        self.w_head = WOutputHead(in_dim, Nt, w_representation)
        self.rate_head = RateOutputHead(in_dim) if predict_rate else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Produce beamforming outputs.

        Parameters
        ----------
        x:
            Feature tensor of shape (batch, in_dim).

        Returns
        -------
        dict with keys:
            'M':    (batch, 2*Nr)  — RHS excitation weight
            'w':    (batch, 2*Nt)  — combiner weight
            'rate': (batch, 1)     — predicted rate (if predict_rate=True)
        """
        out: dict[str, torch.Tensor] = {
            "M": self.M_head(x),
            "w": self.w_head(x),
        }
        if self.rate_head is not None:
            out["rate"] = self.rate_head(x)
        return out
