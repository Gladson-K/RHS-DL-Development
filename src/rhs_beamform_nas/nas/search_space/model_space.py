"""RHSAdaptiveBeamformingModelSpace — the full NAS supernet.

This is the top-level searchable model that the NAS strategy optimises.
It stacks multiple RHSAdaptiveCells followed by the BeamformingOutputCell.

Search dimensions
-----------------
Operator choice:    One of 12 RHS-specific primitives per cell (LayerChoice)
Depth:              {2, 4, 6, 8} cells (ValueChoice)
Hidden dimension:   {128, 192, 256, 384} (ValueChoice)
Conditioning:       {add, film, cross_attention, concat} (ValueChoice)
M representation:   {amplitude_phase, real_imag} (ValueChoice)
w representation:   {real_imag, complex_unit} (ValueChoice)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from nni.nas.nn.pytorch import ModelSpace, LayerChoice, ValueChoice
    _NNI_AVAILABLE = True
except ImportError:
    _NNI_AVAILABLE = False
    ModelSpace = nn.Module  # type: ignore[misc,assignment]

from rhs_beamform_nas.nas.search_space.rhs_adaptive_cell import RHSAdaptiveCell
from rhs_beamform_nas.nas.search_space.beamforming_cell import BeamformingOutputCell
from rhs_beamform_nas.nas.search_space.config_conditioning import (
    build_conditioning,
    CONDITIONING_REGISTRY,
)

# Search space constants (match README-NASRecipe.md)
DEPTH_CHOICES: list[int] = [2, 4, 6, 8]
HIDDEN_DIM_CHOICES: list[int] = [128, 192, 256, 384]
CONDITIONING_CHOICES: list[str] = ["add", "film", "cross_attention", "concat"]
M_REPR_CHOICES: list[str] = ["amplitude_phase", "real_imag"]
W_REPR_CHOICES: list[str] = ["real_imag", "complex_unit"]


class RHSAdaptiveBeamformingModelSpace(ModelSpace):
    """Full RHS NAS supernet with searchable depth, width, and operators.

    The model space stacks `depth` RHSAdaptiveCells, each independently
    choosing its operator. The system configuration is injected through
    the conditioning mechanism after each cell.

    Parameters
    ----------
    input_dim:
        Dimension of the raw input features (e.g. 2*Nr for complex channel).
    config_dim:
        Dimension of the system configuration embedding.
    Nr:
        Number of RHS elements (output M dimension).
    Nt:
        Number of transmit antennas (output w dimension).
    hidden_dim:
        Hidden feature dimension. If NNI is available, this becomes a
        ValueChoice over HIDDEN_DIM_CHOICES.
    depth:
        Number of searchable cells. If NNI is available, this becomes a
        ValueChoice over DEPTH_CHOICES.
    conditioning_type:
        Config conditioning mechanism. If NNI is available, ValueChoice
        over CONDITIONING_CHOICES.
    M_representation:
        M output representation. ValueChoice over M_REPR_CHOICES.
    w_representation:
        w output representation. ValueChoice over W_REPR_CHOICES.
    predict_rate:
        Whether to include the auxiliary rate prediction head.
    """

    # Fixed defaults used when NNI is not available (testing / development)
    _DEFAULT_HIDDEN = 256
    _DEFAULT_DEPTH = 4
    _DEFAULT_COND = "film"
    _DEFAULT_M_REPR = "amplitude_phase"
    _DEFAULT_W_REPR = "real_imag"

    def __init__(
        self,
        input_dim: int,
        config_dim: int,
        Nr: int,
        Nt: int,
        *,
        hidden_dim: int | None = None,
        depth: int | None = None,
        conditioning_type: str | None = None,
        M_representation: str | None = None,
        w_representation: str | None = None,
        predict_rate: bool = True,
    ) -> None:
        super().__init__()

        # --- Search dimensions (ValueChoice when NNI available) ---
        if _NNI_AVAILABLE:
            self.hidden_dim: int = ValueChoice(
                HIDDEN_DIM_CHOICES, label="hidden_dim"
            )
            self.depth: int = ValueChoice(DEPTH_CHOICES, label="rhs_depth")
            self._cond_type: str = ValueChoice(
                CONDITIONING_CHOICES, label="conditioning"
            )
            self._M_repr: str = ValueChoice(
                M_REPR_CHOICES, label="M_representation"
            )
            self._w_repr: str = ValueChoice(
                W_REPR_CHOICES, label="w_representation"
            )
        else:
            # Development / test mode without NNI
            self.hidden_dim = hidden_dim or self._DEFAULT_HIDDEN
            self.depth = depth or self._DEFAULT_DEPTH
            self._cond_type = conditioning_type or self._DEFAULT_COND
            self._M_repr = M_representation or self._DEFAULT_M_REPR
            self._w_repr = w_representation or self._DEFAULT_W_REPR

        self.Nr = Nr
        self.Nt = Nt
        self.config_dim = config_dim

        # --- Input stem: project raw features to hidden_dim ---
        _hdim = hidden_dim or self._DEFAULT_HIDDEN
        self.input_stem = nn.Sequential(
            nn.Linear(input_dim, _hdim),
            nn.LayerNorm(_hdim),
            nn.GELU(),
        )

        # --- Config encoder: project system config to config_dim ---
        self.config_encoder = nn.Sequential(
            nn.Linear(config_dim, config_dim),
            nn.GELU(),
        )

        # --- Searchable cell stack ---
        # Build maximum-depth cell stack; runtime depth gates unused cells
        _max_depth = max(DEPTH_CHOICES)
        _ctype = conditioning_type or self._DEFAULT_COND
        self.cells = nn.ModuleList([
            RHSAdaptiveCell(
                dim=_hdim,
                label=f"rhs_operator_{i}",
            )
            for i in range(_max_depth)
        ])

        # --- Per-cell config conditioning ---
        self.conditioners = nn.ModuleList([
            build_conditioning(_ctype, _hdim, config_dim)
            for _ in range(_max_depth)
        ])

        # --- Output head ---
        _M_repr = M_representation or self._DEFAULT_M_REPR
        _w_repr = w_representation or self._DEFAULT_W_REPR
        self.output_cell = BeamformingOutputCell(
            in_dim=_hdim,
            Nr=Nr,
            Nt=Nt,
            M_representation=_M_repr,
            w_representation=_w_repr,
            predict_rate=predict_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        config: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the supernet.

        Parameters
        ----------
        x:
            Input feature tensor of shape (batch, input_dim).
        config:
            System configuration tensor of shape (batch, config_dim).

        Returns
        -------
        dict with keys 'M', 'w', and optionally 'rate'.
        """
        # Encode inputs
        h = self.input_stem(x)
        cfg = self.config_encoder(config)

        # Pass through active cells (runtime depth selection)
        _depth = self.depth if isinstance(self.depth, int) else self._DEFAULT_DEPTH
        for i in range(_depth):
            h = self.cells[i](h)
            h = self.conditioners[i](h, cfg)

        # Beamforming output
        return self.output_cell(h)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> dict:
        """Return a summary of the current (fixed/sampled) architecture."""
        return {
            "hidden_dim": self.hidden_dim,
            "rhs_depth": self.depth,
            "conditioning": self._cond_type,
            "M_representation": self._M_repr,
            "w_representation": self._w_repr,
            "Nr": self.Nr,
            "Nt": self.Nt,
            "parameters": self.count_parameters(),
        }
