"""Fixed instantiated model built from an exported NAS architecture descriptor.

Creates a clean, non-supernet PyTorch model for full retraining and deployment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from rhs_beamform_nas.nas.export.architecture import ArchitectureDescriptor
from rhs_beamform_nas.nas.search_space.primitives import build_primitive
from rhs_beamform_nas.nas.search_space.config_conditioning import build_conditioning
from rhs_beamform_nas.nas.search_space.beamforming_cell import BeamformingOutputCell


class FixedRHSModel(nn.Module):
    """Instantiated fixed neural network derived from a NAS-discovered architecture.

    Contains no search parameters or dynamic branch selection, making it
    suitable for production inference, latency benchmarking, and standalone retraining.
    """

    def __init__(
        self,
        descriptor: ArchitectureDescriptor | Dict[str, Any],
        predict_rate: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(descriptor, dict):
            self.desc = ArchitectureDescriptor.from_dict(descriptor)
        else:
            self.desc = descriptor

        hdim = self.desc.hidden_dim
        cdim = self.desc.config_dim

        # Input stem
        self.input_stem = nn.Sequential(
            nn.Linear(self.desc.input_dim, hdim),
            nn.LayerNorm(hdim),
            nn.GELU(),
        )

        # Config encoder
        self.config_encoder = nn.Sequential(
            nn.Linear(cdim, cdim),
            nn.GELU(),
        )

        # Cell operators
        self.cells = nn.ModuleList([
            build_primitive(op_name, hdim)
            for op_name in self.desc.spatial_operator
        ])

        # Conditioning modules
        self.conditioners = nn.ModuleList([
            build_conditioning(self.desc.conditioning, hdim, cdim)
            for _ in range(len(self.desc.spatial_operator))
        ])

        # Output head
        self.output_cell = BeamformingOutputCell(
            in_dim=hdim,
            Nr=self.desc.Nr,
            Nt=self.desc.Nt,
            M_representation=self.desc.M_representation,
            w_representation=self.desc.w_representation,
            predict_rate=predict_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        config: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for fixed model."""
        if config is None:
            config = torch.zeros(x.shape[0], self.desc.config_dim, device=x.device)

        h = self.input_stem(x)
        cfg = self.config_encoder(config)

        for cell, cond in zip(self.cells, self.conditioners):
            h = cell(h)
            h = cond(h, cfg)

        return self.output_cell(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
