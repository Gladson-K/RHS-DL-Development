"""Accuracy and similarity objective for RHS NAS.

Measures beamforming prediction alignment / cosine similarity / MSE relative to GP teacher.
"""

from __future__ import annotations

from typing import Dict
import torch


class AccuracyObjective:
    """Evaluates prediction fidelity with respect to the GP reference teacher."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(self, pred_M: torch.Tensor, target_M: torch.Tensor) -> float:
        """Compute cosine similarity or 1 - normalized MSE."""
        mse = torch.nn.functional.mse_loss(pred_M, target_M).item()
        similarity = max(0.0, 1.0 - mse)
        return similarity * self.weight
