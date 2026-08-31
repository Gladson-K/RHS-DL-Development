"""Constraint evaluator for RHS beamforming architectures.

Verifies physical and hardware constraints:
1. Unit-modulus constraint for RHS holographic elements (|M_n| = 1)
2. Transmit power constraints for combiner / beamforming weights (||w||^2 <= P_max)
"""

from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn


class ConstraintEvaluator:
    """Evaluates physical and mathematical constraint violations."""

    def __init__(self, target_power: float = 1.0, atol: float = 1e-4) -> None:
        self.target_power = target_power
        self.atol = atol

    @torch.no_grad()
    def evaluate_constraints(
        self,
        model: nn.Module,
        dataloader: Any,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """Compute constraint violation metrics over validation dataset.

        Returns
        -------
        Dict with keys:
            'unit_modulus_violation': Mean deviation from |M_n| = 1.
            'power_violation': Mean deviation from target transmit power.
            'constraint_violation': Combined total violation penalty.
        """
        model.eval()
        total_m_viol = 0.0
        total_w_viol = 0.0
        total_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0].to(device)
                config = batch[1].to(device) if len(batch) > 1 else torch.zeros(x.shape[0], 4, device=device)
                out = model(x, config)

                # Check M unit modulus violation: |real^2 + imag^2 - 1|
                if "M" in out:
                    M = out["M"]
                    dim = M.shape[-1] // 2
                    real, imag = M[..., :dim], M[..., dim:]
                    amp_sq = real.square() + imag.square()
                    m_viol = (amp_sq - 1.0).abs().mean().item()
                    total_m_viol += m_viol

                # Check w power constraint
                if "w" in out:
                    w = out["w"]
                    dim_w = w.shape[-1] // 2
                    real_w, imag_w = w[..., :dim_w], w[..., dim_w:]
                    power = (real_w.square() + imag_w.square()).sum(dim=-1)
                    w_viol = (power - self.target_power).clamp_min(0.0).mean().item()
                    total_w_viol += w_viol

            total_batches += 1

        n_b = max(1, total_batches)
        avg_m_viol = total_m_viol / n_b
        avg_w_viol = total_w_viol / n_b

        return {
            "unit_modulus_violation": round(avg_m_viol, 6),
            "power_violation": round(avg_w_viol, 6),
            "constraint_violation": round(avg_m_viol + avg_w_viol, 6),
        }
