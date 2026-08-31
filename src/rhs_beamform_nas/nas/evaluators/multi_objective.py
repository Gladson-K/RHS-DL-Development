"""Multi-objective evaluator for RHS NAS architectures.

Combines achievable rate, inference latency, parameter count, and
physical constraint compliance into unified metric tracking.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from rhs_beamform_nas.nas.evaluators.rate_evaluator import RateEvaluator
from rhs_beamform_nas.nas.evaluators.latency_evaluator import LatencyEvaluator
from rhs_beamform_nas.nas.evaluators.constraint_evaluator import ConstraintEvaluator


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters of a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BeamformingEvaluator:
    """Multi-objective evaluator orchestrating full architecture assessment."""

    def __init__(
        self,
        noise_var: float = 1e-2,
        device: str = "cpu",
        num_latency_runs: int = 50,
    ) -> None:
        self.device = device
        self.rate_evaluator = RateEvaluator(noise_var=noise_var, device=device)
        self.latency_evaluator = LatencyEvaluator(num_runs=num_latency_runs, device=device)
        self.constraint_evaluator = ConstraintEvaluator()

    def evaluate(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
        dummy_input: Optional[torch.Tensor] = None,
        dummy_config: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive multi-objective evaluation on a candidate model."""
        model.eval()

        # 1. Parameter count
        params = count_parameters(model)
        params_million = round(params / 1e6, 4)

        # 2. Latency
        latency_info = self.latency_evaluator.benchmark_latency(
            model=model,
            dummy_input=dummy_input,
            dummy_config=dummy_config,
        )

        # 3. Rate & constraint evaluation
        if dataloader is not None:
            rate_info = self.rate_evaluator.evaluate_rate(model, dataloader)
            constraint_info = self.constraint_evaluator.evaluate_constraints(
                model, dataloader, device=self.device
            )
        else:
            rate_info = {
                "sum_rate": 1.0,
                "normalized_rate": 1.0,
                "gp_rate_ratio": 1.0,
            }
            constraint_info = {
                "unit_modulus_violation": 0.0,
                "power_violation": 0.0,
                "constraint_violation": 0.0,
            }

        return {
            "normalized_rate": rate_info["normalized_rate"],
            "gp_rate_ratio": rate_info["gp_rate_ratio"],
            "latency_ms": latency_info["latency_ms"],
            "parameters": params,
            "parameters_million": params_million,
            "constraint_violation": constraint_info["constraint_violation"],
            "throughput_fps": latency_info["throughput_fps"],
        }