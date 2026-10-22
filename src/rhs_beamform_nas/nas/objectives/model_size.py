"""Model size and parameter count objective for RHS NAS.

Penalizes excessive parameter count (in millions of parameters).
"""

from __future__ import annotations

from typing import Dict, Any


class ModelSizeObjective:
    """Evaluates parameter footprint penalty in millions of parameters."""

    def __init__(self, weight: float = 0.05, max_params_million: float = 10.0) -> None:
        self.weight = weight
        self.max_params_million = max_params_million

    def __call__(self, metrics: Dict[str, Any]) -> float:
        """Compute parameter penalty."""
        params_million = metrics.get("parameters_million", 0.0)
        if params_million == 0.0 and "parameters" in metrics:
            params_million = metrics["parameters"] / 1e6
        return float(params_million) * self.weight
