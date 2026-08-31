"""Achievable rate objective for RHS NAS.

Maximizes spectral efficiency (bits/s/Hz) or normalized rate relative to GP.
"""

from __future__ import annotations

from typing import Dict, Any


class RateObjective:
    """Evaluates normalized achievable rate score."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(self, metrics: Dict[str, Any]) -> float:
        """Extract and score normalized rate."""
        rate = metrics.get("normalized_rate", metrics.get("sum_rate", 0.0))
        return float(rate) * self.weight
