"""Latency penalty objective for RHS NAS.

Penalizes inference latency to discover hardware-friendly and edge-deployable models.
"""

from __future__ import annotations

from typing import Dict, Any


class LatencyObjective:
    """Evaluates inference latency penalty in milliseconds."""

    def __init__(self, weight: float = 0.1, max_latency_ms: float = 10.0) -> None:
        self.weight = weight
        self.max_latency_ms = max_latency_ms

    def __call__(self, metrics: Dict[str, Any]) -> float:
        """Compute latency penalty term (negative or scaled)."""
        latency = metrics.get("latency_ms", 0.0)
        return float(latency) * self.weight
