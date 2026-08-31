"""Combined multi-objective NAS scoring function.

Implements the formal RHS NAS objective function specified in README-NASRecipe.md:

    score = (
        alpha * normalized_rate
        + beta * beamforming_similarity
        - gamma * latency_ms
        - delta * parameter_million
        - epsilon * constraint_violation
    )
"""

from __future__ import annotations

from typing import Any, Dict


class RHSNASObjective:
    """Scalarized multi-objective evaluation metric for RHS architecture search."""

    def __init__(
        self,
        alpha: float = 1.0,    # Rate weight
        beta: float = 0.5,     # Similarity / GP alignment weight
        gamma: float = 0.05,   # Latency penalty weight (ms)
        delta: float = 0.02,   # Parameter count penalty weight (millions)
        epsilon: float = 1.0,  # Constraint violation penalty weight
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def compute_score(self, metrics: Dict[str, Any]) -> float:
        """Compute aggregate scalar score from metrics dictionary.

        Parameters
        ----------
        metrics:
            Dict containing normalized_rate, gp_similarity, latency_ms,
            parameters_million (or parameters), and constraint_violation.

        Returns
        -------
        float: Scalar fitness score (higher is better).
        """
        rate = float(metrics.get("normalized_rate", metrics.get("gp_rate_ratio", 0.0)))
        similarity = float(metrics.get("beamforming_similarity", rate))
        latency = float(metrics.get("latency_ms", 0.0))

        params_m = metrics.get("parameters_million")
        if params_m is None:
            params_m = float(metrics.get("parameters", 0.0)) / 1e6
        else:
            params_m = float(params_m)

        constraint = float(metrics.get("constraint_violation", 0.0))

        score = (
            self.alpha * rate
            + self.beta * similarity
            - self.gamma * latency
            - self.delta * params_m
            - self.epsilon * constraint
        )

        return round(score, 6)

    def __call__(self, metrics: Dict[str, Any]) -> float:
        return self.compute_score(metrics)
