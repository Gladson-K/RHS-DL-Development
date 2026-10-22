"""Multi-objective scoring for the real 64x64 NAS search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreWeights:
    alpha_rate: float = 1.0          # maximize rate R^2 / GP rate ratio
    beta_similarity: float = 0.5     # maximize M similarity to GP target
    gamma_latency: float = 0.05      # minimize latency (ms)
    delta_params: float = 0.02       # minimize params (millions)
    epsilon_constraint: float = 0.0  # minimize constraint violation


def compute_score(metrics: dict, w: ScoreWeights | None = None) -> float:
    """Combine rate, beamforming similarity, latency and size into one scalar.

    ``rate`` term uses the rate R^2 (clipped to [-1, 1]) mapped to [0, 1].
    Similarity uses ``1 - n_mse`` of the predicted M (clipped to [0, 1]).
    """
    if w is None:
        w = ScoreWeights()
    rate_r2 = float(metrics.get("rate", {}).get("r2", 0.0))
    rate_term = max(0.0, min(1.0, rate_r2))
    m_metrics = metrics.get("M", {})
    n_mse = float(m_metrics.get("n_mse", 1.0))
    cos = float(m_metrics.get("cos", 0.0))
    similarity = 0.5 * max(0.0, 1.0 - n_mse) + 0.5 * max(0.0, min(1.0, cos))
    latency = float(metrics.get("latency_ms", 0.0))
    params_m = float(metrics.get("parameters", 0)) / 1e6
    violation = float(metrics.get("constraint_violation", 0.0))
    return (
        w.alpha_rate * rate_term
        + w.beta_similarity * similarity
        - w.gamma_latency * latency
        - w.delta_params * params_m
        - w.epsilon_constraint * violation
    )
