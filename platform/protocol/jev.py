"""JEV (joint eigen-value) decision scoring.

The production model uses ``torch.linalg.eigh`` inside
``rhs_beamforming.models.jev_head``.  This module implements the small,
dependency-light scoring layer used by the platform services, the gateway and
the unit tests: a probe is reduced to a top-k spectral signature and compared
with the calibration centroid of every registered model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    """Calibration profile for one registry entry."""

    name: str
    centroid: tuple[float, ...]
    latency_ms: float = 0.0
    kind: str = "dl"
    error: float = 0.0
    meta: Mapping[str, float] = field(default_factory=dict)


def softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a non-empty sequence."""
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def spectral_signature(
    eigenvalues: Iterable[float], k: int = 4, temperature: float = 0.25
) -> list[float]:
    """Reduce a spectrum to a ``k``-bin softmax signature.

    ``eigenvalues`` may be in any order; the largest ``k`` are selected.  Spectra
    from beamforming matrices span several orders of magnitude, so values are
    log-compressed and normalised by the peak before the temperature-scaled
    softmax.  This keeps the signature discriminative instead of saturating to
    a one-hot vector.  Missing bins are padded with the smallest observed value
    so the output always has length ``k``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    values = sorted((max(float(v), 0.0) for v in eigenvalues), reverse=True)[:k]
    if not values:
        return [1.0 / k] * k
    if len(values) < k:
        values.extend([values[-1]] * (k - len(values)))
    logs = [math.log1p(value) for value in values]
    peak = max(logs)
    if peak <= 1e-12:
        return [1.0 / k] * k
    scaled = [value / peak for value in logs]
    return softmax([value / temperature for value in scaled])


def signature_from_gram(gram: Sequence[Sequence[complex]], k: int = 4) -> list[float]:
    """Compute the top-k signature of a Hermitian Gram matrix.

    Requires ``numpy`` (available in every service image).  The matrix is
    symmetrised to absorb tiny numerical asymmetries.
    """
    import numpy as np

    matrix = np.asarray(gram, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a square matrix")
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hermitian).real
    return spectral_signature(eigenvalues, k=k)


def l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """L1 distance between two equal-length signatures."""
    if len(left) != len(right):
        raise ValueError("signatures must have equal length")
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right))


def decide(
    probe: Sequence[float],
    profiles: Mapping[str, ModelProfile | Mapping[str, object]],
    latency_weight: float = 0.15,
    spectral_weight: float = 1.0,
) -> dict[str, object]:
    """Choose the closest calibrated model for ``probe``.

    The score is ``spectral_weight * L1(probe, centroid) + profile.error +
    latency_weight * normalised_latency``; the chosen model minimises it.
    ``error`` lets the caller add validation-error terms (beamforming cosine,
    rate error) when ground truth is available.  ``confidence`` is the softmax
    probability of the winning model over the negative scores.
    """
    if not profiles:
        raise ValueError("at least one model profile is required")
    entries: list[ModelProfile] = []
    for name, raw in profiles.items():
        if isinstance(raw, ModelProfile):
            entries.append(raw)
        else:
            entries.append(
                ModelProfile(
                    name=str(name),
                    centroid=tuple(float(v) for v in raw.get("centroid", ())),  # type: ignore[union-attr]
                    latency_ms=float(raw.get("latency_ms", 0.0)),  # type: ignore[union-attr]
                    kind=str(raw.get("kind", "dl")),  # type: ignore[union-attr]
                    error=float(raw.get("error", 0.0)),  # type: ignore[union-attr]
                )
            )
    max_latency = max((profile.latency_ms for profile in entries), default=0.0) or 1.0
    scores: dict[str, float] = {}
    for profile in entries:
        distance = spectral_weight * l1_distance(probe, profile.centroid)
        penalty = latency_weight * (profile.latency_ms / max_latency)
        scores[profile.name] = distance + profile.error + penalty
    winner = min(scores, key=scores.get)
    probabilities = softmax([-value for value in scores.values()])
    ordered = sorted(scores, key=scores.get)
    confidence = dict(zip(ordered, probabilities))[winner]
    return {
        "chosen_model": winner,
        "confidence": float(confidence),
        "scores": scores,
        "probe": [float(v) for v in probe],
        "latency_weight": float(latency_weight),
        "spectral_weight": float(spectral_weight),
    }
