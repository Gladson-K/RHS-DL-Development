"""Runtime configuration for the inference container (env driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Paths and knobs read from the environment."""

    repo_src: str = os.getenv("REPO_SRC", "/repo/src")
    hybrid_checkpoint: str = os.getenv(
        "HYBRID_CHECKPOINT", "/repo/search_results/mimo64_full_data/checkpoints/hybrid_jev.pt"
    )
    nas_checkpoint: str = os.getenv("NAS_CHECKPOINT", "/repo/checkpoints/mimo64_nas_best.pt")
    scaler: str = os.getenv("SCALER", "/repo/data/processed/mimo64_full/scaler.json")
    meta: str = os.getenv("META", "/repo/data/processed/mimo64_full/meta.json")
    fixture: str = os.getenv("FIXTURE", "/app/fixtures/static_samples.npz")
    rate_mean: float = _env_float("RATE_MEAN", 51.794)
    rate_std: float = _env_float("RATE_STD", 12.298)
    jev_k: int = _env_int("JEV_K", 4)
    latency_weight: float = _env_float("JEV_LATENCY_WEIGHT", 0.05)
    spectral_weight: float = _env_float("JEV_SPECTRAL_WEIGHT", 0.2)
    strict: bool = os.getenv("INFERENCE_STRICT", "0") == "1"


SETTINGS = Settings()
