"""Beamforming-vector comparison metrics (GP ground truth vs DL output)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

M_SHAPE = (20, 64, 10)


def to_complex(real: Any, imag: Any) -> np.ndarray:
    """Combine real/imag nested (or flat) lists into a complex matrix."""
    re = np.asarray(real, dtype=np.float64)
    im = np.asarray(imag, dtype=np.float64)
    if re.shape != im.shape:
        raise ValueError(f"real/imag shape mismatch: {re.shape} vs {im.shape}")
    flat = re.ravel() + 1j * im.ravel()
    if flat.size == int(np.prod(M_SHAPE)):
        return flat.reshape(M_SHAPE)
    return flat


def matrix_hash(real: Any, imag: Any, digits: int = 6) -> str:
    """Stable hash of a rounded complex matrix."""
    matrix = to_complex(real, imag)
    rounded = np.round(matrix, digits)
    payload = rounded.real.astype(np.float32).tobytes() + rounded.imag.astype(np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()


def beamforming_metrics(
    pred_real: Any, pred_imag: Any, gp_real: Any, gp_imag: Any
) -> dict[str, float]:
    """Return cosine similarity, normalised MSE and raw MSE between two matrices."""
    pred = to_complex(pred_real, pred_imag)
    target = to_complex(gp_real, gp_imag)
    diff = pred - target
    se = float(np.sum(np.abs(diff) ** 2))
    energy = float(np.sum(np.abs(target) ** 2))
    dot = float(np.real(np.sum(pred * np.conj(target))))
    norm_p = float(np.linalg.norm(pred))
    norm_t = float(np.linalg.norm(target))
    cosine = dot / (norm_p * norm_t) if norm_p and norm_t else 0.0
    return {
        "m_cos": round(cosine, 6),
        "m_nmse": round(se / energy if energy else 0.0, 6),
        "m_mse": round(se / pred.size, 9),
        "m_norm_pred": round(norm_p, 6),
        "m_norm_gp": round(norm_t, 6),
    }


def diff_report(
    pred_real: Any,
    pred_imag: Any,
    gp_real: Any,
    gp_imag: Any,
    dl_rate: float,
    gp_rate: float,
) -> dict[str, Any]:
    """Full diff block stored with every published sample."""
    metrics = beamforming_metrics(pred_real, pred_imag, gp_real, gp_imag)
    return {
        "rate_delta": round(float(dl_rate) - float(gp_rate), 6),
        "rate_abs_delta": round(abs(float(dl_rate) - float(gp_rate)), 6),
        "dl_rate": round(float(dl_rate), 6),
        "gp_rate": round(float(gp_rate), 6),
        **metrics,
        "dl_hash": matrix_hash(pred_real, pred_imag),
        "gp_hash": matrix_hash(gp_real, gp_imag),
    }
