"""Beamforming diff metric unit tests."""

from __future__ import annotations

import numpy as np
from app.metrics import beamforming_metrics, diff_report, matrix_hash, to_complex


def _matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(20, 64, 10)) + 1j * rng.normal(size=(20, 64, 10))).astype(
        np.complex64
    )


def test_identical_matrices() -> None:
    matrix = _matrix()
    metrics = beamforming_metrics(matrix.real, matrix.imag, matrix.real, matrix.imag)
    assert metrics["m_cos"] == 1.0
    assert metrics["m_nmse"] == 0.0
    assert metrics["m_mse"] == 0.0


def test_perturbed_matrix_degrades() -> None:
    target = _matrix(1)
    prediction = target + 0.1 * _matrix(2)
    metrics = beamforming_metrics(
        prediction.real, prediction.imag, target.real, target.imag
    )
    assert 0.0 < metrics["m_cos"] < 1.0
    assert metrics["m_nmse"] > 0.0


def test_diff_report_has_hashes_and_rates() -> None:
    target = _matrix(3)
    prediction = target * 0.9
    report = diff_report(
        prediction.real, prediction.imag, target.real, target.imag, 52.0, 55.0
    )
    assert report["rate_delta"] == -3.0
    assert report["rate_abs_delta"] == 3.0
    assert report["gp_hash"] == matrix_hash(target.real, target.imag)
    assert report["dl_hash"] == matrix_hash(prediction.real, prediction.imag)
    assert report["gp_hash"] != report["dl_hash"]


def test_to_complex_accepts_flat_lists() -> None:
    matrix = _matrix(4)
    combined = to_complex(matrix.real.ravel().tolist(), matrix.imag.ravel().tolist())
    assert combined.shape == (20, 64, 10)
    assert np.allclose(combined, matrix)
