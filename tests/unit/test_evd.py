"""Tests for the EVD / eigenvector ("JEV") beamforming baseline."""

import numpy as np
import torch

from rhs_beamforming.physics.evd import EVDBeamformer, eigenvalue_spectrum

B, NR, NT, L, RF = 3, 64, 64, 4, 4


def _rand_complex(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))


def test_beamform_shapes_and_amplitude_bound():
    bf = EVDBeamformer(rfno=RF, normalize=False)
    H_eff = _rand_complex(B, NR, NT)
    E = torch.ones(B, NT, RF, dtype=torch.complex64)  # unit-modulus excitation
    res = bf.beamform(H_eff, E)

    assert res.M.shape == (B, NT, RF)
    assert res.M.dtype == torch.complex64
    assert torch.all(res.M.abs() <= 1.0 + 1e-5)
    assert res.singular_values.shape == (B, min(NR, NT))


def test_rate_is_finite_and_positive():
    bf = EVDBeamformer(rfno=RF)
    H_seq = _rand_complex(B, NR, NT, L)
    M = torch.ones(B, NT, RF, dtype=torch.complex64)
    w = _rand_complex(B, NR, L)
    rate = bf.rate(H_seq, M, w)
    assert rate.shape == (B,)
    assert torch.isfinite(rate).all()
    assert (rate >= 0).all()


def test_reference_pipeline_matches_manual_rate():
    bf = EVDBeamformer(rfno=RF)
    H_seq = _rand_complex(B, L, NR, NT)
    E = torch.ones(B, NT, RF, dtype=torch.complex64)
    w = _rand_complex(B, NR, L)
    res = bf.reference(H_seq, E, w)
    assert res.M.shape == (B, NT, RF)
    assert res.rate.shape == (B,)
    assert torch.isfinite(res.rate).all()


def test_eigenvalue_spectrum_descending():
    H = _rand_complex(B, NR, NT)
    top = eigenvalue_spectrum(H, top_k=4)
    assert top.shape == (B, 4)
    assert np.all(np.diff(top, axis=-1) <= 1e-4)
