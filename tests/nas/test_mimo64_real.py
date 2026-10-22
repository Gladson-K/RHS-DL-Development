"""Tests for the real 64x64 MIMO dataset loader and cache pipeline."""

import numpy as np
import pytest

from rhs_beamforming.data.mimo64 import (
    FEATURE_INDICES,
    N_META_COLS,
    NR,
    NT,
    OUTPUT_DIM,
    RATE_COLUMN,
)
from rhs_beamforming.nas.real_data import load_cache, prepare_cache

N = 40
L = 4
RF = 4
TS = 7


def _write_synthetic_h5(path, n=N, seed=0):
    """Write a tiny HDF5 with the real schema (non-M arrays zeroed to stay small)."""
    import h5py

    rng = np.random.default_rng(seed)
    meta = np.zeros((n, N_META_COLS), dtype=np.float64)
    meta[:, 2] = L          # L
    meta[:, 3] = 0.03       # lamd -> ~10 GHz
    meta[:, 5] = 20.0       # SNR
    meta[:, 6] = 1.0        # Pt
    meta[:, 8] = TS         # test_size
    meta[:, 9] = RF         # RFno
    meta[:, 11] = rng.uniform(1e-3, 1e-2, size=n)  # noise_var (varies, not stratified)
    meta[:, RATE_COLUMN] = rng.uniform(20.0, 60.0, size=n)

    M = np.zeros((n, 20, NT, 10), dtype=np.complex64)
    M[:, :TS, :, :RF] = (
        rng.standard_normal((n, TS, NT, RF)) + 1j * rng.standard_normal((n, TS, NT, RF))
    ).astype(np.complex64) / 10.0

    with h5py.File(path, "w") as f:
        f.create_dataset("metadata", data=meta)
        f.create_dataset("M", data=M, compression="gzip")
        f.create_dataset("w", data=np.zeros((n, 20, NT, 8), np.complex64))
        f.create_dataset("E", data=np.zeros((n, 20, NT, 10), np.complex64))
        f.create_dataset("H_testD", data=np.zeros((n, 20, NR, NT, 8), np.complex64))
        f.create_dataset("W_MMSE", data=np.zeros((n, 20, NR, NT), np.complex64))


@pytest.fixture(scope="module")
def synthetic_h5(tmp_path_factory):
    path = tmp_path_factory.mktemp("mimo") / "synthetic.h5"
    _write_synthetic_h5(path)
    return path


@pytest.fixture(scope="module")
def cache(synthetic_h5, tmp_path_factory):
    cache_path = tmp_path_factory.mktemp("cache") / "cache.npz"
    prepare_cache(synthetic_h5, n_samples=N, seed=1, out_path=cache_path)
    return load_cache(cache_path)


def test_metadata_columns(synthetic_h5):
    import h5py

    with h5py.File(synthetic_h5, "r") as f:
        meta = f["metadata"][:]
    assert meta.shape == (N, N_META_COLS)
    assert FEATURE_INDICES[0] == 2
    assert FEATURE_INDICES[-1] == 11
    assert RATE_COLUMN == 12


def test_cache_shapes_and_target(cache):
    assert cache["X"].shape == (N, len(FEATURE_INDICES))
    assert cache["Y"].shape == (N, OUTPUT_DIM)
    assert "M" not in cache  # dropped by default to save memory
    assert np.allclose(cache["Y"][:, -1], cache["rate"], atol=1e-4)


def test_splits_partition_and_are_disjoint(cache):
    train, val, test = cache["train"], cache["val"], cache["test"]
    assert len(train) + len(val) + len(test) == N
    assert len(val) > 0 and len(test) > 0
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_scaler_fit_on_train(cache):
    train_x = cache["X"][cache["train"]]
    assert np.abs(train_x.mean(axis=0)).max() < 1e-4
    s = train_x.std(axis=0)
    # Each scaled feature is either standardised (~1) or degenerate (~0).
    assert np.all((s < 1e-3) | (np.abs(s - 1.0) < 1e-3))
    assert (np.abs(s - 1.0) < 1e-3).any()
