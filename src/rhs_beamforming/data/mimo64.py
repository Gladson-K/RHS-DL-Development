"""Real 64x64 MIMO RHS holographic-beamforming dataset I/O.

This module wires the *actual* research dataset
(``64x64_MIMO_beamforming_dataset.zip`` -> ``64x64_MIMO_beamforming_dataset.h5``)
into the project so that the NAS pipeline is trained/evaluated on real
physics (Rayleigh fading + MMSE receive combiner + Gradient-Projection
holographic beamforming) instead of synthetic noise.

HDF5 schema (N samples, complex64 unless noted)::

    metadata   (N, 15) float64
    M          (N, 20, 64, 10)     holographic beamforming matrix  (target)
    w          (N, 20, 64, 8)      digital beamformer             (target)
    E          (N, 20, 64, 10)     RHS excitation matrix
    H_testD    (N, 20, 64, 64, 8)  effective channel per position/subcarrier
    W_MMSE     (N, 20, 64, 64)     MMSE receive combiner

metadata columns::

    0 Nt  1 Nr  2 L  3 lamd  4 scale  5 SNR  6 Pt  7 deH  8 test_size
    9 RFno  10 stepSize1  11 noise_var  12 rate  13 time  14 aux

The learning contract mirrors the reference ver1.0-CNN:

    X (input)  = metadata[:, [2..11]]            -> 10 config features
    Y (target) = [Re(M).ravel(), Im(M).ravel(), rate]   -> 25601 values

``M`` is padded/truncated to ``(20, 64, 10)`` exactly as the reference
pipeline does, so the same metrics (M_real / M_imag / sum_rate MSE, MAE, R2)
apply.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # torch is optional for pure-data utilities
    import torch
    from torch.utils.data import Dataset
except Exception:  # noqa: BLE001  # pragma: no cover - allows numpy-only usage
    torch = None

    class Dataset:  # type: ignore
        pass


# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
NT = 64
NR = 64
MAX_TEST_SIZE = 20
MAX_L = 8
MAX_RFNO = 10
ELEMENTS_PER_SAMPLE = MAX_TEST_SIZE * NT * MAX_RFNO  # 12800
OUTPUT_DIM = 2 * ELEMENTS_PER_SAMPLE + 1  # 25601

META_COLUMNS = [
    "Nt",
    "Nr",
    "L",
    "lamd",
    "scale",
    "SNR",
    "Pt",
    "deH",
    "test_size",
    "RFno",
    "stepSize1",
    "noise_var",
    "rate",
    "time",
    "aux",
]

# ver1.0-CNN feature columns: [L, lamd, scale, SNR, Pt, deH, test_size,
# RFno, stepSize1, noise_var]
FEATURE_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
FEATURE_NAMES = [
    "L",
    "lamd",
    "scale",
    "SNR_db",
    "Pt",
    "deH",
    "test_size",
    "RFno",
    "stepSize1",
    "noise_var",
]
RATE_COLUMN = 12
N_META_COLS = len(META_COLUMNS)
SPEED_OF_LIGHT = 3e8

DEFAULT_H5_NAME = "64x64_MIMO_beamforming_dataset.h5"
DEFAULT_ZIP_CANDIDATES = [
    "64x64_MIMO_beamforming_dataset.zip",
    "Output.zip",
]


# --------------------------------------------------------------------------- #
# Extraction / access
# --------------------------------------------------------------------------- #
def extract_h5_from_zip(zip_path: str | Path, dest_dir: str | Path, overwrite: bool = False) -> Path:
    """Stream-extract the single HDF5 member from the (potentially huge) zip.

    The archive stores a ~18 GB barely-compressible HDF5 file; we copy in
    chunks so it can be extracted on machines with modest RAM.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith((".h5", ".hdf5"))]
        if not members:
            raise FileNotFoundError(f"No HDF5 member found in {zip_path}")
        member = members[0]
        info = zf.getinfo(member)
        dest = dest_dir / Path(member).name

        if dest.exists() and not overwrite and dest.stat().st_size == info.file_size:
            return dest

        tmp = dest.with_suffix(dest.suffix + ".part")
        with zf.open(member) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(16 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        tmp.replace(dest)
        return dest


def resolve_zip(zip_path: str | Path | None, extra_dirs: list[str | Path] | None = None) -> Path:
    """Locate the dataset zip across common locations."""
    if zip_path is not None and Path(zip_path).exists():
        return Path(zip_path)
    search_dirs = [Path.cwd()]
    if extra_dirs:
        search_dirs += [Path(d) for d in extra_dirs]
    for d in search_dirs:
        for name in DEFAULT_ZIP_CANDIDATES:
            cand = d / name
            if cand.exists() and cand.stat().st_size > 1_000_000:
                return cand
    raise FileNotFoundError("Could not locate 64x64 MIMO dataset zip. Pass --zip explicitly.")


def read_metadata(h5_path: str | Path) -> np.ndarray:
    """Return the full metadata matrix as float64 ``(N, 15)``."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["metadata"][:], dtype=np.float64)


def frequencies_ghz(meta: np.ndarray) -> np.ndarray:
    """Convert wavelength column (m) to carrier frequency in GHz."""
    return SPEED_OF_LIGHT / np.asarray(meta[:, 3], dtype=np.float64) / 1e9


# --------------------------------------------------------------------------- #
# Target construction (ver1.0-CNN contract)
# --------------------------------------------------------------------------- #
def M_to_realimag(M: np.ndarray, test_size: int = MAX_TEST_SIZE, rfno: int = MAX_RFNO) -> np.ndarray:
    """Flatten complex ``M`` to ``[Re, Im]`` with padding/truncation.

    ``M`` accepted as ``(T, 64, R)`` or already padded ``(20, 64, 10)``.
    Returns ``(2 * test_size * 64 * rfno,)`` float32.
    """
    M = np.asarray(M)
    out = np.zeros((test_size, NT, rfno), dtype=np.complex64)
    ts = min(M.shape[0], test_size)
    rf = min(M.shape[1] if M.ndim == 2 else M.shape[2], rfno)
    if M.ndim == 3:
        out[:ts, :, :rf] = M[:ts, :, :rf]
    else:  # (T, RFno) not expected
        raise ValueError(f"Unexpected M shape {M.shape}")
    re = out.real.astype(np.float32).ravel()
    im = out.imag.astype(np.float32).ravel()
    return np.concatenate([re, im])


def build_xy(
    h5_path: str | Path,
    indices: np.ndarray | list[int] | None = None,
    feature_indices: list[int] = FEATURE_INDICES,
    test_size: int = MAX_TEST_SIZE,
    rfno: int = MAX_RFNO,
    return_M: bool = True,
    chunk: int = 512,
) -> dict:
    """Build the ``(X, Y, M, rate)`` arrays for the requested sample indices.

    Returns a dict with:
        X     (n, 10) float32
        Y     (n, 25601) float32   [Re(M), Im(M), rate]
        M     (n, 20, 64, 10) complex64  (if return_M)
        rate  (n,) float32
        metadata (n, 15) float64
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        meta = np.asarray(f["metadata"][:], dtype=np.float64)
        n_total = meta.shape[0]
        if indices is None:
            indices = np.arange(n_total)
        indices = np.asarray(indices, dtype=np.int64)
        order = np.argsort(indices)
        sorted_idx = indices[order]

        X = meta[indices][:, feature_indices].astype(np.float32)
        rate = meta[indices, RATE_COLUMN].astype(np.float32)

        n = len(indices)
        E = test_size * NT * rfno
        Y = np.zeros((n, 2 * E + 1), dtype=np.float32)
        M_out = np.zeros((n, test_size, NT, rfno), dtype=np.complex64) if return_M else None

        dset = f["M"]
        pos = {int(v): k for k, v in enumerate(indices)}
        for start in range(0, n, chunk):
            block = sorted_idx[start : start + chunk]
            data = np.asarray(dset[block], dtype=np.complex64)  # (b,20,64,10)
            for j, gi in enumerate(block):
                k = pos[int(gi)]
                flat = M_to_realimag(data[j], test_size=test_size, rfno=rfno)
                Y[k, : 2 * E] = flat
                Y[k, 2 * E] = rate[k]
                if return_M:
                    ts = min(data[j].shape[0], test_size)
                    rf = min(data[j].shape[2], rfno)
                    M_out[k, :ts, :, :rf] = data[j][:ts, :, :rf]
                del gi

        # ensure rate in Y after ordering (already placed per k) - ok
        return {
            "X": X,
            "Y": Y,
            "M": M_out,
            "rate": rate,
            "metadata": meta[indices],
            "feature_names": [META_COLUMNS[i] for i in feature_indices],
            "indices": indices,
        }


# --------------------------------------------------------------------------- #
# Splits / scalers
# --------------------------------------------------------------------------- #
def make_splits(
    metadata: np.ndarray,
    n: int | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified train/val/test split keyed on (SNR bin, frequency bin, RFno)."""
    n = n or metadata.shape[0]
    rng = np.random.default_rng(seed)
    snr = metadata[:n, 5]
    freq = frequencies_ghz(metadata[:n])
    rfno = metadata[:n, 9].astype(int)

    snr_b = np.digitize(snr, np.quantile(snr, [0.25, 0.5, 0.75]))
    freq_b = np.digitize(freq, np.quantile(freq, [0.25, 0.5, 0.75]))
    key = snr_b * 100 + freq_b * 10 + (rfno % 10)

    tr, va, te = [], [], []
    for u in np.unique(key):
        idx = np.where(key == u)[0]
        rng.shuffle(idx)
        if len(idx) < 5:
            # Too few samples in this stratum: keep it in train to preserve
            # the nominal train fraction on small datasets.
            tr.extend(idx)
            continue
        n_test = max(1, round(len(idx) * test_frac))
        n_val = max(1, round(len(idx) * val_frac))
        te.extend(idx[:n_test])
        va.extend(idx[n_test : n_test + n_val])
        tr.extend(idx[n_test + n_val :])
    return {
        "train": np.sort(np.array(tr, dtype=np.int64)),
        "val": np.sort(np.array(va, dtype=np.int64)),
        "test": np.sort(np.array(te, dtype=np.int64)),
    }


@dataclass
class FeatureScaler:
    """Minimal standard scaler (avoids a hard sklearn dependency at inference)."""

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> FeatureScaler:
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        # Guard against zero / numerically-degenerate variance so that
        # near-constant features are not amplified by a tiny divisor.
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean_=mean.astype(np.float32), scale_=scale.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_

    def to_dict(self) -> dict:
        return {"mean": self.mean_, "scale": self.scale_}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureScaler:
        return cls(mean_=np.asarray(d["mean"], dtype=np.float32),
                   scale_=np.asarray(d["scale"], dtype=np.float32))


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
class Mimo64TorchDataset(Dataset):
    """Torch dataset over pre-built ``X``/``Y`` arrays.

    Item: ``{"x": [10], "y": [25601], "M": [20,64,10] complex, "rate": scalar}``
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray | None = None,
                 rate: np.ndarray | None = None):
        if torch is None:
            raise RuntimeError("torch is required for Mimo64TorchDataset")
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.M = torch.as_tensor(M, dtype=torch.complex64) if M is not None else None
        self.rate = torch.as_tensor(rate, dtype=torch.float32) if rate is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> dict:
        item = {"x": self.X[i], "y": self.Y[i]}
        if self.M is not None:
            item["M"] = self.M[i]
        if self.rate is not None:
            item["rate"] = self.rate[i]
        return item


def validation_report(h5_path: str | Path) -> dict:
    """Dataset validation report (shapes, ranges, physics sanity)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        meta = np.asarray(f["metadata"][:], dtype=np.float64)
        report = {
            "h5_path": str(h5_path),
            "n_samples": int(meta.shape[0]),
            "metadata_cols": int(meta.shape[1]),
            "datasets": {},
            "features": [META_COLUMNS[i] for i in FEATURE_INDICES],
        }
        for k in f:
            d = f[k]
            report["datasets"][k] = {"shape": list(d.shape), "dtype": str(d.dtype)}
    freq = frequencies_ghz(meta)
    report.update(
        {
            "freq_ghz_min": float(freq.min()),
            "freq_ghz_max": float(freq.max()),
            "snr_db_min": float(meta[:, 5].min()),
            "snr_db_max": float(meta[:, 5].max()),
            "L_min": int(meta[:, 2].min()),
            "L_max": int(meta[:, 2].max()),
            "test_size_min": int(meta[:, 8].min()),
            "test_size_max": int(meta[:, 8].max()),
            "RFno_min": int(meta[:, 9].min()),
            "RFno_max": int(meta[:, 9].max()),
            "rate_min": float(meta[:, 12].min()),
            "rate_max": float(meta[:, 12].max()),
            "rate_mean": float(meta[:, 12].mean()),
            "nan_free": bool(not np.isnan(meta).any()),
        }
    )
    return report
