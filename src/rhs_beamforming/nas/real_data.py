"""Data preparation + loader helpers for the real 64x64 MIMO dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ..data.mimo64 import (
    FeatureScaler,
    Mimo64TorchDataset,
    build_xy,
    make_splits,
    read_metadata,
)


def prepare_cache(
    h5_path: str | Path,
    n_samples: int = 5000,
    seed: int = 42,
    out_path: str | Path = "data/processed/mimo64_5000.npz",
    stratify: bool = True,
) -> Path:
    """Sample ``n_samples``, build X/Y, split, scale, and cache to npz."""
    meta = read_metadata(h5_path)
    rng = np.random.default_rng(seed)
    if n_samples >= meta.shape[0]:
        idx = np.arange(meta.shape[0])
    else:
        idx = np.sort(rng.choice(meta.shape[0], size=n_samples, replace=False))

    data = build_xy(h5_path, idx, return_M=True)
    X, Y, M, rate = data["X"], data["Y"], data["M"], data["rate"]
    sub_meta = data["metadata"]

    if stratify:
        splits = make_splits(sub_meta, n=len(idx), seed=seed)
    else:
        pos = np.arange(len(idx))
        splits = {
            "train": pos[: int(0.8 * len(idx))],
            "val": pos[int(0.8 * len(idx)) : int(0.9 * len(idx))],
            "test": pos[int(0.9 * len(idx)) :],
        }
    # splits indices are positions into the sub-array
    tr = splits["train"]
    scaler = FeatureScaler.fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)
    rate_mean = float(rate[tr].mean())
    rate_std = float(rate[tr].std() or 1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=Xs,
        X_raw=X,
        Y=Y,
        M=M,
        rate=rate,
        metadata=sub_meta,
        global_indices=idx,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        X_mean=scaler.mean_,
        X_scale=scaler.scale_,
        rate_mean=np.array([rate_mean], dtype=np.float64),
        rate_std=np.array([rate_std], dtype=np.float64),
        feature_names=np.array([str(x) for x in data["feature_names"]]),
    )
    return out_path


def load_cache(path: str | Path, include_M: bool = False) -> dict:
    """Load a prepared cache. ``M`` (512 MB) is skipped unless requested."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files if include_M or k != "M"}
    return out


def build_loaders(cache: dict, batch_size: int = 64, num_workers: int = 0,
                  device: str = "cpu") -> dict[str, DataLoader]:
    """Return train/val/test DataLoaders from a prepared cache."""
    X, Y, rate = cache["X"], cache["Y"], cache["rate"]
    M = cache.get("M")
    loaders = {}
    for split in ["train", "val", "test"]:
        idx = cache[split]
        M_split = M[idx] if M is not None else None
        ds = Mimo64TorchDataset(X[idx], Y[idx], M_split, rate[idx])
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train" and len(ds) > batch_size),
        )
    return loaders
