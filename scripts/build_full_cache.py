#!/usr/bin/env python3
"""Build a full-data training cache for the whole 64x64 MIMO dataset.

The on-disk HDF5 stores ``M`` gzip-compressed with chunk ``(128,3,8,3)``, which
makes random per-sample reads very slow (especially over a v9fs/DrvFs mount).
This script performs a **single sequential pass** over the HDF5 (fast) and writes
a plain, uncompressed ``.npy`` pair that can be memory-mapped for training:

    X.npy  (N,10)      float32, standardised config features
    Y.npy  (N,25601)   float32, [Re(M).ravel(), Im(M).ravel(), rate]
    split_{train,val,test}.npy   int64 index arrays
    scaler.json, meta.json

With ~15000 samples Y is ~1.54 GB and X is ~0.6 MB; both fit on the native
filesystem and are memory-mapped (not fully loaded) during training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rhs_beamforming.data.mimo64 import (
    ELEMENTS_PER_SAMPLE,
    FEATURE_INDICES,
    OUTPUT_DIM,
    RATE_COLUMN,
    FeatureScaler,
    M_to_realimag,
    make_splits,
    read_metadata,
)


def build(h5_path, outdir, seed, chunk):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = read_metadata(h5_path)
    n = meta.shape[0]
    rate = meta[:, RATE_COLUMN].astype(np.float32)
    splits = make_splits(meta, n=n, seed=seed)
    tr = splits["train"]

    scaler = FeatureScaler.fit(meta[tr][:, FEATURE_INDICES].astype(np.float32))
    X = scaler.transform(meta[:, FEATURE_INDICES].astype(np.float32)).astype(np.float32)

    Y = np.lib.format.open_memmap(outdir / "Y.npy", mode="w+", dtype=np.float32,
                                  shape=(n, OUTPUT_DIM))
    np.save(outdir / "X.npy", X)
    E = ELEMENTS_PER_SAMPLE
    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            dset = f["M"]
            t0 = time.time()
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                block = np.asarray(dset[start:stop], dtype=np.complex64)
                for j in range(block.shape[0]):
                    Y[start + j, : 2 * E] = M_to_realimag(block[j])
                Y[start:stop, 2 * E] = rate[start:stop]
                Y.flush()
                if (start // chunk) % 10 == 0:
                    done = stop
                    el = time.time() - t0
                    eta = el / done * (n - done)
                    print(f"  {done}/{n}  {done/el:.1f} samples/s  eta {eta/60:.1f} min",
                          flush=True)
    finally:
        del Y

    for name in ("train", "val", "test"):
        np.save(outdir / f"split_{name}.npy", splits[name])
    (outdir / "scaler.json").write_text(json.dumps(
        {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, indent=2))
    (outdir / "meta.json").write_text(json.dumps({
        "h5": str(h5_path), "n": int(n),
        "train": len(splits["train"]), "val": len(splits["val"]),
        "test": len(splits["test"]),
        "rate_mean": float(rate[tr].mean()), "rate_std": float(rate[tr].std() or 1.0),
        "output_dim": OUTPUT_DIM, "elements_per_sample": E,
    }, indent=2))
    print(f"cache written to {outdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default="/tmp/opencode/mimo/64x64_MIMO_beamforming_dataset.h5")
    p.add_argument("--outdir", default="data/processed/mimo64_full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=64)
    args = p.parse_args()
    build(args.h5, args.outdir, args.seed, args.chunk)


if __name__ == "__main__":
    main()