#!/usr/bin/env python3
"""Prepare the real 64x64 MIMO dataset cache (X/Y, splits, scalers).

Produces a compressed ``.npz`` containing scaled features, targets, splits and
scalers so the NAS search can reload instantly without touching the 18 GB file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rhs_beamforming.nas.real_data import prepare_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed/mimo64_5000.npz")
    args = p.parse_args()

    path = prepare_cache(args.h5, n_samples=args.samples, seed=args.seed, out_path=args.out)
    cache = np.load(path, allow_pickle=True)
    print(f"cache: {path} ({Path(path).stat().st_size/1e6:.1f} MB)")
    print(f"X: {cache['X'].shape}  Y: {cache['Y'].shape}  M: {cache['M'].shape}")
    print(f"train/val/test: {len(cache['train'])}/{len(cache['val'])}/{len(cache['test'])}")
    print(f"rate mean/std: {float(cache['rate_mean'][0]):.3f}/{float(cache['rate_std'][0]):.3f}")
    print(f"freq GHz: {3e8/cache['metadata'][:,3].min()/1e9:.2f}..{3e8/cache['metadata'][:,3].max()/1e9:.2f}")
    print(f"features: {list(cache['feature_names'])}")


if __name__ == "__main__":
    main()
