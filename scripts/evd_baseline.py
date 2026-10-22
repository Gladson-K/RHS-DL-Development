#!/usr/bin/env python3
"""Compute the EVD / eigenvector (a.k.a. "JEV") beamforming reference.

Compares the eigen-based reference rate against the stored GP reference rate
on a subset of the real 64x64 dataset and reports the ratio/correlation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import torch

from rhs_beamforming.data.mimo64 import read_metadata
from rhs_beamforming.physics.evd import EVDBeamformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--cache", default=None, help="prepared npz (uses global_indices subset)")
    p.add_argument("--samples", type=int, default=1000, help="cap on samples to process")
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--out", default="search_results/mimo64_nas/evd_baseline.json")
    args = p.parse_args()

    meta = read_metadata(args.h5)
    if args.cache and Path(args.cache).exists():
        cache = np.load(args.cache, allow_pickle=True)
        indices = np.asarray(cache["global_indices"], dtype=np.int64)
    else:
        indices = np.arange(meta.shape[0])
    indices = np.sort(indices)[: args.samples]

    beamformer = EVDBeamformer(rfno=10, normalize=False, device="cpu")
    evd_rates = np.zeros(len(indices), dtype=np.float32)

    with h5py.File(args.h5, "r") as f:
        for s in range(0, len(indices), args.block):
            block = indices[s : s + args.block]
            Hb = np.asarray(f["H_testD"][block])  # (B,20,Nr,Nt,L)
            Eb = np.asarray(f["E"][block])        # (B,20,Nt,RFno)
            wb = np.asarray(f["w"][block])        # (B,20,Nr,MAX_L)
            for j, gi in enumerate(block):
                P = min(int(meta[gi, 8]), 20)
                L = int(meta[gi, 2])
                H = torch.as_tensor(Hb[j, :P, :, :, :L].copy())  # [P,Nr,Nt,L]
                E = torch.as_tensor(Eb[j, :P].copy())            # [P,Nt,RFno]
                w = torch.as_tensor(wb[j, :P, :, :L].copy())     # [P,Nr,L]
                Heff = H.mean(dim=3)                             # [P,Nr,Nt]
                res = beamformer.beamform(Heff, E)               # M [P,Nt,RFno]
                r = beamformer.rate(H, res.M, w)                 # [P]  H=[P,Nr,Nt,L]
                evd_rates[s + j] = float(r.mean().item())
            print(f"  {s + len(block)}/{len(indices)}", flush=True)

    gp_rates = meta[indices, 12].astype(np.float32)
    mask = np.isfinite(evd_rates) & np.isfinite(gp_rates)
    ratio = float(np.mean(evd_rates[mask] / np.clip(gp_rates[mask], 1e-9, None)))
    corr = float(np.corrcoef(evd_rates[mask], gp_rates[mask])[0, 1])
    report = {
        "n_samples": int(mask.sum()),
        "gp_rate_mean": float(gp_rates[mask].mean()),
        "evd_rate_mean": float(evd_rates[mask].mean()),
        "evd_over_gp_ratio": ratio,
        "pearson_corr": corr,
        "gp_rate_std": float(gp_rates[mask].std()),
        "evd_rate_std": float(evd_rates[mask].std()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    np.savez_compressed(out.with_suffix(".npz"), indices=indices, gp_rate=gp_rates, evd_rate=evd_rates)
    print(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
