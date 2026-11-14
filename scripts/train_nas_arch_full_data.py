#!/usr/bin/env python3
"""Retrain a NAS-discovered architecture on the WHOLE 64x64 MIMO dataset (15K).

Consumes the memory-mappable cache from ``scripts/build_full_cache.py`` and the
winning architecture JSON from a NAS run (``.../metrics/best.json`` or
``.../best.json``). The model is the tabular ``RHSTabularNet``; training and
metrics reuse the exact helpers in ``scripts/train_best_full_data.py`` so the
numbers are directly comparable to the ``hybrid_jev`` full-data run.

Usage::

    python3 scripts/train_nas_arch_full_data.py \
        --arch-json search_results/nas_study/random_b4/metrics/best.json \
        --cache-dir data/processed/mimo64_full \
        --epochs 40 --patience 5 --outdir search_results/nas_full_data/random_b4
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch import nn

from rhs_beamforming.nas.real_space import Architecture, build_model, count_parameters
from train_best_full_data import _set_seed, eval_stream, train_epoch


def run(arch_json, cache_dir, epochs, batch_size, lr, threads, seed, patience, outdir):
    if threads:
        torch.set_num_threads(threads)
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    device = "cpu"

    arch_doc = json.loads(Path(arch_json).read_text())
    arch = Architecture.from_dict(arch_doc["architecture"])
    print(f"arch: {arch.name}")

    X_np = np.load(cache_dir / "X.npy")
    Y_np = np.load(cache_dir / "Y.npy", mmap_mode="r")
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)

    class MMapY:
        def __getitem__(self, idx):
            return torch.as_tensor(np.ascontiguousarray(Y_np[np.asarray(idx)]),
                                   dtype=torch.float32, device=device)

        def __len__(self):
            return Y_np.shape[0]

    Y = MMapY()
    tr = np.load(cache_dir / "split_train.npy")
    va = np.load(cache_dir / "split_val.npy")
    te = np.load(cache_dir / "split_test.npy")
    rate_mean = float(meta["rate_mean"])
    rate_std = float(meta["rate_std"])
    print(f"cache: {cache_dir}  total={meta['n']} train={len(tr)} val={len(va)} test={len(te)}")

    outdir = Path(outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    model = build_model(arch)
    params = count_parameters(model)
    print(f"model: {arch.name}  params={params/1e6:.3f}M")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 1e-2)
    lossfn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    rate_mean_t = torch.tensor(rate_mean, dtype=torch.float32, device=device)
    rate_std_t = torch.tensor(rate_std, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss = train_epoch(model, X, Y, tr, opt, batch_size, 4, 0.0,
                              rate_mean_t, rate_std_t, device, rng, lossfn)
        sched.step()
        val = eval_stream(model, X, Y, va, batch_size, rate_mean, rate_std, device)
        val_loss = val["M_mse"] + val["rate_mse"] / max(rate_std ** 2, 1e-9)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_rate_r2": val["rate_r2"], "val_M_n_mse": val["M_n_mse"]})
        print(f"    epoch {epoch+1:2d} train={tr_loss:.4f} val={val_loss:.4f} "
              f"rate_r2={val['rate_r2']:.4f} M_cos={val['M_cos']:.4f} M_nMSE={val['M_n_mse']:.4f} "
              f"({time.time()-t0:.0f}s)")
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    >> early stop: restoring best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    val = eval_stream(model, X, Y, va, batch_size, rate_mean, rate_std, device)
    test = eval_stream(model, X, Y, te, batch_size, rate_mean, rate_std, device)

    from rhs_beamforming.nas.real_trainer import benchmark_latency

    latency = benchmark_latency(model, input_dim=arch.input_dim, device=device)
    torch.save(model.state_dict(), outdir / "checkpoints" / "nas_best.pt")
    with (outdir / "nas_best.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    result = {
        "architecture": arch.to_dict(),
        "arch_name": arch.name,
        "arch_source": str(arch_json),
        "cache_dir": str(cache_dir),
        "samples_total": meta["n"], "train": len(tr), "val": len(va), "test": len(te),
        "epochs_requested": epochs, "epochs_reported": best_epoch,
        "early_stopped": len(history) < epochs, "best_val_loss": best_val,
        "parameters": params, "latency_ms": latency,
        "val_rate_r2": val["rate_r2"], "val_rate_mae": val["rate_mae"],
        "val_M_n_mse": val["M_n_mse"], "val_M_cos": val["M_cos"], "val_M_r2": val["M_r2"],
        "test_rate_r2": test["rate_r2"], "test_rate_mae": test["rate_mae"],
        "test_rate_mse": test["rate_mse"], "test_M_n_mse": test["M_n_mse"],
        "test_M_cos": test["M_cos"], "test_M_r2": test["M_r2"], "test_M_mse": test["M_mse"],
        "rate_mean": rate_mean, "rate_std": rate_std,
    }
    result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in result.items()}
    (outdir / "results.json").write_text(json.dumps(result, indent=2))
    print("\n=== FINAL ===")
    print(json.dumps({x: result[x] for x in [
        "arch_name", "epochs_reported", "early_stopped", "parameters", "latency_ms",
        "val_rate_r2", "test_rate_r2", "test_rate_mae",
        "test_M_cos", "test_M_n_mse", "test_M_r2"]}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch-json", required=True)
    p.add_argument("--cache-dir", default="data/processed/mimo64_full")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="search_results/nas_full_data")
    args = p.parse_args()
    run(args.arch_json, args.cache_dir, args.epochs, args.batch_size, args.lr,
        args.threads, args.seed, args.patience, args.outdir)


if __name__ == "__main__":
    main()
