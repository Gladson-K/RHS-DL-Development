#!/usr/bin/env python3
"""Train the best NAS recipe on the WHOLE 64x64 MIMO dataset (15K, CPU).

Best NAS recipe selected from the five-setup study: ``hybrid_jev`` (GNN +
attachable Transformer + differentiable JEV eigen-decision head). It achieved
the best beamforming fidelity on the 5K cache (M cos 0.761, M nMSE 0.422,
M R^2 0.578, best rate MAE 3.97) and trained longest/most stably under the JEV
spectral-consistency regulariser.

This script consumes the memory-mappable cache produced by
``scripts/build_full_cache.py`` (X.npy / Y.npy / split_*.npy), so it does not
touch the slow gzip HDF5 during training and never materialises the 1.5 GB Y.

Usage::

    python3 scripts/train_best_full_data.py \
        --cache-dir data/processed/mimo64_full \
        --epochs 40 --patience 5 --batch-size 64 --threads 8 \
        --outdir search_results/mimo64_full_data
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn

from rhs_beamforming.nas.real_trainer import E, benchmark_latency
from rhs_beamforming.nas.setups import build_setup_by_name

PATIENCE = 5


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spectral_target(y, k: int = 4, device=None):
    """EVD spectral signature softmax(top-k eig(M^H M)) for the JEV regulariser."""
    E_slice = (y.shape[1] - 1) // 2 if y.shape[1] % 2 == 1 else y.shape[1] // 2
    mc = torch.complex(y[:, :E_slice].reshape(y.shape[0], -1, 10),
                       y[:, E_slice : 2 * E_slice].reshape(y.shape[0], -1, 10))
    gram = mc.conj().transpose(-1, -2) @ mc
    evals = torch.linalg.eigvalsh(gram)
    return torch.softmax(evals[:, -k:], dim=-1).to(device)


def train_epoch(model, X, Y, indices, opt, batch_size, k, jev_alpha, rate_mean,
                rate_std, device, rng, lossfn, grad_clip=1.0):
    model.train()
    running, nb = 0.0, 0
    order = rng.permutation(len(indices))
    for start in range(0, len(order), batch_size):
        idx = indices[order[start : start + batch_size]]
        x = X[idx]
        y = Y[idx]
        out = model(x)
        pmre, pMim, prate = out[:, :E], out[:, E : 2 * E], out[:, 2 * E]
        tmre, tMim = y[:, :E], y[:, E : 2 * E]
        trate = (y[:, 2 * E] - rate_mean) / rate_std
        loss = 0.5 * (lossfn(pmre, tmre) + lossfn(pMim, tMim)) + lossfn(prate, trate)
        if jev_alpha > 0:
            head = model.head
            from rhs_beamforming.models.jev_head import JEVDecisionHead

            if isinstance(head, JEVDecisionHead):
                w_model = torch.softmax(head.last_eigenvalues, dim=-1)
                loss = loss + jev_alpha * lossfn(w_model, spectral_target(y, k, device))
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        running += loss.item()
        nb += 1
    return running / max(nb, 1)


def eval_stream(model, X, Y, indices, batch_size, rate_mean, rate_std, device):
    """Chunked evaluation using the SAME metrics as ``real_trainer.evaluate``.

    Reproduces M_mse / M_n_mse (energy-normalised) / M_cos / M_r2 and rate
    R2/MAE without materialising all Y. ``y`` holds the RAW rate at column 2E,
    so only the prediction is rescaled back from the normalised head output.
    """
    model.eval()
    se_M = se_rate = 0.0
    ae_rate = 0.0
    dot = norm_p = norm_t = sum_sq = 0.0
    n = 0
    mean_re = mean_im = 0.0
    rate_t, rate_p = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            x, y = X[idx], Y[idx]
            out = model(x)
            b = out.shape[0]
            n += b
            pmre, pMim, prate = out[:, :E], out[:, E : 2 * E], out[:, 2 * E]
            tmre, tMim, trate = y[:, :E], y[:, E : 2 * E], y[:, 2 * E]
            se_M += (pmre - tmre).pow(2).sum().item() + (pMim - tMim).pow(2).sum().item()
            p_rate = prate * rate_std + rate_mean
            se_rate += (p_rate - trate).pow(2).sum().item()
            ae_rate += (p_rate - trate).abs().sum().item()
            mean_re += tmre.mean().item() * b
            mean_im += tMim.mean().item() * b
            p = torch.cat([pmre, pMim], dim=1)
            t = torch.cat([tmre, tMim], dim=1)
            dot += (p * t).sum().item()
            norm_p += p.pow(2).sum().item()
            norm_t += t.pow(2).sum().item()
            sum_sq += t.pow(2).sum().item()
            rate_t.append(trate.cpu().numpy())
            rate_p.append(p_rate.cpu().numpy())
    mean_re /= max(n, 1)
    mean_im /= max(n, 1)
    # second pass for the target sum-of-squares about the mean (M R2)
    ss_tot = 0.0
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        y = Y[idx]
        tmre, tMim = y[:, :E], y[:, E : 2 * E]
        ss_tot += (tmre - mean_re).pow(2).sum().item() + (tMim - mean_im).pow(2).sum().item()
    total_elems = max(n * 2 * E, 1)
    rate_t = np.concatenate(rate_t)
    rate_p = np.concatenate(rate_p)
    return {
        "n": n,
        "M_mse": se_M / total_elems,
        "M_n_mse": se_M / max(sum_sq, 1e-9),
        "M_cos": dot / max(np.sqrt(norm_p) * np.sqrt(norm_t), 1e-9),
        "M_r2": 1.0 - se_M / max(ss_tot, 1e-9),
        "rate_r2": 1.0 - se_rate / max(((rate_t - rate_t.mean()) ** 2).sum(), 1e-9),
        "rate_mae": ae_rate / max(n, 1),
        "rate_mse": se_rate / max(n, 1),
    }


def run(cache_dir, epochs, batch_size, lr, threads, outdir, jev_alpha, seed, patience):
    if threads:
        torch.set_num_threads(threads)
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    device = "cpu"

    # keep the 1.5 GB target memory-mapped and only index batches into it
    X_np = np.load(cache_dir / "X.npy")
    Y_np = np.load(cache_dir / "Y.npy", mmap_mode="r")
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)

    def y_batch(arr):
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)

    class MMapY:
        def __getitem__(self, idx):
            return y_batch(Y_np[np.asarray(idx)])

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
    model = build_setup_by_name("hybrid_jev")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: hybrid_jev  params={params/1e6:.3f}M  jev_alpha={jev_alpha}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 1e-2)
    lossfn = nn.MSELoss()
    k = 4
    rng = np.random.default_rng(seed)
    rate_mean_t = torch.tensor(rate_mean, dtype=torch.float32, device=device)
    rate_std_t = torch.tensor(rate_std, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss = train_epoch(model, X, Y, tr, opt, batch_size, k, jev_alpha,
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
                print(f"    >> early stop: no improvement for {patience} epochs; "
                      f"restoring epoch-{patience} (best epoch {best_epoch}) weights")
                break

    model.load_state_dict(best_state)
    val = eval_stream(model, X, Y, va, batch_size, rate_mean, rate_std, device)
    test = eval_stream(model, X, Y, te, batch_size, rate_mean, rate_std, device)
    latency = benchmark_latency(model, input_dim=10, device=device)
    torch.save(model.state_dict(), outdir / "checkpoints" / "hybrid_jev.pt")
    with (outdir / "hybrid_jev.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    result = {
        "setup": "hybrid_jev", "cache_dir": str(cache_dir),
        "samples_total": meta["n"], "train": len(tr), "val": len(va), "test": len(te),
        "epochs_requested": epochs, "epochs_reported": best_epoch,
        "early_stopped": len(history) < epochs, "best_val_loss": best_val,
        "parameters": params, "latency_ms": latency, "jev_alpha": jev_alpha,
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
    print("\n=== FINAL (best/restored weights) ===")
    print(json.dumps({x: result[x] for x in [
        "epochs_reported", "early_stopped", "parameters", "latency_ms",
        "val_rate_r2", "test_rate_r2", "test_rate_mae",
        "test_M_cos", "test_M_n_mse", "test_M_r2"]}, indent=2))
    print(f"saved {outdir/'results.json'} and {outdir/'checkpoints'/'hybrid_jev.pt'}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="data/processed/mimo64_full")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--jev-alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="search_results/mimo64_full_data")
    args = p.parse_args()
    run(args.cache_dir, args.epochs, args.batch_size, args.lr, args.threads, args.outdir,
        args.jev_alpha, args.seed, args.patience)


if __name__ == "__main__":
    main()