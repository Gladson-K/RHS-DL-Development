#!/usr/bin/env python3
"""Full 35-40 epoch training of the five RHS setups with early stopping.

Differences vs the 3-epoch smoke run:
  * trains up to ``--epochs`` (default 35) with early stopping patience ``--patience``
    (default 4): when validation loss worsens for 4 consecutive epochs, training
    stops and the **best (current-4th epoch) weights** are restored and reported.
  * ``hybrid_jev`` additionally receives a **JEV spectral-consistency
    regulariser**: its learned token-Gram eigenvalue weights are aligned with the
    classical EVD spectrum ``eig(M^H M)`` computed per batch from the target M
    (power-laws of the top-k eigenvalues, softmax-normalised). This forces the
    eigen-decision head to stay consistent with the true beamformer spectrum.
  * reports validation AND held-out test metrics for the restored best weights.

Outputs (``--outdir``, default ``search_results/mimo64_setups_full/``):
  results.json | results.md | per-setup <name>.csv learning curves |
  checkpoints/<name>.pt (best weights)
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

from rhs_beamforming.models.jev_head import JEVDecisionHead
from rhs_beamforming.nas.real_data import build_loaders, load_cache
from rhs_beamforming.nas.real_trainer import E, benchmark_latency, evaluate
from rhs_beamforming.nas.setups import SETUP_VARIANTS, build_setup_by_name

PATIENCE = 4


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spectral_target(y, k: int = 4, device=None):
    """Classical EVD spectrum of the target beamformer: softmax(top-k eig(M^H M)).

    ``y`` is the flat cache target ``[Re(M), Im(M), rate]`` with ``E = 12800 =
    20*64*10``. We reconstruct the complex per-sample matrix ``M (1280, 10)`` and
    take the eigenvalues of its Gram matrix (10x10 Hermitian eigendecomposition,
    vectorised). Returning the softmax of the top-k eigenvalues gives the
    *spectral signature* a classical EVD decision head would follow.
    """
    E_slice = (y.shape[1] - 1) // 2 if y.shape[1] % 2 == 1 else y.shape[1] // 2
    re = y[:, :E_slice].reshape(y.shape[0], -1, 10)
    im = y[:, E_slice : 2 * E_slice].reshape(y.shape[0], -1, 10)
    mc = torch.complex(re, im)
    gram = mc.conj().transpose(-1, -2) @ mc  # (B,10,10) Hermitian PSD
    evals = torch.linalg.eigvalsh(gram)      # (B,10) ascending real
    w = torch.softmax(evals[:, -k:], dim=-1)
    return w.to(device)


def jev_consistency_loss(model, y, k: int, device=None):
    """Penalise deviation of learned eigen-weights from the EVD spectral target."""
    head = model.head
    if not isinstance(head, JEVDecisionHead):
        return torch.tensor(0.0, device=device)
    w_model = torch.softmax(head.last_eigenvalues, dim=-1)   # (B,k)
    target = spectral_target(y, k=k, device=device)
    return nn.functional.mse_loss(w_model, target)


def train_setup(model, family, loaders, epochs, lr, rate_mean, rate_std,
                device="cpu", grad_clip=1.0, patience=PATIENCE, jev_alpha=0.0,
                quiet=False):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 1e-2)
    lossfn = nn.MSELoss()
    rate_std_t = torch.tensor(rate_std, dtype=torch.float32, device=device)
    rate_mean_t = torch.tensor(rate_mean, dtype=torch.float32, device=device)
    k = SETUP_VARIANTS[family].num_eig if family == "hybrid_jev" else 0
    train_loader = loaders["train"]

    best_state = copy.deepcopy(model.state_dict())
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            pmre, pMim, prate = out[:, :E], out[:, E : 2 * E], out[:, 2 * E]
            tmre, tMim = y[:, :E], y[:, E : 2 * E]
            trate = (y[:, 2 * E] - rate_mean_t) / rate_std_t
            loss = 0.5 * (lossfn(pmre, tmre) + lossfn(pMim, tMim)) + lossfn(prate, trate)
            if family == "hybrid_jev" and jev_alpha > 0:
                spec = jev_consistency_loss(model, y, k, device)
                loss = loss + jev_alpha * spec
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        val = evaluate(model, loaders["val"], rate_mean, rate_std, device=device)
        val_loss = val["M"]["mse"] + val["rate"]["mse"] / max(rate_std ** 2, 1e-9)
        history.append({"epoch": epoch + 1, "train_loss": running / max(nb, 1), "val_loss": val_loss,
                        "rate_r2": val["rate"]["r2"], "M_cos": val["M"]["cos"]})
        if not quiet:
            print(f"    epoch {epoch+1:2d} train={running/max(nb,1):.4f} "
                  f"val={val_loss:.4f} rate_r2={val['rate']['r2']:.4f}")
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                if not quiet:
                    print(f"    >> early stop: no improvement for {patience} epochs; "
                          f"restoring epoch-{patience} (best epoch {best_epoch}) weights")
                break
    model.load_state_dict(best_state)  # the epoch_current - 4 weight
    return model, history, best_epoch, epoch + 1


REASONING = {
    "mlp": "Tabular residual MLP (256-256-128-64) — the config-to-output path with "
           "no structural prior. Baseline. Expected: best latency/params, rate R2 ~0.80, "
           "M cos ~0.75 (matches the earlier 40-epoch run). Fastest to converge.",
    "gnn": "HaloBeam-GNN: 16 learned tokens + 2 graph message-passing blocks with "
           "trainable adjacency. Structural prior that M is spatially coupled across the "
           "array. Expected: better M fidelity than MLP (M cos ~0.76-0.79), same-rate, "
           "~2.9 ms. Good accuracy-latency for a GNN.",
    "transformer": "Transformer encoder (2 blocks, 4 heads) over 16 tokens; attention "
           "captures long-range element couplings the graph only sees locally. Smoke run "
           "already gave the best M cos (0.751) at 3 epochs. Expected winner on M fidelity: "
           "M cos ~0.79-0.82, rate R2 ~0.81-0.83 at ~5 ms.",
    "hybrid": "GNN + Transformer interleaved (G,T,G,T) — the 'attachable Transformer' "
           "variants the user requested. Combines local message passing with global "
           "attention; most expressive but heaviest (8.2 ms). Expected between GNN and "
           "transformer on M (M cos ~0.77-0.80), best rate stability.",
    "hybrid_jev": "GNN + Transformer + differentiable JEV (eigen) decision head, now "
           "regularised to follow the classical EVD spectrum eig(M^H M) of the target. "
           "The eigen decision should track the true beamformer; the regulariser adds a "
           "physics prior and stabilises the head. Expected to reach or beat transformer "
           "on M cos (~0.80-0.83) once converged, at ~5.5 ms.",
}

PREDICTED_RANGE = {
    "mlp": ("~0.80", "~0.75"),
    "gnn": ("~0.79-0.81", "~0.76-0.79"),
    "transformer": ("~0.81-0.83", "~0.79-0.82"),
    "hybrid": ("~0.80-0.81", "~0.77-0.80"),
    "hybrid_jev": ("~0.80-0.82", "~0.80-0.83"),
}


def run(cache, epochs, batch_size, lr, device, threads, setups, outdir, jev_alpha, seed):
    if threads:
        torch.set_num_threads(threads)
    loaders = build_loaders(cache, batch_size=batch_size)
    rate_mean = float(cache["rate_mean"][0])
    rate_std = float(cache["rate_std"][0])
    outdir = Path(outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    rows = []
    for name in setups:
        print(f"\n=== setup: {name} ===")
        _set_seed(seed)
        model = build_setup_by_name(name)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        model, history, best_epoch, last_epoch = train_setup(
            model, name, loaders, epochs, lr, rate_mean, rate_std,
            device=device, patience=PATIENCE, jev_alpha=jev_alpha,
        )
        train_s = time.time() - t0
        torch.save(model.state_dict(), outdir / "checkpoints" / f"{name}.pt")
        with (outdir / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        val = evaluate(model, loaders["val"], rate_mean, rate_std, device=device)
        test = evaluate(model, loaders["test"], rate_mean, rate_std, device=device)
        latency = benchmark_latency(model, input_dim=10, device=device)
        row = {
            "setup": name,
            "parameters": params,
            "latency_ms": latency,
            "train_seconds": train_s,
            "epochs_reported": best_epoch,
            "epochs_last": last_epoch,
            "early_stopped": last_epoch < epochs,
            "val_rate_r2": val["rate"]["r2"], "val_M_cos": val["M"]["cos"], "val_M_n_mse": val["M"]["n_mse"],
            "test_rate_r2": test["rate"]["r2"], "test_rate_mae": test["rate"]["mae"],
            "test_M_cos": test["M"]["cos"], "test_M_n_mse": test["M"]["n_mse"], "test_M_r2": test["M"]["r2"],
        }
        rows.append(row)
        print(f"  best val from epoch {best_epoch}/{last_epoch} (early={last_epoch < epochs}) "
              f"params={params/1e6:.3f}M lat={latency:.2f}ms t={train_s:.0f}s")
        print(f"  VAL  rate_r2={row['val_rate_r2']:.4f} M_cos={row['val_M_cos']:.4f} nMSE={row['val_M_n_mse']:.4f}")
        print(f"  TEST rate_r2={row['test_rate_r2']:.4f} M_cos={row['test_M_cos']:.4f} "
              f"nMSE={row['test_M_n_mse']:.4f} M_r2={row['test_M_r2']:.4f}")
    return rows, outdir


def write_md(rows, outdir, epochs, batch_size, lr, jev_alpha, seed, samples, splits):
    lines = [
        "# RHS-HaloBeamGNN — Full-Training Results (5 Setups)",
        "",
        (f"Real 64x64 MIMO cache ({samples} samples; train/val/test = "
         f"{splits[0]}/{splits[1]}/{splits[2]}). Training: up to {epochs} epochs, batch "
         f"{batch_size}, Adam lr {lr}, **early stopping patience {PATIENCE}** (stops when "
         f"val loss worsens for 4 consecutive epochs and **restores the current-4th-best "
         "weights**), cosine LR, grad-clip 1.0. `hybrid_jev` also trained with the "
         f"**JEV spectral-consistency regulariser** alpha={jev_alpha} (align learned "
         "token-Gram eigen-weights with the classical EVD spectrum `eig(M^H M)`)."),
        "",
        "## 1. Results (test split, best/restored weights)",
        "",
        "| setup | params (M) | lat (ms) | best epoch | early-stop | rate R² | rate MAE | M cos | M nMSE | M R² |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['setup']} | {r['parameters']/1e6:.3f} | {r['latency_ms']:.2f} | "
            f"{r['epochs_reported']} | {'yes' if r['early_stopped'] else 'no'} | "
            f"{r['test_rate_r2']:.4f} | {r['test_rate_mae']:.3f} | {r['test_M_cos']:.4f} | "
            f"{r['test_M_n_mse']:.4f} | {r['test_M_r2']:.4f} |")
    lines += [
        "",
        "## 2. Validation results (what early stopping optimised)",
        "",
        "| setup | best epoch | val rate R² | val M cos | val M nMSE |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['setup']} | {r['epochs_reported']} | {r['val_rate_r2']:.4f} | "
                     f"{r['val_M_cos']:.4f} | {r['val_M_n_mse']:.4f} |")
    lines += [
        "",
        "## 3. Reasoning per config + prediction vs actual (test)",
        "",
        "| setup | reasoning (design + why) | predicted rate R² / M cos | actual rate R² / M cos |",
        "|---|---|---|---|",
    ]
    for r in rows:
        pred_r, pred_m = PREDICTED_RANGE[r["setup"]]
        lines.append(f"| {r['setup']} | {REASONING[r['setup']]} | {pred_r} / {pred_m} | "
                     f"{r['test_rate_r2']:.3f} / {r['test_M_cos']:.3f} |")
    lines += [
        "",
        "## 4. How to reproduce",
        "",
        "```bash",
        "python3 scripts/train_setups_full.py --cache data/processed/mimo64_5000.npz \\",
        "    --epochs 35 --patience 4 --batch-size 64 --threads 8 \\",
        "    --jev-alpha 0.1 --outdir search_results/mimo64_setups_full",
        "```",
        "",
        "Per-setup learning curves: `search_results/mimo64_setups_full/<name>.csv`.",
        "Best weights: `search_results/mimo64_setups_full/checkpoints/<name>.pt`.",
        "",
    ]
    (outdir / "results.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/processed/mimo64_5000.npz")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--jev-alpha", type=float, default=0.1, help="JEV spectral-consistency weight")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--setups", nargs="*", default=list(SETUP_VARIANTS.keys()))
    p.add_argument("--outdir", default="search_results/mimo64_setups_full")
    args = p.parse_args()

    cache = load_cache(args.cache)
    print(f"cache: {args.cache}  X={cache['X'].shape} "
          f"train={len(cache['train'])} val={len(cache['val'])} "
            f"test={len(cache['test'])}")
    rows, outdir = run(cache, args.epochs, args.batch_size, args.lr, "cpu", args.threads,
                       args.setups, args.outdir, args.jev_alpha, args.seed)
    write_md(rows, outdir, args.epochs, args.batch_size, args.lr, args.jev_alpha,
             args.seed, cache["X"].shape[0], (len(cache["train"]), len(cache["val"]), len(cache["test"])))
    (outdir / "results.json").write_text(json.dumps(
        {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
         "lr": args.lr, "jev_alpha": args.jev_alpha, "seed": args.seed,
         "samples": int(cache["X"].shape[0]), "results": rows}, indent=2))
    print(f"\nsaved {outdir / 'results.json'} and {outdir / 'results.md'}")


if __name__ == "__main__":
    main()