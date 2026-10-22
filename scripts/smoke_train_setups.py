#!/usr/bin/env python3
"""Smoke-train the five RHS architecture setups and report metrics.

Trains each setup from ``nas/setups.py`` for a small number of epochs on the
prepared 64x64 cache, evaluates it, benchmarks latency, and writes a JSON +
Markdown results file. Intended for fast, CPU-friendly comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn

from rhs_beamforming.nas.real_data import build_loaders, load_cache
from rhs_beamforming.nas.real_trainer import E, benchmark_latency, evaluate
from rhs_beamforming.nas.setups import SETUP_VARIANTS, build_setup_by_name


def _set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_setup(model, train_loader, val_loader, epochs, lr, rate_mean, rate_std,
                device="cpu", grad_clip=1.0, quiet=False):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 1e-2)
    lossfn = nn.MSELoss()
    rate_std_t = torch.tensor(rate_std, dtype=torch.float32, device=device)
    rate_mean_t = torch.tensor(rate_mean, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
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
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        val = evaluate(model, val_loader, rate_mean, rate_std, device=device)
        val_loss = val["M"]["mse"] + val["rate"]["mse"] / max(rate_std ** 2, 1e-9)
        history.append({"epoch": epoch + 1, "train_loss": running / max(nb, 1), "val_loss": val_loss})
        if not quiet:
            print(f"    epoch {epoch+1:2d} train={running/max(nb,1):.4f} "
                  f"val={val_loss:.4f} rate_r2={val['rate']['r2']:.4f}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history


def run(cache, epochs, batch_size, lr, device, threads, setups, output):
    if threads:
        torch.set_num_threads(threads)
    loaders = build_loaders(cache, batch_size=batch_size)
    rate_mean = float(cache["rate_mean"][0])
    rate_std = float(cache["rate_std"][0])

    rows = []
    for name in setups:
        print(f"\n=== setup: {name} ===")
        _set_seed(42)
        model = build_setup_by_name(name)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        train_setup(model, loaders["train"], loaders["val"], epochs, lr, rate_mean, rate_std, device=device)
        train_s = time.time() - t0
        metrics = evaluate(model, loaders["val"], rate_mean, rate_std, device=device)
        latency = benchmark_latency(model, input_dim=10, device=device)
        row = {
            "setup": name,
            "parameters": params,
            "latency_ms": latency,
            "train_seconds": train_s,
            "epochs": epochs,
            "rate_r2": metrics["rate"]["r2"],
            "rate_mae": metrics["rate"]["mae"],
            "M_cos": metrics["M"]["cos"],
            "M_n_mse": metrics["M"]["n_mse"],
            "M_r2": metrics["M"]["r2"],
            "M_real_mse": metrics["M_real"]["mse"],
            "M_imag_mse": metrics["M_imag"]["mse"],
        }
        rows.append(row)
        print(f"  params={params/1e6:.3f}M lat={latency:.2f}ms t={train_s:.0f}s "
              f"rate_r2={row['rate_r2']:.4f} M_cos={row['M_cos']:.4f} nMSE={row['M_n_mse']:.4f}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"epochs": epochs, "batch_size": batch_size, "lr": lr, "samples": int(cache["X"].shape[0]),
         "train": len(cache["train"]), "val": len(cache["val"]), "results": rows}, indent=2))

    md = ["| setup | params (M) | latency (ms) | rate R² | rate MAE | M cos | M nMSE | M R² |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['setup']} | {r['parameters']/1e6:.3f} | {r['latency_ms']:.2f} | "
            f"{r['rate_r2']:.4f} | {r['rate_mae']:.3f} | {r['M_cos']:.4f} | "
            f"{r['M_n_mse']:.4f} | {r['M_r2']:.4f} |")
    (output.with_suffix(".md")).write_text("\n".join(md) + "\n")
    print(f"\nsaved {output} and {output.with_suffix('.md')}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/processed/mimo64_5000.npz")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--setups", nargs="*", default=list(SETUP_VARIANTS.keys()))
    p.add_argument("--output", default="search_results/mimo64_setups/results.json")
    args = p.parse_args()

    cache = load_cache(args.cache)
    print(f"cache: {args.cache}  X={cache['X'].shape} "
          f"train={len(cache['train'])} val={len(cache['val'])} test={len(cache['test'])}")
    run(cache, args.epochs, args.batch_size, args.lr, "cpu", args.threads, args.setups, args.output)


if __name__ == "__main__":
    main()
