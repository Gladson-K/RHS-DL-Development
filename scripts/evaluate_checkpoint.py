#!/usr/bin/env python3
"""Standalone evaluation of a saved RHS-DL NAS checkpoint.

Loads `checkpoints/mimo64_nas_best_run3.pt`, rebuilds the architecture, and
evaluates it on the held-out TEST split (and val, for reference) with the
project's own `rhs_beamforming.nas.real_trainer.evaluate`.

Usage:
    python3 scripts/evaluate_checkpoint.py \
        --checkpoint checkpoints/mimo64_nas_best_run3.pt \
        --cache data/processed/mimo64_5000.npz \
        --out results/mimo64_nas_best_run3_eval.json
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
import torch

from rhs_beamforming.nas.real_data import build_loaders, load_cache
from rhs_beamforming.nas.real_space import Architecture
from rhs_beamforming.nas.real_trainer import evaluate


def build_model(arch: Architecture, input_dim: int, output_dim: int) -> torch.nn.Module:
    from rhs_beamforming.nas.real_space import build_model as _build
    return _build(arch)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/mimo64_nas_best_run3.pt")
    p.add_argument("--cache", default="data/processed/mimo64_5000.npz")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="results/mimo64_nas_best_run3_eval.json")
    p.add_argument("--baseline", default="search_results/mimo64_nas/architectures/best.json",
                   help="reference metrics json from the NAS search, for comparison")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    cache = load_cache(args.cache)
    loaders = build_loaders(cache, batch_size=args.batch_size)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = Architecture.from_dict(payload["arch"])
    rate_mean = float(payload["rate_mean"])
    rate_std = float(payload["rate_std"])

    model = build_model(arch, int(cache["X"].shape[1]), int(cache["Y"].shape[1]))
    model.load_state_dict(payload["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())

    print(f"checkpoint : {ckpt_path}")
    print(f"architecture: {arch.name}")
    print(f"parameters : {n_params}")

    t0 = time.time()
    test_metrics = evaluate(model, loaders["test"], rate_mean, rate_std)
    test_metrics["parameters"] = n_params
    val_metrics = evaluate(model, loaders["val"], rate_mean, rate_std)
    eval_secs = time.time() - t0

    out = {
        "checkpoint": str(ckpt_path),
        "architecture": arch.to_dict(),
        "architecture_name": arch.name,
        "test": test_metrics,
        "val": val_metrics,
        "parameters": n_params,
        "eval_seconds": eval_secs,
        "n_test_samples": test_metrics.get("n_samples"),
    }

    base_path = Path(args.baseline)
    if base_path.exists():
        base = json.loads(base_path.read_text())
        # best.json stores the NAS search's own metrics for the same architecture
        # (same val split, same arch) -> the apples-to-apples reference.
        base_metrics = base.get("metrics", base.get("test", base))
        out["baseline_reference"] = {
            "source": str(base_path),
            "split": "val",
            "metrics": base_metrics,
        }
        out["delta_vs_baseline"] = {
            "split": "val",
            "rate_r2": val_metrics["rate"]["r2"] - base_metrics.get("rate", {}).get("r2", 0.0),
            "M_cos": val_metrics["M"]["cos"] - base_metrics.get("M", {}).get("cos", 0.0),
            "M_n_mse": val_metrics["M"]["n_mse"] - base_metrics.get("M", {}).get("n_mse", 0.0),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
