#!/usr/bin/env python3
"""Retrain the best NAS architecture on the full training split and evaluate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from rhs_beamforming.nas.real_data import build_loaders, load_cache
from rhs_beamforming.nas.real_space import Architecture
from rhs_beamforming.nas.real_trainer import TrainConfig, evaluate, train_candidate


def load_arch(path: Path) -> Architecture:
    payload = json.loads(Path(path).read_text())
    return Architecture.from_dict(payload.get("architecture", payload))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/processed/mimo64_5000.npz")
    p.add_argument("--best", default="search_results/mimo64_nas/architectures/best.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/mimo64_nas_best.pt")
    args = p.parse_args()

    cache = load_cache(args.cache)
    arch = load_arch(Path(args.best))
    print(f"architecture: {arch.name}")

    rate_mean = float(cache["rate_mean"][0])
    rate_std = float(cache["rate_std"][0])
    loaders = build_loaders(cache, batch_size=args.batch_size)

    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       seed=args.seed, device="cpu", early_stop_patience=15)
    model, out = train_candidate(arch, loaders["train"], loaders["val"], tcfg, rate_mean, rate_std, quiet=False)

    test_metrics = evaluate(model, loaders["test"], rate_mean, rate_std)
    test_metrics["parameters"] = sum(p.numel() for p in model.parameters())
    print("\n=== TEST METRICS ===")
    print(json.dumps(test_metrics, indent=2, default=str))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch.to_dict(), "state_dict": model.state_dict(),
                "rate_mean": rate_mean, "rate_std": rate_std}, args.out)
    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"architecture": arch.to_dict(), "test": test_metrics,
                                        "val": out["metrics"]}, indent=2, default=str))
    print(f"saved {args.out} and {metrics_path}")


if __name__ == "__main__":
    main()
