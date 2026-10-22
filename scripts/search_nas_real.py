#!/usr/bin/env python3
"""Run the multi-model NAS search on the real 64x64 MIMO dataset (CPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhs_beamforming.nas.real_data import load_cache, prepare_cache
from rhs_beamforming.nas.real_objective import ScoreWeights
from rhs_beamforming.nas.real_search import MultiModelSearch, SearchConfig
from rhs_beamforming.nas.real_space import RealSearchSpace

_DEFAULTS = {
    "cache": "data/processed/mimo64_5000.npz",
    "samples": 5000,
    "strategy": "random",
    "num_candidates": 10,
    "population": 8,
    "generations": 2,
    "budget_epochs": 6,
    "final_epochs": 25,
    "top_k": 3,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "threads": 0,
    "halving": True,
    "output": "search_results/mimo64_nas",
}


def _load_yaml_config(path: str) -> dict:
    import yaml

    doc = yaml.safe_load(Path(path).read_text()) or {}
    search = doc.get("search", doc)
    training = search.get("training", {})
    out = {
        "strategy": search.get("strategy"),
        "num_candidates": search.get("num_candidates"),
        "population": search.get("population"),
        "generations": search.get("generations"),
        "budget_epochs": search.get("budget_epochs"),
        "final_epochs": search.get("final_epochs"),
        "top_k": search.get("top_k"),
        "output": search.get("output_dir"),
        "seed": search.get("seed"),
        "halving": search.get("halving"),
        "cache": search.get("cache"),
    }
    out.update({k: training.get(k) for k in ("batch_size", "lr", "weight_decay")})
    out["threads"] = training.get("torch_threads")
    out["score"] = search.get("score", {})
    return {k: v for k, v in out.items() if v is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML config (configs/nas/real_multisearch.yaml)")
    p.add_argument("--cache", default=None)
    p.add_argument("--h5", default=None, help="HDF5 path (used to prepare cache if missing)")
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--strategy", default=None, choices=["random", "evolution"])
    p.add_argument("--num-candidates", type=int, default=None)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--budget-epochs", type=int, default=None)
    p.add_argument("--final-epochs", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--halving", dest="halving", action="store_true", default=None)
    p.add_argument("--no-halving", dest="halving", action="store_false")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    values = dict(_DEFAULTS)
    if args.config:
        values.update(_load_yaml_config(args.config))
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    values.update(cli)
    score_kwargs = values.get("score", {}) if isinstance(values.get("score"), dict) else {}

    cache_path = Path(values["cache"])
    if not cache_path.exists():
        if not args.h5:
            raise SystemExit(f"cache not found: {cache_path}; pass --h5 to prepare it")
        cache_path = prepare_cache(
            args.h5, n_samples=int(values["samples"]), seed=int(values["seed"]), out_path=cache_path
        )
    cache = load_cache(cache_path)
    print(f"cache: {cache_path}  X={cache['X'].shape}  "
          f"train={len(cache['train'])} val={len(cache['val'])} test={len(cache['test'])}")

    space = RealSearchSpace()
    cfg = SearchConfig(
        strategy=values["strategy"],
        num_candidates=int(values["num_candidates"]),
        population=int(values["population"]),
        generations=int(values["generations"]),
        budget_epochs=int(values["budget_epochs"]),
        final_epochs=int(values["final_epochs"]),
        top_k=int(values["top_k"]),
        batch_size=int(values["batch_size"]),
        lr=float(values["lr"]),
        weight_decay=float(values["weight_decay"]),
        seed=int(values["seed"]),
        torch_threads=int(values["threads"]),
        halving=bool(values["halving"]),
        output_dir=values["output"],
        score=ScoreWeights(**score_kwargs) if score_kwargs else ScoreWeights(),
    )
    result = MultiModelSearch(space, cache, cfg).run()
    print("\n=== BEST ARCHITECTURE ===")
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
