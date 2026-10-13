"""Multi-model NAS search over the real 64x64 config -> (M, rate) task.

Supports:
    * ``random``         - sample N architectures
    * ``evolution``      - population + mutation/crossover
    * successive halving - train many cheaply, promote the best to longer runs

All candidates are trained on CPU and ranked with the multi-objective score
(rate, M-similarity, latency, parameters). The best architecture is exported
to ``best.json`` / ``best.yaml`` and every candidate is written to
``candidates.csv`` plus a Pareto frontier.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ..data.mimo64 import Mimo64TorchDataset
from .real_objective import ScoreWeights, compute_score
from .real_space import Architecture, RealSearchSpace
from .real_trainer import TrainConfig, train_candidate


@dataclass
class SearchConfig:
    strategy: str = "random"  # random | evolution
    num_candidates: int = 12
    population: int = 8
    generations: int = 2
    budget_epochs: int = 8
    final_epochs: int = 30
    top_k: int = 3
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    m_weight: float = 1.0
    rate_weight: float = 1.0
    early_stop_patience: int = 6
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 0
    halving: bool = True
    output_dir: str = "search_results/mimo64_nas"
    score: ScoreWeights = field(default_factory=ScoreWeights)


def _subset_loader(cache, idx, batch_size, shuffle):
    M = cache.get("M")
    M_idx = M[idx] if M is not None else None
    ds = Mimo64TorchDataset(cache["X"][idx], cache["Y"][idx], M_idx, cache["rate"][idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=(shuffle and len(ds) > batch_size))


class MultiModelSearch:
    def __init__(self, space: RealSearchSpace, cache: dict, cfg: SearchConfig):
        self.space = space
        self.cache = cache
        self.cfg = cfg
        self.rate_mean = float(cache["rate_mean"][0])
        self.rate_std = float(cache["rate_std"][0])
        self.records: list[dict] = []
        self.rng = __import__("random").Random(cfg.seed)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_lines: list[str] = []

    def _log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_lines.append(line)
        print(line, flush=True)

    def _loaders_for(self, train_idx, val_idx):
        tr = _subset_loader(self.cache, train_idx, self.cfg.batch_size, True)
        va = _subset_loader(self.cache, val_idx, self.cfg.batch_size, False)
        return tr, va

    def train_eval(self, arch: Architecture, epochs: int, train_idx, val_idx, tag=""):
        tr, va = self._loaders_for(train_idx, val_idx)
        tcfg = TrainConfig(
            epochs=epochs,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            optimizer=self.cfg.optimizer,
            m_weight=self.cfg.m_weight,
            rate_weight=self.cfg.rate_weight,
            early_stop_patience=self.cfg.early_stop_patience,
            seed=self.cfg.seed,
            device=self.cfg.device,
            torch_threads=self.cfg.torch_threads,
        )
        t0 = time.time()
        model, out = train_candidate(arch, tr, va, tcfg, self.rate_mean, self.rate_std, quiet=True)
        metrics = out["metrics"]
        metrics["score"] = float(compute_score(metrics, self.cfg.score))
        metrics["train_seconds"] = time.time() - t0
        metrics["tag"] = tag
        self.records.append(metrics)
        self._log(
            f"{tag:>4} {arch.name[:48]:48s} score={metrics['score']:.4f} "
            f"rate_r2={metrics['rate']['r2']:.4f} M_cos={metrics['M']['cos']:.4f} "
            f"nMSE={metrics['M']['n_mse']:.4f} params={metrics['parameters']/1e6:.2f}M "
            f"lat={metrics['latency_ms']:.2f}ms t={metrics['train_seconds']:.0f}s"
        )
        return model, metrics

    # -- strategies ------------------------------------------------------- #
    def _sample(self, n):
        return [self.space.sample(self.rng) for _ in range(n)]

    def run(self) -> dict:
        cfg = self.cfg
        train_idx = self.cache["train"]
        val_idx = self.cache["val"]
        self._log(f"search start: strategy={cfg.strategy} n={cfg.num_candidates} halving={cfg.halving}")

        if cfg.strategy == "evolution":
            population = self._sample(cfg.population)
            for gen in range(cfg.generations):
                self._log(f"generation {gen}")
                scored = []
                for a in population:
                    _, m = self.train_eval(a, cfg.budget_epochs, train_idx, val_idx, tag=f"g{gen}")
                    scored.append((m["score"], a, m))
                scored.sort(key=lambda x: x[0], reverse=True)
                elites = scored[: max(2, cfg.population // 4)]
                if gen < cfg.generations - 1:
                    children = []
                    while len(children) < cfg.population - len(elites):
                        parent = self.rng.choice(elites)[1]
                        if self.rng.random() < 0.5:
                            children.append(self.space.mutate(parent, self.rng))
                        else:
                            other = self.rng.choice(elites)[1]
                            children.append(self.space.crossover(parent, other, self.rng))
                    population = [a for _, a, _ in elites] + children
        else:
            # random + successive halving
            candidates = self._sample(cfg.num_candidates)
            active = candidates
            if cfg.halving:
                rungs = [
                    (0.25, max(1, cfg.budget_epochs // 2)),
                    (0.5, cfg.budget_epochs),
                ]
                for frac, ep in rungs:
                    n_tr = max(1, int(len(train_idx) * frac))
                    sub_tr = np.sort(self.rng.sample(list(train_idx), n_tr))
                    scored = []
                    for a in active:
                        _, m = self.train_eval(a, ep, sub_tr, val_idx, tag=f"sh{int(frac*100)}")
                        scored.append((m["score"], a))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    active = [a for _, a in scored[: max(cfg.top_k, len(scored) // 2)]]
                active = active[: cfg.top_k]

        # final full training on promoted candidates
        self._log("final training on promoted candidates")
        finalists = []
        for a in active:
            _, m = self.train_eval(a, cfg.final_epochs, train_idx, val_idx, tag="final")
            finalists.append((m["score"], a, m))
        finalists.sort(key=lambda x: x[0], reverse=True)

        result = self._finalize(finalists)
        self._log(f"best: {finalists[0][1].name} score={finalists[0][0]:.4f}")
        return result

    # -- reporting -------------------------------------------------------- #
    def _pareto(self):
        pts = self.records
        front = []
        for i, a in enumerate(pts):
            dominated = False
            for j, b in enumerate(pts):
                if i == j:
                    continue
                if (b["rate"]["r2"] >= a["rate"]["r2"] and b["M"]["cos"] >= a["M"]["cos"]
                        and b["parameters"] <= a["parameters"] and b["latency_ms"] <= a["latency_ms"]
                        and (b["rate"]["r2"] > a["rate"]["r2"] or b["M"]["cos"] > a["M"]["cos"]
                             or b["parameters"] < a["parameters"] or b["latency_ms"] < a["latency_ms"])):
                    dominated = True
                    break
            if not dominated:
                front.append(a)
        return front

    def _finalize(self, finalists):
        out = self.output_dir
        arch_dir = out / "architectures"
        metrics_dir = out / "metrics"
        plots_dir = out / "plots"
        for d in (arch_dir, metrics_dir, plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        # candidates csv
        keys = ["arch_name", "score", "parameters", "latency_ms", "train_seconds"]
        with open(metrics_dir / "candidates.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(keys + ["rate_r2", "rate_mae", "M_cos", "M_n_mse", "M_r2", "tag",
                                "hidden_layers", "activation", "dropout", "batchnorm", "residual", "head"])
            for m in self.records:
                arch = m["arch"]
                wr.writerow(
                    [m.get(k, "") for k in keys]
                    + [m["rate"]["r2"], m["rate"]["mae"], m["M"]["cos"], m["M"]["n_mse"], m["M"]["r2"], m.get("tag", "")]
                    + [str(arch["hidden_layers"]), arch["activation"], arch["dropout"], arch["batchnorm"],
                       arch["residual"], arch["head"]]
                )

        # pareto
        front = self._pareto()
        with open(metrics_dir / "pareto.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["arch_name", "score", "rate_r2", "M_cos", "parameters", "latency_ms"])
            for m in front:
                wr.writerow([m["arch_name"], m["score"], m["rate"]["r2"], m["M"]["cos"], m["parameters"], m["latency_ms"]])

        # save best + top-k archs
        best_arch, best_metrics = finalists[0][1], finalists[0][2]
        (arch_dir / "best.json").write_text(json.dumps(
            {"architecture": best_arch.to_dict(), "metrics": best_metrics, "name": best_arch.name}, indent=2))
        for i, (score, arch, m) in enumerate(finalists):
            (arch_dir / f"arch_{i:04d}.json").write_text(json.dumps(
                {"architecture": arch.to_dict(), "metrics": m, "name": arch.name}, indent=2))

        best_payload = {
            "architecture": best_arch.to_dict(),
            "metrics": {
                "score": best_metrics["score"],
                "rate_r2": best_metrics["rate"]["r2"],
                "rate_mse": best_metrics["rate"]["mse"],
                "beamforming_similarity": best_metrics["M"]["cos"],
                "M_n_mse": best_metrics["M"]["n_mse"],
                "latency_ms": best_metrics["latency_ms"],
                "parameters": best_metrics["parameters"],
            },
        }
        (metrics_dir / "best.json").write_text(json.dumps(best_payload, indent=2))
        (out / "best.yaml").write_text(_to_yaml(best_payload))
        (out / "search.log").write_text("\n".join(self.log_lines) + "\n")

        self._plots(front, plots_dir)
        return {
            "best": best_payload,
            "pareto": front,
            "records": self.records,
            "output_dir": str(out),
        }

    def _plots(self, front, plots_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return
        recs = self.records
        rate = [m["rate"]["r2"] for m in recs]
        cos = [m["M"]["cos"] for m in recs]
        params = [m["parameters"] / 1e6 for m in recs]
        lat = [m["latency_ms"] for m in recs]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(params, rate, c=[m["score"] for m in recs], cmap="viridis")
        ax[0].set_xlabel("parameters (M)"); ax[0].set_ylabel("rate R2"); ax[0].set_title("rate vs params")
        ax[1].scatter(lat, cos, c=[m["score"] for m in recs], cmap="magma")
        ax[1].set_xlabel("latency (ms)"); ax[1].set_ylabel("M similarity (cos)"); ax[1].set_title("similarity vs latency")
        fig.tight_layout(); fig.savefig(plots_dir / "search_overview.png", dpi=120); plt.close(fig)
        if front:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter([m["latency_ms"] for m in front], [m["rate"]["r2"] for m in front], marker="*", s=120)
            ax.set_xlabel("latency (ms)"); ax.set_ylabel("rate R2"); ax.set_title("Pareto frontier")
            fig.tight_layout(); fig.savefig(plots_dir / "pareto_frontier.png", dpi=120); plt.close(fig)


def _to_yaml(obj, indent=0) -> str:
    lines = []
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_to_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}- {v}")
    else:
        lines.append(f"{pad}{obj}")
    return "\n".join(lines)
