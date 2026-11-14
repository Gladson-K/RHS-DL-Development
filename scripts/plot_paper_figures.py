#!/usr/bin/env python3
"""Generate every figure used in the IEEE paper.

Sources
-------
* ``search_results/mimo64_setups_full/*.csv``  - 5-setup learning curves (5K)
* ``search_results/mimo64_full_data/*``        - hybrid_jev full 15K run
* ``search_results/nas_study/<run>/metrics``   - NAS candidates + Pareto front
* ``search_results/nas_full_data/*``           - NAS winner retrained on 15K
* ``data/processed/mimo64_full``               - GP ground truth (M, rate)

Colour convention for GP comparisons: GP = red, model = blue.

Outputs (PNG 300 dpi + PDF) in ``paper/figures``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

E = 12800
SETUPS = ["mlp", "gnn", "transformer", "hybrid", "hybrid_jev"]
MODEL_COLORS = {
    "mlp": "#4c72b0",
    "gnn": "#dd8452",
    "transformer": "#55a868",
    "hybrid": "#c44e52",
    "hybrid_jev": "#8172b3",
    "nas_best": "#000000",
}
GP_COLOR = "#d62728"
MODEL_BLUE = "#1f77b4"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def _read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
    return rows


# --------------------------------------------------------------------------- #
# Learning-curve figures
# --------------------------------------------------------------------------- #
def fig_learning_curves():
    curves = {}
    for s in SETUPS:
        p = ROOT / f"search_results/mimo64_setups_full/{s}.csv"
        if p.exists():
            curves[s] = _read_csv(p)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for s, rows in curves.items():
        ep = [r["epoch"] for r in rows]
        axes[0].plot(ep, [r["train_loss"] for r in rows], color=MODEL_COLORS[s], lw=1.6, label=s)
        axes[1].plot(ep, [r["val_loss"] for r in rows], color=MODEL_COLORS[s], lw=1.6, label=s)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=2)
    _save(fig, "fig_training_loss")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for s, rows in curves.items():
        ep = [r["epoch"] for r in rows]
        axes[0].plot(ep, [r["rate_r2"] for r in rows], color=MODEL_COLORS[s], lw=1.6, label=s)
        axes[1].plot(ep, [r["M_cos"] for r in rows], color=MODEL_COLORS[s], lw=1.6, label=s)
    axes[0].set_title("Validation rate R²")
    axes[1].set_title("Validation M cosine")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R²")
    axes[1].set_ylabel("cosine")
    axes[0].legend(ncol=2)
    _save(fig, "fig_train_val_metrics")


def fig_test_bars():
    path = ROOT / "search_results/mimo64_setups_full/results.json"
    if not path.exists():
        return
    res = {r["setup"]: r for r in json.loads(path.read_text())["results"]}
    names = [s for s in SETUPS if s in res]
    x = np.arange(len(names))
    metrics = [
        ("test_rate_r2", "Rate R²"),
        ("test_M_cos", "M cosine"),
        ("test_M_r2", "M R²"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, label) in zip(axes, metrics):
        vals = [res[n][key] for n in names]
        bars = ax.bar(x, vals, color=[MODEL_COLORS[n] for n in names])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)
    _save(fig, "fig_test_bars")


def fig_5k_vs_15k():
    p5 = ROOT / "search_results/mimo64_setups_full/results.json"
    p15 = ROOT / "search_results/mimo64_full_data/results.json"
    if not (p5.exists() and p15.exists()):
        return
    r5 = {r["setup"]: r for r in json.loads(p5.read_text())["results"]}["hybrid_jev"]
    r15 = json.loads(p15.read_text())
    keys = [("rate_r2", "Rate R²"), ("M_cos", "M cosine"), ("M_n_mse", "M nMSE"), ("M_r2", "M R²")]
    x = np.arange(len(keys))
    v5 = [r5[f"test_{k}"] for k, _ in keys]
    v15 = [r15[f"test_{k}"] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(x - 0.2, v5, 0.4, label="5K cache", color="#9ecae1")
    ax.bar(x + 0.2, v15, 0.4, label="15K full", color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys])
    ax.set_ylabel("score")
    ax.set_title("hybrid_jev: 5K vs full 15K")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    _save(fig, "fig_5k_vs_15k")


# --------------------------------------------------------------------------- #
# NAS search figures
# --------------------------------------------------------------------------- #
def _run_dirs():
    base = ROOT / "search_results/nas_study"
    if not base.exists():
        return {}
    out = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "metrics/candidates.csv").exists():
            out[d.name] = d
    return out


def fig_nas_search():
    runs = _run_dirs()
    if not runs:
        return
    markers = {"random_b4": "o", "random_b8": "s", "evolution": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, d in runs.items():
        rows = list(csv.DictReader(open(d / "metrics/candidates.csv")))
        if not rows:
            continue
        params = [float(r["parameters"]) / 1e6 for r in rows]
        rate = [float(r["rate_r2"]) for r in rows]
        cos = [float(r["M_cos"]) for r in rows]
        lat = [float(r["latency_ms"]) for r in rows]
        score = [float(r["score"]) for r in rows]
        mk = markers.get(name, "o")
        axes[0].scatter(params, rate, c=score, cmap="viridis", marker=mk, label=name, s=34)
        axes[1].scatter(lat, cos, c=score, cmap="magma", marker=mk, label=name, s=34)
        axes[2].scatter(lat, score, marker=mk, label=name, s=34)
    axes[0].set_xlabel("parameters (M)")
    axes[0].set_ylabel("rate R²")
    axes[0].set_title("Accuracy vs. model size")
    axes[1].set_xlabel("latency (ms)")
    axes[1].set_ylabel("M cosine")
    axes[1].set_title("Beamformer fidelity vs. latency")
    axes[2].set_xlabel("latency (ms)")
    axes[2].set_ylabel("multi-objective score")
    axes[2].set_title("Score vs. latency")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    _save(fig, "fig_nas_search")


def fig_budget_comparison():
    runs = _run_dirs()
    if not runs:
        return
    names, best = [], []
    for name, d in runs.items():
        bj = d / "metrics/best.json"
        if not bj.exists():
            continue
        doc = json.loads(bj.read_text())
        m = doc["metrics"]
        names.append(name)
        best.append({
            "score": m["score"], "rate_r2": m["rate_r2"], "M_cos": m["beamforming_similarity"],
            "latency": m["latency_ms"], "params": m["parameters"] / 1e6,
        })
    if not names:
        return
    keys = [("score", "Score"), ("rate_r2", "Rate R²"), ("M_cos", "M cosine")]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    palette = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    for ax, (key, label) in zip(axes, keys):
        vals = [b[key] for b in best]
        ax.bar(x, vals, color=palette[: len(names)])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    _save(fig, "fig_budget_comparison")


# --------------------------------------------------------------------------- #
# GP comparison (inference)
# --------------------------------------------------------------------------- #
def _load_models():
    """Return {label: (model, kind)} for available checkpoints."""
    import torch
    from rhs_beamforming.nas.setups import build_setup_by_name
    from rhs_beamforming.nas.real_space import Architecture, build_model

    out = {}
    hj = ROOT / "search_results/mimo64_full_data/checkpoints/hybrid_jev.pt"
    if hj.exists():
        m = build_setup_by_name("hybrid_jev")
        m.load_state_dict(torch.load(hj, map_location="cpu"))
        m.eval()
        out["hybrid_jev"] = m

    nas_dirs = sorted((ROOT / "search_results/nas_full_data").glob("*/results.json"))
    for rj in nas_dirs:
        info = json.loads(rj.read_text())
        ckpt = rj.parent / "checkpoints/nas_best.pt"
        if not ckpt.exists():
            continue
        m = build_model(Architecture.from_dict(info["architecture"]))
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        out[f"nas_best:{rj.parent.name}"] = m
    return out


def _predict(model, X, indices, batch=64, rate_mean=51.794, rate_std=12.298):
    import torch
    preds = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            idx = indices[start:start + batch]
            x = torch.as_tensor(np.asarray(X[idx]), dtype=torch.float32)
            out = model(x).numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)


def fig_gp_comparison():
    import torch

    cache = ROOT / "data/processed/mimo64_full"
    if not (cache / "Y.npy").exists():
        print("  (skip GP figures: full cache not found)")
        return
    meta = json.loads((cache / "meta.json").read_text())
    rate_mean, rate_std = float(meta["rate_mean"]), float(meta["rate_std"])
    X = np.load(cache / "X.npy", mmap_mode="r")
    te = np.load(cache / "split_test.npy")

    models = _load_models()
    if not models:
        print("  (skip GP figures: no checkpoints)")
        return

    # ground truth in caller order
    Y = np.load(cache / "Y.npy", mmap_mode="r")
    gt_parts = []
    for start in range(0, len(te), 128):
        idx = te[start:start + 128]
        gt_parts.append(np.asarray(Y[idx]))
    gt = np.concatenate(gt_parts, axis=0)
    gp_re, gp_im, gp_rate = gt[:, :E], gt[:, E:2 * E], gt[:, 2 * E]

    n_show = min(300, len(te))
    for label, model in models.items():
        p = _predict(model, X, te, rate_mean=rate_mean, rate_std=rate_std)
        pred_rate = p[:, 2 * E] * rate_std + rate_mean
        pred_re, pred_im = p[:, :E], p[:, E:2 * E]

        # sum-rate parity + per-sample line ---------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        lo = min(gp_rate.min(), pred_rate.min())
        hi = max(gp_rate.max(), pred_rate.max())
        axes[0].scatter(gp_rate, pred_rate, s=6, color=MODEL_BLUE, alpha=0.45)
        axes[0].plot([lo, hi], [lo, hi], color=GP_COLOR, lw=1.6)
        axes[0].set_xlabel("GP sum rate")
        axes[0].set_ylabel("model sum rate")
        axes[0].set_title("Sum-rate parity")
        axes[0].grid(alpha=0.3)
        axes[1].plot(np.arange(n_show), gp_rate[:n_show], color=GP_COLOR, lw=1.4, label="GP")
        axes[1].plot(np.arange(n_show), pred_rate[:n_show], color=MODEL_BLUE, lw=1.4, label="model")
        axes[1].set_xlabel("test sample")
        axes[1].set_ylabel("sum rate")
        axes[1].set_title(f"{label}: per-sample sum rate")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        _save(fig, f"fig_gp_sum_rate_{label.replace(':','_')}")

        # representative beamforming vector real/imag -----------------------
        sel = 0
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        k = 256
        axes[0].plot(gp_re[sel, :k], color=GP_COLOR, lw=1.2, label="GP")
        axes[0].plot(pred_re[sel, :k], color=MODEL_BLUE, lw=1.2, label="model")
        axes[0].set_title("Beamforming vector — real")
        axes[1].plot(gp_im[sel, :k], color=GP_COLOR, lw=1.2, label="GP")
        axes[1].plot(pred_im[sel, :k], color=MODEL_BLUE, lw=1.2, label="model")
        axes[1].set_title("Beamforming vector — imaginary")
        for ax in axes:
            ax.set_xlabel("element index")
            ax.set_ylabel("value")
            ax.legend()
            ax.grid(alpha=0.3)
        _save(fig, f"fig_gp_M_real_imag_{label.replace(':','_')}")

        # magnitude distribution + CDF --------------------------------------
        gp_mag = np.sqrt(gp_re ** 2 + gp_im ** 2).ravel()[::11]
        pr_mag = np.sqrt(pred_re ** 2 + pred_im ** 2).ravel()[::11]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        bins = np.linspace(0, np.percentile(gp_mag, 99.5), 60)
        axes[0].hist(gp_mag, bins=bins, density=True, color=GP_COLOR, alpha=0.45, label="GP")
        axes[0].hist(pr_mag, bins=bins, density=True, color=MODEL_BLUE, alpha=0.45, label="model")
        axes[0].set_xlabel("|M|")
        axes[0].set_ylabel("density")
        axes[0].set_title("Beamforming magnitude distribution")
        axes[0].legend()
        for arr, c in ((gp_mag, GP_COLOR), (pr_mag, MODEL_BLUE)):
            x = np.sort(arr)
            axes[1].plot(x, np.linspace(0, 1, len(x)), color=c, lw=1.4)
        axes[1].set_xlabel("|M|")
        axes[1].set_ylabel("CDF")
        axes[1].set_title("Magnitude CDF")
        for ax in axes:
            ax.grid(alpha=0.3)
        _save(fig, f"fig_gp_magnitude_{label.replace(':','_')}")


def main():
    print("Generating paper figures ...")
    fig_learning_curves()
    fig_train_val_metrics()
    fig_test_bars()
    fig_5k_vs_15k()
    fig_nas_search()
    fig_budget_comparison()
    fig_gp_comparison()
    print(f"All figures in {FIG}")


if __name__ == "__main__":
    main()
