"""Training + evaluation for real 64x64 NAS candidates (CPU friendly)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .real_space import ELEMENTS_PER_SAMPLE, Architecture, build_model, count_parameters

E = ELEMENTS_PER_SAMPLE  # 12800


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    grad_clip: float = 1.0
    m_weight: float = 1.0
    rate_weight: float = 1.0
    scheduler: str = "cosine"
    early_stop_patience: int = 8
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 0
    drop_last: bool = True


def _set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


@torch.no_grad()
def evaluate(model: nn.Module, loader, rate_mean: float, rate_std: float, device: str = "cpu") -> dict:
    model.eval()
    se_Mre = se_Mim = se_rate = 0.0
    ae_Mre = ae_Mim = ae_rate = 0.0
    n = 0
    mean_Mre = mean_Mim = 0.0
    sum_sq = 0.0
    dot = 0.0
    norm_pred = 0.0
    norm_true = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x)
        pmre, pMim = out[:, :E], out[:, E : 2 * E]
        prate = out[:, 2 * E] * rate_std + rate_mean
        tmre, tMim = y[:, :E], y[:, E : 2 * E]
        trate = y[:, 2 * E]
        b = x.shape[0]
        n += b
        se_Mre += (pmre - tmre).pow(2).sum().item()
        se_Mim += (pMim - tMim).pow(2).sum().item()
        ae_Mre += (pmre - tmre).abs().sum().item()
        ae_Mim += (pMim - tMim).abs().sum().item()
        se_rate += (prate - trate).pow(2).sum().item()
        ae_rate += (prate - trate).abs().sum().item()
        mean_Mre += tmre.mean().item() * b
        mean_Mim += tMim.mean().item() * b
        # similarity / n_mse on concatenated M re+im
        p = torch.cat([pmre, pMim], dim=1)
        t = torch.cat([tmre, tMim], dim=1)
        dot += (p * t).sum().item()
        norm_pred += p.pow(2).sum().item()
        norm_true += t.pow(2).sum().item()
        sum_sq += t.pow(2).sum().item()

    mean_Mre /= max(n, 1)
    mean_Mim /= max(n, 1)
    # total sum of squares about mean (approx using element means)
    ss_tot = 0.0
    for batch in loader:
        y = batch["y"]
        tmre, tMim = y[:, :E], y[:, E : 2 * E]
        ss_tot += (tmre - mean_Mre).pow(2).sum().item() + (tMim - mean_Mim).pow(2).sum().item()
    total_elems = max(n * 2 * E, 1)
    r2_M = 1.0 - (se_Mre + se_Mim) / max(ss_tot, 1e-9)
    n_mse_M = (se_Mre + se_Mim) / max(sum_sq, 1e-9)
    cos_M = dot / max(np.sqrt(norm_pred) * np.sqrt(norm_true), 1e-9)
    # rate R2
    trate_all = []
    for batch in loader:
        trate_all.append(batch["y"][:, 2 * E])
    trate_all = torch.cat(trate_all)
    r2_rate = 1.0 - se_rate / max(((trate_all - trate_all.mean()).pow(2).sum().item()), 1e-9)
    return {
        "M_real": {"mse": se_Mre / max(n * E, 1), "mae": ae_Mre / max(n * E, 1)},
        "M_imag": {"mse": se_Mim / max(n * E, 1), "mae": ae_Mim / max(n * E, 1)},
        "M": {"mse": (se_Mre + se_Mim) / total_elems, "n_mse": n_mse_M, "r2": r2_M, "cos": cos_M},
        "rate": {"mse": se_rate / max(n, 1), "mae": ae_rate / max(n, 1), "r2": r2_rate},
        "n_samples": n,
    }


def benchmark_latency(model: nn.Module, input_dim: int = 10, device: str = "cpu",
                      warmup: int = 3, runs: int = 10, batch: int = 1) -> float:
    model.eval()
    x = torch.randn(batch, input_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        dt = time.perf_counter() - t0
    return dt * 1000.0 / runs


def train_candidate(
    arch: Architecture,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    rate_mean: float,
    rate_std: float,
    quiet: bool = True,
) -> tuple[nn.Module, dict]:
    """Train one candidate; returns (best model, history/metrics)."""
    if cfg.torch_threads and cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    _set_seed(cfg.seed)
    device = cfg.device
    model = build_model(arch).to(device)
    opt = _make_optimizer(model, cfg)
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 1e-2)
    lossfn = nn.MSELoss()

    rate_std_t = torch.tensor(rate_std, dtype=torch.float32, device=device)
    rate_mean_t = torch.tensor(rate_mean, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_rate_r2": []}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        nb = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            pmre, pMim = out[:, :E], out[:, E : 2 * E]
            prate = out[:, 2 * E]
            tmre, tMim = y[:, :E], y[:, E : 2 * E]
            trate_scaled = (y[:, 2 * E] - rate_mean_t) / rate_std_t
            loss = cfg.m_weight * 0.5 * (lossfn(pmre, tmre) + lossfn(pMim, tMim))
            loss = loss + cfg.rate_weight * lossfn(prate, trate_scaled)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item()
            nb += 1
        if scheduler is not None:
            scheduler.step()
        val = evaluate(model, val_loader, rate_mean, rate_std, device=device)
        val_loss = val["M"]["mse"] + cfg.rate_weight * val["rate"]["mse"] / max(rate_std**2, 1e-9)
        history["train_loss"].append(running / max(nb, 1))
        history["val_loss"].append(val_loss)
        history["val_rate_r2"].append(val["rate"]["r2"])
        if not quiet:
            print(f"    epoch {epoch+1:3d} train={running/max(nb,1):.4f} val={val_loss:.4f} rate_r2={val['rate']['r2']:.4f}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, rate_mean, rate_std, device=device)
    metrics["parameters"] = count_parameters(model)
    metrics["latency_ms"] = benchmark_latency(model, arch.input_dim, device=device)
    metrics["epochs_run"] = len(history["train_loss"])
    metrics["arch"] = arch.to_dict()
    metrics["arch_name"] = arch.name
    history["best_val_loss"] = best_val
    return model, {"metrics": metrics, "history": history}
