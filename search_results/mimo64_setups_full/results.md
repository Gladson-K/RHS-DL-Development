# RHS-HaloBeamGNN — Full-Training Results (5 Setups)

Real 64x64 MIMO cache (5000 samples; train/val/test = 3994/503/503). Training: up to 35 epochs, batch 64, Adam lr 0.001, **early stopping patience 4** (stops when val loss worsens for 4 consecutive epochs and **restores the current-4th-best weights**), cosine LR, grad-clip 1.0. `hybrid_jev` also trained with the **JEV spectral-consistency regulariser** alpha=0.1 (align learned token-Gram eigen-weights with the classical EVD spectrum `eig(M^H M)`).

## 1. Results (test split, best/restored weights)

| setup | params (M) | lat (ms) | best epoch | early-stop | rate R² | rate MAE | M cos | M nMSE | M R² |
|---|---|---|---|---|---|---|---|---|---|
| mlp | 0.998 | 0.24 | 9 | yes | 0.7937 | 4.026 | 0.7406 | 0.4560 | 0.5440 |
| gnn | 3.638 | 4.86 | 13 | yes | 0.7950 | 4.045 | 0.7582 | 0.4258 | 0.5741 |
| transformer | 3.703 | 3.29 | 5 | yes | 0.8071 | 3.997 | 0.7507 | 0.4396 | 0.5604 |
| hybrid | 4.034 | 6.05 | 7 | yes | 0.7948 | 4.088 | 0.7493 | 0.4401 | 0.5599 |
| hybrid_jev | 4.084 | 7.46 | 17 | yes | 0.8011 | 3.968 | 0.7606 | 0.4223 | 0.5776 |

## 2. Validation results (what early stopping optimised)

| setup | best epoch | val rate R² | val M cos | val M nMSE |
|---|---|---|---|---|
| mlp | 9 | 0.7783 | 0.7404 | 0.4579 |
| gnn | 13 | 0.7788 | 0.7598 | 0.4240 |
| transformer | 5 | 0.7754 | 0.7514 | 0.4396 |
| hybrid | 7 | 0.7759 | 0.7506 | 0.4391 |
| hybrid_jev | 17 | 0.7857 | 0.7617 | 0.4208 |

## 3. Reasoning per config + prediction vs actual (test)

| setup | reasoning (design + why) | predicted rate R² / M cos | actual rate R² / M cos |
|---|---|---|---|
| mlp | Tabular residual MLP (256-256-128-64) — the config-to-output path with no structural prior. Baseline. Expected: best latency/params, rate R2 ~0.80, M cos ~0.75 (matches the earlier 40-epoch run). Fastest to converge. | ~0.80 / ~0.75 | 0.794 / 0.741 |
| gnn | HaloBeam-GNN: 16 learned tokens + 2 graph message-passing blocks with trainable adjacency. Structural prior that M is spatially coupled across the array. Expected: better M fidelity than MLP (M cos ~0.76-0.79), same-rate, ~2.9 ms. Good accuracy-latency for a GNN. | ~0.79-0.81 / ~0.76-0.79 | 0.795 / 0.758 |
| transformer | Transformer encoder (2 blocks, 4 heads) over 16 tokens; attention captures long-range element couplings the graph only sees locally. Smoke run already gave the best M cos (0.751) at 3 epochs. Expected winner on M fidelity: M cos ~0.79-0.82, rate R2 ~0.81-0.83 at ~5 ms. | ~0.81-0.83 / ~0.79-0.82 | 0.807 / 0.751 |
| hybrid | GNN + Transformer interleaved (G,T,G,T) — the 'attachable Transformer' variants the user requested. Combines local message passing with global attention; most expressive but heaviest (8.2 ms). Expected between GNN and transformer on M (M cos ~0.77-0.80), best rate stability. | ~0.80-0.81 / ~0.77-0.80 | 0.795 / 0.749 |
| hybrid_jev | GNN + Transformer + differentiable JEV (eigen) decision head, now regularised to follow the classical EVD spectrum eig(M^H M) of the target. The eigen decision should track the true beamformer; the regulariser adds a physics prior and stabilises the head. Expected to reach or beat transformer on M cos (~0.80-0.83) once converged, at ~5.5 ms. | ~0.80-0.82 / ~0.80-0.83 | 0.801 / 0.761 |

## 4. How to reproduce

```bash
python3 scripts/train_setups_full.py --cache data/processed/mimo64_5000.npz \
    --epochs 35 --patience 4 --batch-size 64 --threads 8 \
    --jev-alpha 0.1 --outdir search_results/mimo64_setups_full
```

Per-setup learning curves: `search_results/mimo64_setups_full/<name>.csv`.
Best weights: `search_results/mimo64_setups_full/checkpoints/<name>.pt`.

