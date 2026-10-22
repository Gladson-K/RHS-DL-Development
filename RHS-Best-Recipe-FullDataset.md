# RHS HaloBeam-GNN — Whole-Dataset Training (15K, CPU) — Best NAS Recipe

**Date:** 2026-10-22 · **Branch:** `DL-RHS-HoloBeamGNNNet-dev`
**Recipe:** `hybrid_jev` = GNN + attachable Transformer + differentiable JEV
(eigen) decision head — the best NAS result from the five-setup study.
**Data:** the WHOLE real 64×64 MIMO dataset, **15,000** samples
(train/val/test = 11,988/1,506/1,506; stratified on SNR / frequency / RFno).
**Hardware:** CPU (8 torch threads), ~3.8 GB RAM total → the 1.5 GB target is
memory-mapped, never fully resident.
**Training:** up to 40 epochs, Adam lr 1e-3, batch 64, cosine LR, grad-clip 1.0,
**early stopping patience 5** — when val loss worsens for 5 consecutive epochs,
training stops and the **best (`current_epoch − 5`) weights are restored**.
JEV spectral-consistency regularizer α = 0.1 (align learned token-Gram
eigen-weights with the classical EVD spectrum `eig(MᴴM)` of the target).

## Status: COMPLETE ✔ — weights saved

| artifact | path |
|---|---|
| **best weights** | `search_results/mimo64_full_data/checkpoints/hybrid_jev.pt` (16.3 MB) |
| results | `search_results/mimo64_full_data/results.json` |
| learning curve | `search_results/mimo64_full_data/hybrid_jev.csv` |
| full log | `search_results/mimo64_full_data.log` |

## Final metrics

| split | rate R² | rate MAE | M cos | M nMSE | M R² |
|---|---|---|---|---|---|
| val | 0.8069 | 3.936 | 0.7753 | 0.3991 | 0.6009 |
| **test** | **0.8044** | **4.026** | **0.7772** | **0.3961** | **0.6038** |

- parameters: **4,083,733** (4.084 M) · latency: **8.3 ms** (batch 1, CPU)
- best epoch: **19** of 24 run (early-stopped; weights restored to epoch 19)
- best val_loss 0.2133

### vs the 5K-cache study (same recipe, same test distribution)

| metric | 5K cache | full 15K | Δ |
|---|---|---|---|
| rate R² (test) | 0.801 | **0.804** | +0.003 |
| M cos (test) | 0.761 | **0.777** | **+0.016** |
| M nMSE (test) | 0.422 | **0.396** | **−0.026** |
| M R² (test) | 0.578 | **0.604** | **+0.026** |

Training on the whole dataset **materially improves beamforming-matrix fidelity**
(M cos / nMSE / R²) while rate is essentially saturated (~0.80) — exactly as the
5K analysis predicted (rate is capped by the 10-feature config input; gains come
from the M/JEV side).

## Learning curve (val)

```
epoch  1  rate_r2=0.7575  M_cos=0.7289  M_nMSE=0.4893
epoch  6  rate_r2=0.7997  M_cos=0.7582  M_nMSE=0.4254
epoch 12  rate_r2=0.7976  M_cos=0.7704  M_nMSE=0.4068
epoch 17  rate_r2=0.8065  M_cos=0.7753  M_nMSE=0.3997
epoch 19  rate_r2=0.8069  M_cos=0.7753  M_nMSE=0.3991  ← best (restored)
epoch 21  rate_r2=0.8067  M_cos=0.7764  M_nMSE=0.3980
epoch 24  rate_r2=0.8061  M_cos=0.7775  M_nMSE=0.3955
>> early stop: no improvement for 5 epochs; restoring epoch-5 (best epoch 19) weights
```

## Reproduce

```bash
# 1. extract the real HDF5 (source zip on the D drive)
python3 scripts/extract_mimo64_dataset.py \
    --zip "/mnt/d/Workspace_Glad/64x64_MIMO_Dataset/64x64_MIMO_beamforming_dataset.zip" \
    --dest data/raw --validate

# 2. build the memory-mappable full cache (single sequential HDF5 pass, ~4.5 min)
python3 scripts/build_full_cache.py \
    --h5 data/raw/64x64_MIMO_beamforming_dataset.h5 \
    --outdir data/processed/mimo64_full

# 3. train the best recipe on the whole dataset (~2 h CPU; early-stopped at epoch 24)
python3 scripts/train_best_full_data.py \
    --cache-dir data/processed/mimo64_full \
    --epochs 40 --patience 5 --batch-size 64 --threads 8 \
    --jev-alpha 0.1 --outdir search_results/mimo64_full_data
```

### Inference with the saved weights

```python
import numpy as np, torch
from rhs_beamforming.nas.setups import build_setup_by_name
from rhs_beamforming.data.mimo64 import FeatureScaler
from rhs_beamforming.nas.real_trainer import E

sc = FeatureScaler.from_dict(json.load(open("data/processed/mimo64_full/scaler.json")))
meta = json.load(open("data/processed/mimo64_full/meta.json"))  # rate_mean/rate_std

model = build_setup_by_name("hybrid_jev")
model.load_state_dict(torch.load(
    "search_results/mimo64_full_data/checkpoints/hybrid_jev.pt", map_location="cpu"))
model.eval()

x = sc.transform(config_features[None]).astype("float32")     # (1, 10)
out = model(torch.from_numpy(x))[0].numpy()
M_re, M_im = out[:E], out[E:2*E]                              # 12800 each -> (20,64,10)
M = (M_re + 1j * M_im).reshape(20, 64, 10)                    # beamforming matrix
rate = out[2*E] * meta["rate_std"] + meta["rate_mean"]        # achievable rate
```

## Engineering notes (why the full run works on 3.8 GB)

1. The on-disk HDF5 stores `M` **gzip-compressed with chunk `(128,3,8,3)`**, so
   random per-sample reads are ~0.1 s each. The extraction landed on a **v9fs
   (WSL 9p) mount**, where HDF5 random I/O effectively hangs.
2. Solution: `scripts/build_full_cache.py` makes **one sequential pass** and
   writes uncompressed `X.npy` + `Y.npy` (1.54 GB) + `split_*.npy`; training
   memory-maps `Y.npy` and indexes only the current batch.
3. `scripts/train_best_full_data.py` reproduces the repo's `real_trainer.evaluate`
   metrics exactly (verified identical: M nMSE / cos / R² / rate R²) while never
   holding all Y in RAM.

## Next steps (real-time product)

- Export `hybrid_jev` to ONNX for C++/ONNX-Runtime inference and benchmark.
- Coordinate → channel → beamforming end-to-end tool on top of this checkpoint.
- Gradient-Projection-Algorithm C++ implementation to compare against the DL head.
