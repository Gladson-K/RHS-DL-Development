# NAS Hyperparameters (Canonical Record)

This document is the single source of truth for the RHS NAS study. Every value
below is loaded from YAML by `scripts/search_nas_real.py` and forwarded into
`SearchConfig` and `TrainConfig`. Do not change a run's hyperparameters without
updating this table.

Task: map a 10-dim system-config vector `x` to the flattened target
`[Re(M).ravel(), Im(M).ravel(), sum_rate]` (dim `25601`, with `E = 20*64*10 =
12800` per real/imag part).

## 1. Dataset and splits

| Item | Value |
|---|---|
| Source | `data/raw/64x64_MIMO_beamforming_dataset.h5` (15,000 samples) |
| Search cache | `data/processed/mimo64_5000.npz` (5,000 samples) |
| Search split | train / val / test = 3,994 / 503 / 503 (stratified) |
| Full retrain cache | `data/processed/mimo64_full/` |
| Full split | 11,988 / 1,506 / 1,506 |
| Input features | `metadata[:, 2:12]` → `[L, lamd, scale, SNR, Pt, deH, test_size, RFno, stepSize1, noise_var]` |
| Output | `[Re(M), Im(M), rate]`, dim 25,601 |
| Rate normalization | `(rate - 51.794) / 12.298` for the rate output only |

## 2. Fixed training hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | `optimizer: adam` |
| Learning rate | 1e-3 | cosine schedule to `lr*1e-2` |
| Weight decay | 1e-4 | |
| Batch size | 64 | |
| Gradient clipping | 1.0 | global norm |
| Loss | `1.0*0.5*(MSE_Re + MSE_Im) + 1.0*MSE_rate` | `m_weight=rate_weight=1.0` |
| Early-stop patience | 6 | search; full retrain uses 5 |
| Seed | 42 | Python/NumPy/PyTorch |
| Device | CPU | 8 torch threads |
| Successive halving | on | rung 1 = 25% data @ `budget//2` epochs, rung 2 = 50% data @ `budget` epochs |
| Final stage | on | top-`k` retrained on 100% train data |
| Final epochs | per config | B4 = 15, B8 = 25, Evolution = 20 |
| Budget epochs | per config | B4 = 4, B8 = 8, Evolution = 6 |

## 3. Search space

| Axis | Choices |
|---|---|
| Hidden layers | `[1024,512,256,128,64]`, `[512,256,128]`, `[768,384,192,96]`, `[1024,768,384,192]`, `[512,512,256,128]`, `[256,256,128,64]`, `[1024,512,256]`, `[384,192,96,48]`, `[2048,1024,512,256]`, `[640,320,160,80]` |
| Activation | relu, leaky_relu, gelu, prelu, elu, silu |
| Dropout | 0.0, 0.05, 0.1, 0.15, 0.2, 0.3 |
| Batch norm | true, false |
| Residual | true, false |
| Output head | dense, residual, deep |

## 4. Multi-objective score

`score = 1.0*rate_r2 + 0.5*M_cos - 0.05*latency_ms - 0.02*params_M - 0.0*violation`

(Pareto selection uses rate R², M cosine, parameter count and latency.)

## 5. Search configurations (three runs)

| Run | File | Strategy | Candidates / population | Budget | Final | Top-k | Output |
|---|---|---|---|---|---|---|---|
| random_b4 | `configs/nas/real_multisearch.yaml` | random | 12 | 4 | 15 | 3 | `search_results/nas_study/random_b4` |
| random_b8 | `configs/nas/real_multisearch_budget8.yaml` | random | 12 | 8 | 25 | 3 | `search_results/nas_study/random_b8` |
| evolution | `configs/nas/real_multisearch_evolution.yaml` | evolution | pop 8 / gen 3 | 6 | 20 | 3 | `search_results/nas_study/evolution` |

## 6. What was fixed

The original loader (`scripts/search_nas_real.py::_load_yaml_config`) forwarded
only `batch_size`, `lr`, `weight_decay` and `torch_threads`. The following YAML
keys were silently ignored, so editing a config changed nothing in the search:

- `training.early_stop_patience`
- `training.optimizer`, `training.scheduler`, `training.grad_clip`, `training.device`
- `loss.m_weight`, `loss.rate_weight`

All of these are now loaded and wired through `SearchConfig` into `TrainConfig`.
`SearchConfig` gained `scheduler` and `grad_clip` fields. The successive-halving
rung schedule now guarantees a strictly larger epoch budget on the second rung,
removing the degenerate case for `budget_epochs < 3`. Script defaults were
aligned with `SearchConfig` defaults (12 candidates, budget 8, final 30).

**Observed effect:** the B4 run is behaviourally identical to the historical
`search_results/mimo64_nas` run (same effective schedule), while the B8 and
evolution runs now genuinely use their configured patience/loss/optimizer values
instead of the dataclass defaults.

## 7. Reproduce

```bash
export PATH=/home/glad/.local/bin:$PATH
python3 -u scripts/search_nas_real.py --config configs/nas/real_multisearch.yaml
python3 -u scripts/search_nas_real.py --config configs/nas/real_multisearch_budget8.yaml
python3 -u scripts/search_nas_real.py --config configs/nas/real_multisearch_evolution.yaml
```
