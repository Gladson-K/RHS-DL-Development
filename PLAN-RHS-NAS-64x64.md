# RHS NAS on the real 64x64 MIMO dataset

This document captures the work that wires the **real** research dataset
(`64x64_MIMO_beamforming_dataset.zip`) into the NAS recipe track, runs a
multi-model architecture search on CPU, and adds an eigen-decomposition
(``JEV``) beamforming reference.

Branch: `DL-RHS-HoloBeamGNNNet-dev`.

## 1. Task contract (matches `dev/ver1.0-CNN`)

| item | value |
| --- | --- |
| input `X` | `metadata[:, [2..11]]` → 10 config features (L, λ, scale, SNR, Pt, dH, test_size, RFno, stepSize1, noise_var) |
| target `Y` | `[Re(M).ravel(), Im(M).ravel(), rate]` |
| `M` | complex `(20, 64, 10)` (positions × RHS elements × RF chains) |
| output dim | `2*20*64*10 + 1 = 25601` |

Metadata columns (15): `Nt, Nr, L, lamd, scale, SNR, Pt, deH, test_size,
RFno, stepSize1, noise_var, rate, time, aux`.

## 2. Dataset pipeline

```
scripts/extract_mimo64_dataset.py   zip -> data/raw/*.h5 (streamed, ~18 GB)
scripts/validate_mimo64_dataset.py  schema + physics sanity report
scripts/prepare_mimo64_dataset.py   sample N, build X/Y, stratified split, cache -> data/processed/*.npz
```

Loader: `src/rhs_beamforming/data/mimo64.py`
- `build_xy`, `M_to_realimag`, `make_splits` (stratified on SNR×freq×RFno),
  `FeatureScaler`, `Mimo64TorchDataset`, `validation_report`.
- Cache prep: `src/rhs_beamforming/nas/real_data.py` (`prepare_cache`,
  `load_cache`, `build_loaders`). `M` is dropped from RAM by default.

Observed dataset ranges: freq 6.99–15.0 GHz (FP3), SNR 10–30 dB, L 2–7,
RFno 1–9, test_size 10–49, rate 21.5–75.0.

## 3. Multi-model NAS search

`src/rhs_beamforming/nas/`
- `real_space.py` — `RealSearchSpace` (depth/width, activation, dropout,
  batchnorm, residual, head) + `RHSTabularNet` / `build_model`.
- `real_trainer.py` — CPU trainer/eval (M real/imag, rate; MSE/MAE/R², nMSE, cosine).
- `real_objective.py` — `ScoreWeights` multi-objective score.
- `real_search.py` — `MultiModelSearch` (random | evolution, successive halving,
  CSV leaderboard, Pareto frontier, plots, `best.json`/`best.yaml`).

Score: `α·rate_r2 + β·M_similarity − γ·latency − δ·params − ε·violation`.

## 4. EVD / eigenvector ("JEV") baseline

`src/rhs_beamforming/physics/evd.py` — `EVDBeamformer`.
For each sample: average the effective channel over paths, take the dominant
right-singular vectors (SVD) as ideal transmit directions, project onto the
unit-modulus excitation `E` (`a = clip(Re(conj(E)·f), 0, 1)`) and compute the
GP-compatible achievable rate. Run with `scripts/evd_baseline.py`.

Reference on 1000 held-out samples: GP rate mean **51.96**, EVD rate mean
**45.95**, ratio **0.868**, Pearson **0.663**.

## 5. Results (5000 samples, CPU, held-out test = 503)

Config-selected best (score 1.056, `search_results/mimo64_nas/`, driven by
`configs/nas/real_multisearch.yaml`):

```yaml
architecture: {hidden_layers: [256,256,128,64], activation: leaky_relu,
               dropout: 0.1, batchnorm: false, residual: true, head: deep}
parameters: 997,825
latency_ms: 0.52
```

| metric | best | runner-up (512x512x256x128, silu) |
| --- | --- | --- |
| rate R² | **0.800** | 0.806 |
| rate MAE | 3.94 | 3.93 |
| M cosine | 0.751 | 0.769 |
| M nMSE | 0.445 | 0.413 |
| M R² | 0.555 | 0.587 |
| M real/imag MSE | 0.0371 / 0.0372 | 0.0344 / 0.0345 |
| params | **0.998 M** | 2.27 M |
| latency (ms) | 0.52 | 0.42 |

The selected model trades a small accuracy drop for ~2.3× fewer parameters
and the top multi-objective score.

## 6. How to run

```bash
make -f Makefile.nasarc real-extract
make -f Makefile.nasarc real-prepare
make -f Makefile.nasarc real-nas
make -f Makefile.nasarc real-evd
make -f Makefile.nasarc real-train-best
# or the whole thing:
make -f Makefile.nasarc real-all
```

Configs: `configs/dataset/real_64x64.yaml`,
`configs/nas/search_space_real.yaml`,
`configs/nas/real_multisearch.yaml`,
`configs/model/nas_best_real.yaml`,
`configs/model/evd_baseline.yaml`.

## 7. Tests

`tests/unit/test_evd.py`, `tests/nas/test_mimo64_real.py`,
`tests/nas/test_real_search.py`.

```bash
pytest tests/unit/test_evd.py tests/nas/test_mimo64_real.py tests/nas/test_real_search.py
```

## 8. Next steps

- Scale to the full 15K training split and re-benchmark.
- Extend the search space with residual/GNN primitives already present in
  `nas/search_space/primitives.py` (currently a tabular MLP family).
- Add constraint-violation evaluation to the score (`epsilon_constraint`).
