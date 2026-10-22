# RHS HaloBeam-GNN NAS — 5 Setups, Architecture & Results

Dataset: real **64×64 MIMO** HDF5 (`M (20,64,10)`, `w`, `E`, `H_testD`, `W_MMSE`).
Task: 10 config features → `[Re(M), Im(M), rate]` = **25601** outputs.
Hardware: CPU (8 threads), 5000-sample cache (3994/503/503 train/val/test).

---

## 1. Is the current pipeline using the HaloBeam-GNN NAS setup?

**No.** Audit result:

| Component | Location | Real-data? | Status |
|---|---|---|---|
| Tabular NAS search (what we ran) | `src/rhs_beamforming/nas/real_*.py` | ✅ | plain MLP over 10 features |
| HaloBeam-GNN NAS supernet | `src/rhs_beamform_nas/nas/` | ❌ synthetic only | rich cells, **stubs** for DARTS/ENAS/rate-evaluator |
| HaloBeam-GNN skeleton | `src/holobeam/` | ❌ | broken (`NrPlaceholder` NameError, no `__init__`) |
| Functional HaloBeam-GNN | `src/rhs_beamforming/models/holobeam_gnn.py` | ❌ synthetic `H` | shallow MLP-over-`H`, no GNN/attention |
| EVD / JEV baseline | `src/rhs_beamforming/physics/evd.py` | ✅ | classical eigen reference |

Problems in the existing NAS supernet: `RHSAdaptiveCell` mixes `(B,D)` and
`(B,2D)` candidates (runtime error for complex ops), no Transformer, no real
data path, and the three-path `RHSCell` is orphaned.

**This work adds a working HaloBeam-GNN–style setup** (GNN + attachable
Transformer + differentiable JEV decision head) that actually consumes the real
64×64 cache.

---

## 2. Architecture

### 2.1 System (proposed proper setup)

```
config x (10 features)
        │
        ▼
  ┌───────────────┐   learned tokens  ┌──────────────────────────────┐
  │  TokenStem    │ ────────────────► │  Backbone (variant-dependent) │
  │  10→D, T=16   │   [B, T, D]        │  GNN  / Transformer / both    │
  └───────────────┘                    └───────────────┬──────────────┘
                                                       │ [B,T,D]
                              ┌────────────────────────┴───────────────────┐
                              ▼                                             ▼
                     mean-pool → Linear(D,25601)           JEV decision head (hybrid_jev)
                     (mlp / gnn / transformer / hybrid)     Gram → eigh → spectral mix
                              │                                             │
                              └──────────────────┬──────────────────────────┘
                                                 ▼
                          [Re(M) (12800) | Im(M) (12800) | rate (1)] = 25601
```

### 2.2 The five setups

```
 SETUP 1  mlp              x → MLP[256,256,128,64] → head(25601)
 SETUP 2  gnn              x → stem → [GNN ×2] → mean → head(25601)
 SETUP 3  transformer      x → stem → [Transformer ×2] → mean → head(25601)
 SETUP 4  hybrid           x → stem → [GNN, Transformer, GNN, Transformer] → mean → head
 SETUP 5  hybrid_jev       x → stem → [GNN, Transformer]×2 → JEV head → 25601
```

### 2.3 JEV decision head (differentiable eigen-decomposition)

```
 h [B,T,D]
   │ normalize
   ▼
 Gram G = h hᵀ / √D                       [B,T,T]  (PSD)
   │ torch.linalg.eigh (differentiable)
   ▼
 evals ↑ , evecs V                        take top-k (num_eig, default 4)
   │ coeff = Vᵀ h  [B,k,D]
   │ weights = softmax(eig_mix(softmax(evals_top)))
   ▼
 gated = Σ_k weights_k · coeff_k          [B,D]     ← eigen-based decision
 base  = mean_t h                          [B,D]
   ▼
 Linear(2D→D) → GELU → Linear(D→25601)
```

This is the learnable analogue of the classical EVD beamformer: the classical
reference picks a **fixed** dominant right-singular direction; the head **learns
how to weight** the eigen-components. `num_eig` and the token count `T` are the
latency/accuracy knobs.

### 2.4 How this maps onto HaloBeam-GNN NAS

| HaloBeam-GNN NAS idea | This setup |
|---|---|
| RHS primitive search space | setup variants (MLP / GNN / Transformer / hybrid / +JEV) |
| `SpatialGNNBlock` | `models/gnn_block.py::GraphMessagePassingBlock` (learned adjacency) |
| attention (`configs/nas/search_space.yaml`) | `models/transformer.py::TransformerEncoderBlock` |
| config conditioning (FiLM/CrossAttn) | `TokenStem` + config→token broadcast |
| EVD/eigen reference | `models/jev_head.py::JEVDecisionHead` (differentiable) |
| multi-objective (rate↑, latency↓, params↓) | `nas/real_objective.py` score |

---

## 3. Setup table (parameter counts)

| setup | family | hidden | tokens | blocks | eig | params |
|---|---|---|---|---|---|---|
| mlp | tabular | — | — | — | — | 997,825 |
| gnn | GNN | 128 | 16 | 2 | — | 3,637,761 |
| transformer | Transformer | 128 | 16 | 2 | — | 3,703,297 |
| hybrid | GNN+Transformer | 128 | 16 | 4 | — | 4,034,305 |
| hybrid_jev | GNN+Transformer+JEV | 128 | 16 | 4 | 4 | 4,083,733 |

---

## 4. Smoke-train results (3 epochs, val = 503, CPU)

Command:
```bash
python3 scripts/smoke_train_setups.py --cache data/processed/mimo64_5000.npz \
    --epochs 3 --batch-size 64 --threads 8 \
    --output search_results/mimo64_setups/results.json
```

| setup | params (M) | latency (ms) | rate R² | rate MAE | M cos | M nMSE | M R² |
|---|---|---|---|---|---|---|---|
| mlp | 0.998 | 0.28 | 0.7787 | 4.203 | 0.6486 | 0.6211 | 0.3789 |
| gnn | 3.638 | 2.88 | 0.7669 | 4.499 | 0.7303 | 0.4710 | 0.5289 |
| transformer | 3.703 | 4.99 | 0.7708 | 4.484 | 0.7514 | 0.4386 | 0.5614 |
| hybrid | 4.034 | 8.25 | 0.7719 | 4.396 | 0.7178 | 0.4892 | 0.5107 |
| hybrid_jev | 4.084 | 5.50 | 0.7703 | 4.516 | 0.7167 | 0.4867 | 0.5132 |

Reference (fully trained): the **mlp** setup reaches **rate R² 0.800, M cos
0.751, M nMSE 0.445** after 40 epochs (`configs/model/nas_best_real.yaml`).

### Reading the table
- At only **3 epochs** the token models already beat the 3-epoch MLP on M
  fidelity (M cos 0.73–0.75 vs 0.65; nMSE 0.44–0.49 vs 0.62). They learn the
  complex `M` structure much faster.
- The **transformer** is the best per-epoch M reconstructor (M cos 0.751,
  nMSE 0.439) but pays **4.99 ms** vs the MLP's **0.28 ms** (~18×).
- `hybrid_jev` is not yet ahead at 3 epochs — its eigen head adds latency
  (5.5 ms) before the spectral prior pays off (see predictions).

---

## 5. Which training will give good results (prediction)

Based on the 3-epoch trajectory + the 40-epoch MLP reference:

| rank | setup | predicted @ 30–40 epochs | rationale |
|---|---|---|---|
| 1 | **transformer** | rate R² ~0.81–0.83, M cos ~0.79–0.82 | best M fidelity per epoch; attention fits the 20×64×10 structure |
| 2 | **hybrid_jev** | rate R² ~0.80–0.82, M cos ~0.80–0.83 | eigen prior should overtake with longer training |
| 3 | hybrid | rate R² ~0.80–0.81, M cos ~0.77–0.80 | between GNN and Transformer |
| 4 | gnn | rate R² ~0.79–0.81, M cos ~0.76–0.79 | graph prior helps M, but rate similar |
| 5 | mlp | rate R² ~0.80 (measured), M cos 0.75 | **best latency by far (0.28 ms)** |

### Recommended recipes
- **Accuracy-first:** `transformer`, ~35 epochs, batch 64, Adam lr 1e-3, cosine
  schedule, grad-clip 1.0, early stop on `M_mse + rate_mse` (this is what
  `scripts/train_best_real.py` already does). Expect M cos ≥ 0.80.
- **Latency-first:** `mlp` (0.28 ms). If you must keep attention, cut
  `num_tokens` 16→8 and `num_blocks` 2→1 → ~4× fewer FLOPs.
- **Physics-first / JEV:** `hybrid_jev`, ~40 epochs, and add a **spectral
  consistency regularizer** (penalize deviation of the JEV head's spectral
  weights from the classical EVD eigen-spectrum computed offline) — predicted
  to give the best M-similarity, at ~5–6 ms.
- **Do NOT** expect the transformer/hybrid to beat the MLP on **latency/params**
  at equal accuracy; the win is M fidelity per epoch and headroom.

### Confidence
Medium. Predictions are extrapolated from 3 epochs and a 40-epoch MLP baseline;
they need a confirmation run (see next steps). The ordering transformer > GNN >
MLP for M-similarity is strongly supported by 3-epoch numbers.

---

## 6. How to reproduce

```bash
# 1. prepare cache (already present)
python3 scripts/prepare_mimo64_dataset.py \
    --h5 /path/to/64x64_MIMO_beamforming_dataset.h5 \
    --samples 5000 --out data/processed/mimo64_5000.npz

# 2. smoke-train all five setups
python3 scripts/smoke_train_setups.py --cache data/processed/mimo64_5000.npz \
    --epochs 3 --batch-size 64 --threads 8 \
    --output search_results/mimo64_setups/results.json

# 3. full train the winner (e.g. transformer) via the existing trainer
python3 scripts/search_nas_real.py --config configs/nas/real_multisearch.yaml
```

Or in Python:
```python
from rhs_beamforming.nas.setups import SETUP_VARIANTS, build_setup_by_name
model = build_setup_by_name("transformer")   # forward(x) -> [B, 25601]
```

---

## 7. Files added

| File | Purpose |
|---|---|
| `src/rhs_beamforming/models/transformer.py` | `TokenStem`, `MultiHeadSelfAttention`, `TransformerEncoderBlock` |
| `src/rhs_beamforming/models/gnn_block.py` | `GraphMessagePassingBlock` |
| `src/rhs_beamforming/models/jev_head.py` | `JEVDecisionHead` (differentiable eigendecomposition) |
| `src/rhs_beamforming/nas/setups.py` | `SetupConfig`, `SETUP_VARIANTS`, `build_setup`, `setup_table` |
| `scripts/smoke_train_setups.py` | smoke-train all setups → JSON + Markdown |
| `search_results/mimo64_setups/results.{json,md}` | smoke-train results |
| `tests/...` | unit tests for the new modules |

## 8. Next steps
1. Confirm the predictions: full 35–40 epoch runs of `transformer`, `hybrid_jev`, `gnn`.
2. Add the JEV spectral-consistency regularizer and re-run `hybrid_jev`.
3. Sweep `num_tokens ∈ {8,16,32}` × `num_eig ∈ {2,4,8}` for the latency/accuracy frontier.
4. Retrain the winner on the full 15K split and benchmark vs GP + EVD.
