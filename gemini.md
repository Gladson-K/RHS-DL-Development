# RHS-DL-Development — Gemini / Antigravity IDE Context

> **Purpose**: This file gives Gemini and the Antigravity IDE a complete, authoritative map of this repository so that AI assistance is accurate, consistent, and aligned with the research objectives.

---

## Repository Identity

| Field        | Value                                                          |
| ------------ | -------------------------------------------------------------- |
| Name         | `RHS-DL-Development`                                          |
| Domain       | Wireless communications / reconfigurable holographic surfaces  |
| Language     | Python ≥ 3.10                                                  |
| Framework    | PyTorch ≥ 2.3                                                  |
| NAS Library  | NNI (Microsoft Neural Network Intelligence)                    |
| Config       | YAML + Pydantic schemas                                        |
| Testing      | pytest                                                         |
| Linting      | ruff + black + mypy                                            |
| Entry points | `Makefile.halobeamgnnNet` and `Makefile.nasarc`               |

---

## Research Context

This repository implements **deep-learning-based adaptive beamforming for Reconfigurable Holographic Surfaces (RHS)** in wireless communication systems.

### Key terms

| Term     | Meaning                                                                 |
| -------- | ----------------------------------------------------------------------- |
| RHS      | Reconfigurable Holographic Surface — a large antenna array              |
| GP       | Gradient-based optimisation — the classical reference/oracle solution    |
| M        | RHS amplitude/phase excitation vector (beamforming weight)              |
| w        | Combiner weight vector                                                   |
| Rate     | Achievable spectral efficiency (bits/s/Hz) — primary performance metric |
| NAS      | Neural Architecture Search                                              |
| DARTS    | Differentiable Architecture Search                                      |
| ENAS     | Efficient Neural Architecture Search (weight sharing)                   |
| FiLM     | Feature-wise Linear Modulation (conditioning mechanism)                 |

---

## Two-Track Development Model

```
RHS-DL-Development
        │
        ├── Track 1: HaloBeamGNNNet
        │     Fixed, manually designed GNN architecture
        │     Reference: README-HaloBeamGnnNet.md
        │
        └── Track 2: NAS Recipe
              Automated architecture search over RHS-specific primitives
              Reference: README-NASRecipe.md
```

Both tracks share:
- Common physics (`src/rhs_beamforming/physics/`)
- Common dataset pipeline (`src/rhs_beamforming/data/`)
- Common GP reference (`src/rhs_beamforming/gp/`)
- Common evaluation methodology
- Common configuration system

**Do not duplicate physics, channel, or evaluation logic between tracks.**

---

## Authoritative Documentation

| File                     | Authoritative for                        |
| ------------------------ | ---------------------------------------- |
| `README.md`              | Repository navigation and principles     |
| `README-HaloBeamGnnNet.md` | HaloBeamGNNNet implementation          |
| `README-NASRecipe.md`    | NAS search space, strategies, evaluation |
| `gemini.md` (this file)  | Gemini / Antigravity IDE project context |
| `claude.md`              | Claude project context                   |

---

## Complete Source Layout

```
src/
├── rhs_beamforming/                  # Core shared library
│   ├── __init__.py
│   ├── config/                       # Pydantic-based config schemas + loader
│   ├── physics/                      # Constants, geometry, channel, MMSE, RHS, rate, constraints
│   ├── channels/                     # Channel generation / simulation
│   ├── gp/                           # GP reference: objective, gradient, optimizer, validation
│   ├── data/                         # Sampler, generator, validator, splits, HDF5, dataset
│   ├── models/                       # HaloBeamGNNNet components
│   │   ├── common/                   # complex.py, normalization.py, embeddings.py
│   │   ├── complex/                  # Complex-valued layer implementations
│   │   ├── efficient/                # mbconv.py, squeeze_excitation.py, efficient_block.py
│   │   ├── gnn/                      # graph.py, message_passing.py, spatial_gnn.py, attention.py
│   │   ├── attention/                # Attention mechanisms
│   │   ├── heads/                    # rhs_head.py (M, w outputs)
│   │   └── holobeam_gen.py           # Top-level HaloBeamGNNNet model
│   ├── nas/                          # NAS integration inside shared package
│   ├── losses/                       # Rate-aware and physics-aware losses
│   ├── training/                     # Training loops and schedulers
│   ├── evaluation/                   # Rate evaluation, benchmarking
│   └── utils/                        # Shared utilities
│
└── rhs_beamform_nas/                 # NAS-specific package
    ├── __init__.py
    └── nas/
        ├── __init__.py
        ├── runner.py                 # Orchestrates search strategy + evaluator
        ├── search_space/
        │   ├── __init__.py
        │   ├── primitives.py         # All RHS NAS primitive operations
        │   ├── rhs_cell.py           # Three-path RHS cell (complex + spatial + config)
        │   ├── rhs_adaptive_cell.py  # NNI ModelSpace cell with LayerChoice
        │   ├── beamforming_cell.py   # Output head cell
        │   ├── config_conditioning.py # FiLM, CrossAttention, Add, Concat
        │   └── model_space.py        # RHSAdaptiveBeamformingModelSpace (supernet)
        ├── strategies/
        │   ├── __init__.py
        │   ├── random_search.py      # Stage 1: random search (20–50 architectures)
        │   ├── darts.py              # Stage 2: differentiable search
        │   ├── enas.py               # Stage 2: weight-sharing one-shot
        │   └── evolution.py          # Stage 3: evolutionary / Pareto search
        ├── evaluators/
        │   ├── __init__.py
        │   ├── rate_evaluator.py     # Achievable-rate metric
        │   ├── latency_evaluator.py  # Inference latency benchmark
        │   ├── constraint_evaluator.py # Beamforming constraint violation
        │   └── multi_objective.py    # BeamformingEvaluator (combined)
        ├── objectives/
        │   ├── __init__.py
        │   ├── accuracy.py           # Prediction accuracy objective
        │   ├── rate.py               # Achievable rate objective
        │   ├── latency.py            # Latency objective
        │   ├── model_size.py         # Parameter count / model size objective
        │   └── combined.py           # RHSNASObjective (weighted combination)
        └── export/
            ├── __init__.py
            ├── architecture.py       # ArchitectureDescriptor dataclass
            ├── exporter.py           # ArchitectureExporter
            └── fixed_model.py        # FixedRHSModel (deployed architecture)
```

---

## NAS Search Space

The NAS search space contains **RHS-beamforming-specific primitives** — not generic image-classification operations.

### Primitive categories

```
RHSAdaptiveBeamformingSearchSpace
├── Complex processing
│   ├── ComplexLinear
│   ├── RealImagLinear
│   ├── ComplexResidualBlock
│   └── ComplexNorm
├── Spatial processing
│   ├── LocalRHSBlock
│   ├── SpatialGNNBlock
│   ├── GraphAttentionBlock
│   └── GlobalAttentionBlock
├── Efficient processing
│   ├── MBConvBlock
│   ├── RHSBottleneck
│   ├── DepthwiseBlock
│   └── ResidualBlock
├── Configuration conditioning
│   ├── AddConditioning
│   ├── FiLMConditioning
│   ├── CrossAttentionConditioning
│   └── ConcatConditioning
├── Depth: {2, 4, 6, 8}
├── Hidden dim: {128, 192, 256, 384}
└── Output heads: M (amplitude/phase), w (combiner), rate
```

### NAS Objective

```python
score = (
    alpha * normalized_rate          # maximize rate
    + beta * beamforming_similarity  # GP teacher alignment
    - gamma * latency_ms             # minimize latency
    - delta * parameter_millions     # minimize parameters
    - epsilon * constraint_violation # penalize constraint violations
)
```

### Three-stage search

| Stage | Strategy        | Purpose                               |
| ----- | --------------- | ------------------------------------- |
| 1     | Random Search   | Validate search space (20–50 archs)   |
| 2     | DARTS / ENAS    | Weight-sharing one-shot search        |
| 3     | Evolution / Pareto | Multi-objective Pareto search      |

---

## Configuration System

All configuration is YAML-driven with Pydantic validation.

```
configs/
├── base.yaml                    # Root config (merged first)
├── system/
│   ├── rhs_v1_fixed.yaml        # Fixed Nr, Nx, Nt, L, RFno, frequency
│   └── rhs_v2_variable.yaml     # Variable system parameters
├── dataset/
│   ├── debug_20.yaml            # 20-sample debug set
│   ├── v1_10k.yaml              # 10K GP-labelled dataset
│   └── v2_generalization.yaml   # Generalization dataset
├── model/
│   ├── baseline_mlp.yaml
│   ├── baseline_gnn.yaml
│   └── holobeam_gen.yaml
├── nas/
│   ├── search_space.yaml        # Search space hyperparameters
│   ├── random.yaml
│   ├── darts.yaml
│   ├── enas.yaml
│   └── evolution.yaml
├── training/
│   ├── baseline.yaml
│   ├── supervised.yaml
│   ├── rate_aware.yaml
│   └── physics_aware.yaml
└── experiments/
    ├── exp_gp.yaml
    ├── exp_mlp.yaml
    ├── exp_gnn.yaml
    ├── exp_holobeam.yaml
    └── exp_nas.yaml
```

---

## Key Scripts

| Script                     | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| `scripts/validate_gp.py`   | Validate GP reference solution             |
| `scripts/generate_dataset.py` | Generate HDF5 dataset with GP labels   |
| `scripts/validate_dataset.py` | Validate generated dataset             |
| `scripts/train.py`         | Train HaloBeamGNNNet or fixed architecture |
| `scripts/search_nas.py`    | Run NAS search                             |
| `scripts/export_architecture.py` | Export best arch to YAML config     |
| `scripts/retrain_architecture.py` | Full retrain of exported arch      |
| `scripts/evaluate.py`      | Evaluate model vs GP reference             |
| `scripts/benchmark.py`     | Inference latency and model-size benchmark |

---

## Dataset Sizes

| Split              | Size             | Purpose                    |
| ------------------ | ---------------- | -------------------------- |
| Debug              | 20 samples       | Quick validation           |
| GP labels          | 10K samples      | NAS search set             |
| NAS search subset  | 500–1000 samples | Stratified NAS evaluation  |
| Full training      | 15K samples      | Final architecture retrain |
| Held-out test      | separate set     | Final benchmark            |

NAS search subset stratified by: frequency, SNR, channel seed, user configuration, geometry.

---

## Research Lifecycle (recommended order)

```
1. Validate config
2. Validate physics
3. Validate GP/reference implementation
4. Generate 20-sample debug dataset
5. Run unit + integration tests
6. Validate HaloBeamGNNNet
7. Generate 10K production dataset
8. Train HaloBeamGNNNet baseline
9. Validate NAS search space
10. Run NAS search (random → DARTS/ENAS → evolution)
11. Select Pareto-optimal architectures (top 3–5)
12. Retrain selected architectures (15K)
13. Evaluate against GP reference
14. Compare HaloBeamGNNNet vs NAS
15. Benchmark latency / model size
16. Select final architecture
```

---

## NAS Result Format

Every NAS run produces:

```
search_results/
└── YYYY-MM-DD_<experiment>/
    ├── config.yaml
    ├── search_space.yaml
    ├── search.log
    ├── architectures/
    │   ├── arch_0001.json
    │   └── best.json
    ├── metrics/
    │   ├── candidates.csv
    │   ├── pareto.csv
    │   └── best.json
    ├── checkpoints/
    │   └── supernet.pt
    └── plots/
        ├── rate_vs_latency.png
        ├── params_vs_rate.png
        └── pareto_frontier.png
```

---

## Testing Structure

```
tests/
├── unit/        — Config, physics, channel, GP, rate fundamentals
├── model/       — HaloBeamGNNNet component tests
├── nas/         — Search space, evaluator, export tests
└── integration/ — End-to-end smoke tests (single sample, training step, NAS smoke)
```

Run: `pytest tests/`

---

## Model Families

| Model             | Purpose                             |
| ----------------- | ----------------------------------- |
| GP                | Classical reference / oracle        |
| HoloBeam-Base     | Hand-designed GNN baseline (Track 1)|
| HoloBeam-NAS      | NAS-optimised model (Track 2)       |
| HoloBeam-NAS-Lite | Latency/edge-optimised variant      |

---

## Important Invariants

These must **never** change between architectures:

- System model (Nr, Nx, Nt, L, RFno, frequency)
- Channel model and RHS geometry
- GP reference formulation
- Beamforming constraints (unit-modulus, power)
- Rate calculation formula
- Evaluation methodology

This ensures scientifically meaningful architecture comparisons.

---

## Development Commands

```bash
# HaloBeamGNNNet workflow
make -f Makefile.halobeamgnnNet

# NAS workflow
make -f Makefile.nasarc

# Run tests
pytest tests/

# Format / lint
ruff check src/ tests/
black src/ tests/
mypy src/
```

---

## Physics Core (never modify without research justification)

- `src/rhs_beamforming/physics/constants.py` — Physical constants
- `src/rhs_beamforming/physics/geometry.py` — RHS geometry and array response
- `src/rhs_beamforming/physics/channel.py` — Channel model
- `src/rhs_beamforming/physics/mmse.py` — MMSE receiver
- `src/rhs_beamforming/physics/rhs.py` — RHS excitation model
- `src/rhs_beamforming/physics/rate.py` — Achievable rate calculation
- `src/rhs_beamforming/physics/constraints.py` — Beamforming constraint enforcement

---

*Last updated: 2026-08-31. Maintained alongside `claude.md`. Both files should be kept in sync when the repository structure changes.*
