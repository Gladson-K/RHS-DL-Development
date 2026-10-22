# RHS-DL-Development — Claude Project Context

## What this repo does

Deep-learning adaptive beamforming for **Reconfigurable Holographic Surfaces (RHS)** in wireless communications. Two parallel research tracks:

1. **HaloBeamGNNNet** — hand-designed GNN-based beamforming model
2. **NAS Recipe** — Neural Architecture Search over RHS-specific primitives

Full docs in `README-NASRecipe.md` and `README-HaloBeamGnnNet.md`.

---

## Critical invariants

The following are shared between both tracks and must **never be duplicated or diverged**:

- Physics: `src/rhs_beamforming/physics/` (geometry, channel, MMSE, rate, constraints)
- Dataset pipeline: `src/rhs_beamforming/data/`
- GP reference: `src/rhs_beamforming/gp/`
- Evaluation: `src/rhs_beamforming/evaluation/`

Architectures may change. The physics model must not.

---

## Source packages

### `src/rhs_beamforming/` — Shared core library

```
config/          Pydantic schemas + YAML loader
physics/         constants, geometry, channel, mmse, rhs, excitation, rate, constraints
channels/        Channel generation
gp/              GP reference: objective, gradient, optimizer, validation, reference
data/            sampler, generator, validator, split, hdf5, dataset
models/
  common/        complex.py, normalization.py, embeddings.py
  complex/       Complex-valued layers
  efficient/     mbconv.py, squeeze_excitation.py, efficient_block.py
  gnn/           graph.py, message_passing.py, spatial_gnn.py, attention.py
  attention/     Attention mechanisms
  heads/         rhs_head.py (outputs M and w)
  holobeam_gen.py  ← Top-level HaloBeamGNNNet model
nas/             NAS integration hooks
losses/          Rate-aware, physics-aware losses
training/        Training loops + schedulers
evaluation/      Rate evaluation, benchmarking
utils/           Shared utilities
```

### `src/rhs_beamform_nas/` — NAS-specific package

```
nas/
  runner.py              Orchestrates strategy + evaluator
  search_space/
    primitives.py        All RHS NAS primitive modules
    rhs_cell.py          Three-path cell (complex + spatial + config)
    rhs_adaptive_cell.py NNI LayerChoice cell
    beamforming_cell.py  Output head cell
    config_conditioning.py FiLM, CrossAttn, Add, Concat
    model_space.py       RHSAdaptiveBeamformingModelSpace (supernet)
  strategies/
    random_search.py     Stage 1 (20–50 archs)
    darts.py             Stage 2 differentiable search
    enas.py              Stage 2 weight-sharing
    evolution.py         Stage 3 Pareto evolutionary search
  evaluators/
    rate_evaluator.py    Achievable rate metric
    latency_evaluator.py Inference latency
    constraint_evaluator.py Beamforming constraint violation
    multi_objective.py   BeamformingEvaluator (combined)
  objectives/
    accuracy.py          Prediction accuracy
    rate.py              Rate objective
    latency.py           Latency objective
    model_size.py        Parameter count
    combined.py          RHSNASObjective (weighted)
  export/
    architecture.py      ArchitectureDescriptor
    exporter.py          ArchitectureExporter
    fixed_model.py       FixedRHSModel
```

---

## NAS search space primitives

The search space uses **RHS-beamforming-specific** operations (NOT generic image-classification ops):

| Category               | Primitives                                                      |
| ---------------------- | --------------------------------------------------------------- |
| Complex processing     | ComplexLinear, RealImagLinear, ComplexResidualBlock, ComplexNorm |
| Spatial processing     | LocalRHSBlock, SpatialGNNBlock, GraphAttentionBlock, GlobalAttentionBlock |
| Efficient processing   | MBConvBlock, RHSBottleneck, DepthwiseBlock, ResidualBlock       |
| Config conditioning    | AddConditioning, FiLMConditioning, CrossAttentionConditioning, ConcatConditioning |
| Depth                  | {2, 4, 6, 8}                                                    |
| Hidden dim             | {128, 192, 256, 384}                                            |
| Output heads           | M (amplitude/phase), w (combiner), rate                         |

Base class for all primitives: `RHSPrimitive(nn.Module)` in `primitives.py`.

---

## NAS objective (multi-objective scoring)

```python
score = (
    alpha * normalized_rate          # maximize achievable rate
    + beta * beamforming_similarity  # alignment with GP teacher
    - gamma * latency_ms             # minimize inference latency
    - delta * parameter_millions     # minimize model size
    - epsilon * constraint_violation # penalize beamforming violations
)
```

---

## Configuration

All config is YAML + Pydantic. Structure:

```
configs/
  base.yaml             Root merged first
  system/               Nr, Nx, Nt, L, RFno, frequency
  dataset/              debug_20, v1_10k, v2_generalization
  model/                baseline_mlp, baseline_gnn, holobeam_gen
  nas/                  search_space, random, darts, enas, evolution
  training/             baseline, supervised, rate_aware, physics_aware
  experiments/          exp_gp, exp_mlp, exp_gnn, exp_holobeam, exp_nas
```

**Always use configs over hardcoded values.**

---

## Dataset sizes

| Split         | Size       | Used for                    |
| ------------- | ---------- | --------------------------- |
| Debug         | 20         | Quick smoke test            |
| GP labels     | 10K        | NAS search evaluation       |
| NAS subset    | 500–1000   | Stratified architecture eval|
| Full train    | 15K        | Final retrain               |
| Held-out test | separate   | Final benchmark             |

---

## Key scripts

```bash
scripts/validate_gp.py          # Validate GP reference
scripts/generate_dataset.py     # Generate HDF5 + GP labels
scripts/train.py                # Train model
scripts/search_nas.py           # Run NAS search
scripts/export_architecture.py  # Export best arch → configs/model/nas_best.yaml
scripts/retrain_architecture.py # Full retrain from exported arch
scripts/evaluate.py             # Evaluate vs GP
scripts/benchmark.py            # Latency / param-count benchmark
```

---

## NAS workflow (3 stages)

```
Stage 1: Random Search (20–50 architectures)
  → validate search space is well-formed

Stage 2: One-Shot (DARTS or ENAS)
  → weight-sharing supernet search

Stage 3: Pareto/Evolutionary
  → multi-objective: rate, latency, parameters
  → output: Accuracy-optimal, Balanced, Latency-optimal architectures
```

Top 3–5 Pareto-optimal architectures → full 15K retrain → final benchmark vs GP.

---

## NAS result artifact format

```
search_results/YYYY-MM-DD_<name>/
  config.yaml | search_space.yaml | search.log
  architectures/   arch_0001.json … best.json
  metrics/         candidates.csv, pareto.csv, best.json
  checkpoints/     supernet.pt
  plots/           rate_vs_latency.png, pareto_frontier.png
```

`best.json` structure:
```json
{
  "architecture": {
    "hidden_dim": 256,
    "rhs_depth": 4,
    "spatial_operator": ["gnn", "gnn", "graph_attention", "complex_residual"],
    "attention_heads": 4,
    "conditioning": "film",
    "M_representation": "amplitude_phase",
    "w_representation": "real_imag"
  },
  "metrics": {
    "normalized_rate": 0.987,
    "gp_rate_ratio": 0.991,
    "latency_ms": 0.82,
    "parameters": 6842312,
    "constraint_violation": 0.0
  }
}
```

---

## Model families

| Model             | Track | Purpose                      |
| ----------------- | ----- | ----------------------------- |
| GP                | —     | Classical oracle reference    |
| HoloBeam-Base     | 1     | Hand-designed DL baseline     |
| HoloBeam-NAS      | 2     | NAS-optimised architecture    |
| HoloBeam-NAS-Lite | 2     | Low-latency deployment model  |

---

## Tests

```
tests/unit/        config, complex, geometry, channel, mmse, gp, rate
tests/model/       efficient, gnn, attention, heads, holobeam
tests/nas/         primitives, rhs_cell, model_space, search_space, evaluator, export
tests/integration/ single_sample, training_step, nas_smoke
```

```bash
pytest tests/                   # all tests
pytest tests/unit/              # unit only
pytest tests/nas/               # NAS only
```

---

## Code style

- Formatter: `black` (line length 100)
- Linter: `ruff`
- Type checker: `mypy`
- Python ≥ 3.10, use modern union syntax (`X | Y`), match statements where appropriate
- All public classes and functions must have docstrings
- Configuration values via Pydantic schemas — never hardcode system parameters

---

## Make commands

```bash
make -f Makefile.halobeamgnnNet   # HaloBeamGNNNet workflow
make -f Makefile.nasarc            # NAS workflow
```

---

*Last updated: 2026-08-31. Keep in sync with `gemini.md`.*
