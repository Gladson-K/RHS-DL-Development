# RHS Setup for DL Inference in Adaptive Beamforming

## Repo Structure

```
holobeam-gen/
│
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .pre-commit-config.yaml
│
├── configs/
│   ├── base.yaml
│   │
│   ├── dataset/
│   │   ├── v1_fixed.yaml
│   │   └── v2_variable.yaml
│   │
│   ├── model/
│   │   ├── holobeam_gen_v1.yaml
│   │   ├── holobeam_gen_small.yaml
│   │   └── holobeam_gen_large.yaml
│   │
│   ├── training/
│   │   ├── baseline.yaml
│   │   ├── rate_aware.yaml
│   │   └── physics_aware.yaml
│   │
│   └── experiments/
│       ├── gp_baseline.yaml
│       ├── mlp_baseline.yaml
│       ├── gnn_baseline.yaml
│       └── holobeam_gen.yaml
│
├── src/
│   └── holobeam/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   └── loader.py
│       │
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── geometry.py
│       │   ├── channel.py
│       │   ├── mmse.py
│       │   ├── rhs.py
│       │   ├── excitation.py
│       │   ├── rate.py
│       │   └── constraints.py
│       │
│       ├── gp/
│       │   ├── __init__.py
│       │   ├── objective.py
│       │   ├── gradient.py
│       │   ├── optimizer.py
│       │   ├── validation.py
│       │   └── reference.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── sampler.py
│       │   ├── generator.py
│       │   ├── validator.py
│       │   ├── split.py
│       │   ├── hdf5.py
│       │   └── dataset.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   │
│       │   ├── common/
│       │   │   ├── complex.py
│       │   │   ├── normalization.py
│       │   │   └── embeddings.py
│       │   │
│       │   ├── efficient/
│       │   │   ├── mbconv.py
│       │   │   ├── squeeze_excitation.py
│       │   │   └── efficient_block.py
│       │   │
│       │   ├── gnn/
│       │   │   ├── graph.py
│       │   │   ├── message_passing.py
│       │   │   ├── spatial_gnn.py
│       │   │   └── attention.py
│       │   │
│       │   ├── heads/
│       │   │   ├── rhs_head.py
│       │   │   ├── beamformer_head.py
│       │   │   └── rate_head.py
│       │   │
│       │   └── holobeam_gen.py
│       │
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── beamforming.py
│       │   ├── rate.py
│       │   ├── power.py
│       │   ├── phase.py
│       │   ├── fairness.py
│       │   └── combined.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── callbacks.py
│       │   ├── metrics.py
│       │   └── checkpoint.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluate.py
│       │   ├── rate_metrics.py
│       │   ├── latency.py
│       │   ├── robustness.py
│       │   └── comparison.py
│       │
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           ├── device.py
│           └── profiling.py
│
├── scripts/
│   ├── validate_gp.py
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── export.py
│
├── tests/
│   ├── test_config.py
│   ├── test_dimensions.py
│   ├── test_channel.py
│   ├── test_mmse.py
│   ├── test_gp.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_inference.py
│
├── notebooks/
│   ├── 01_gp_validation.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_rate_comparison.ipynb
│   └── 05_ablation_study.ipynb
│
├── experiments/
│   ├── README.md
│   ├── runs/
│   └── results/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── README.md
│
├── checkpoints/
│   └── .gitkeep
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── papers/
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── gp_reference.md
    ├── training.md
    ├── evaluation.md
    └── deployment.md

```

## HoloBEAMGNNNet Architecture

**Configurable Physics-Aware Neural Beamforming Network**

```
                    USER CONFIGURATION VECTOR
       ┌──────────────────────────────────────────────┐
       │ frequency, SNR, noise, Nx, Nt, Nr, L, RFno │
       │ spacing, Pt, scale, geometry, etc.          │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
                    Configuration Encoder
                              │
                              ▼
 H_BS ─────┐           ┌───────────────┐
 H_RHS ────┼──────────►│ Complex Input │
 H_testD ──┤           │    Encoder    │
 E ────────┘           └───────┬───────┘
                               │
                               ▼
                    Efficient Spatial Blocks
                         ↓       ↓
                    GNN message passing
                         ↓
                    Spatial Attention
                         ↓
                    Cross-user attention
                         │
                ┌────────┴────────┐
                ▼                 ▼
         RHS Beamformer      User Beamformer
             Head                 Head
                │                 │
                ▼                 ▼
          M_real/M_imag       w_real/w_imag
                │                 │
                └────────┬────────┘
                         ▼
                 Physics Projection
                         │
                         ▼
              Feasible Beamforming
                 M_pred , w_pred
                         │
                         ▼
                  Rate Calculator
                         │
                         ▼
                  Sum-rate / SINR

```

# Details we get from EfficientNet Pytorch Implementation

```bash
Efficient spatial block
     ↓
antenna/RHS embeddings
     ↓
graph interaction
     ↓
attention
     ↓
beamforming vector
```

We can borrow:

- depth scaling
- width scaling
- bottleneck blocks
- skip connections
- squeeze/excitation-style feature weighting

# Core Project Architecture Intended for Development

```
                         ┌─────────────────────┐
                         │   YAML CONFIG       │
                         │ System/Channel/Model│
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
        Classical Reference                     DL Configuration
                 │                                     │
                 ▼                                     │
        Channel Generation                             │
                 │                                     │
                 ▼                                     │
          MMSE Beamforming                              │
                 │                                     │
                 ▼                                     │
          RHS Excitation E                              │
                 │                                     │
                 ▼                                     │
          Corrected GP                                  │
                 │                                     │
           ┌─────┴─────┐                               │
           ▼           ▼                               │
          M_GP        w_GP                             │
           │           │                               │
           └─────┬─────┘                               │
                 │                                     │
                 ▼                                     ▼
              Dataset ─────────────────────────► HoloBeam-GEN
                                                     │
                                  ┌──────────────────┼──────────────┐
                                  │                  │              │
                                  ▼                  ▼              ▼
                              Efficient           GNN          Attention
                               blocks           spatial         global
                                  │                  │              │
                                  └──────────────────┼──────────────┘
                                                     ▼
                                               Output heads
                                                     │
                                            ┌────────┴────────┐
                                            ▼                 ▼
                                           M_DL              w_DL
                                            │                 │
                                            └────────┬────────┘
                                                     ▼
                                             Physics projection
                                                     │
                                                     ▼
                                            Rate/SINR calculation
                                                     │
                                                     ▼
                                            ┌────────────────┐
                                            │ GP vs DL       │
                                            │ Rate/Latency   │
                                            │ Constraints    │
                                            └────────────────┘
```

# Neural Architecture Search Specification Adapted with existing structure

```
RHS-Adaptive-Beamforming-NAS/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .gitignore
├── .gitattributes
├── .pre-commit-config.yaml
│
├── .devcontainer/
│   ├── devcontainer.json
│   ├── Dockerfile
│   └── post-create.sh
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── nas-smoke.yml
│       └── release.yml
│
├── configs/
│   │
│   ├── base.yaml
│   │
│   ├── system/
│   │   ├── rhs_v1_fixed.yaml
│   │   └── rhs_v2_variable.yaml
│   │
│   ├── dataset/
│   │   ├── debug_20.yaml
│   │   ├── v1_10k.yaml
│   │   └── v2_generalization.yaml
│   │
│   ├── model/
│   │   ├── baseline_mlp.yaml
│   │   ├── baseline_gnn.yaml
│   │   └── holobeam_gen.yaml
│   │
│   ├── nas/
│   │   ├── search_space.yaml
│   │   ├── random.yaml
│   │   ├── darts.yaml
│   │   ├── enas.yaml
│   │   └── evolution.yaml
│   │
│   ├── training/
│   │   ├── baseline.yaml
│   │   ├── supervised.yaml
│   │   ├── rate_aware.yaml
│   │   └── physics_aware.yaml
│   │
│   └── experiments/
│       ├── exp_gp.yaml
│       ├── exp_mlp.yaml
│       ├── exp_gnn.yaml
│       ├── exp_holobeam.yaml
│       └── exp_nas.yaml
│
├── src/
│   └── rhs_beamforming/
│       │
│       ├── config/
│       ├── physics/
│       ├── channels/
│       ├── gp/
│       ├── data/
│       │
│       ├── models/
│       │   ├── common/
│       │   ├── complex/
│       │   ├── efficient/
│       │   ├── gnn/
│       │   ├── attention/
│       │   ├── heads/
│       │   └── holobeam_gen.py
│       │
│       ├── nas/
│       │   │
│       │   ├── search_space/
│       │   │   ├── __init__.py
│       │   │   ├── primitives.py
│       │   │   ├── rhs_cell.py
│       │   │   ├── beamforming_cell.py
│       │   │   ├── config_conditioning.py
│       │   │   └── model_space.py
│       │   │
│       │   ├── strategies/
│       │   │   ├── random_search.py
│       │   │   ├── darts.py
│       │   │   ├── enas.py
│       │   │   └── evolution.py
│       │   │
│       │   ├── evaluators/
│       │   │   ├── rate_evaluator.py
│       │   │   ├── latency_evaluator.py
│       │   │   ├── constraint_evaluator.py
│       │   │   └── multi_objective.py
│       │   │
│       │   ├── objectives/
│       │   │   ├── accuracy.py
│       │   │   ├── rate.py
│       │   │   ├── latency.py
│       │   │   ├── model_size.py
│       │   │   └── combined.py
│       │   │
│       │   ├── export/
│       │   │   ├── exporter.py
│       │   │   ├── architecture.py
│       │   │   └── fixed_model.py
│       │   │
│       │   └── runner.py
│       │
│       ├── losses/
│       ├── training/
│       ├── evaluation/
│       └── utils/
│
├── scripts/
│   ├── validate_gp.py
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── train.py
│   ├── search_nas.py
│   ├── export_architecture.py
│   ├── retrain_architecture.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_complex.py
│   │   ├── test_geometry.py
│   │   ├── test_channel.py
│   │   ├── test_mmse.py
│   │   ├── test_gp.py
│   │   └── test_rate.py
│   │
│   ├── model/
│   │   ├── test_efficient.py
│   │   ├── test_gnn.py
│   │   ├── test_attention.py
│   │   ├── test_heads.py
│   │   └── test_holobeam.py
│   │
│   ├── nas/
│   │   ├── test_primitives.py
│   │   ├── test_rhs_cell.py
│   │   ├── test_model_space.py
│   │   ├── test_search_space.py
│   │   ├── test_evaluator.py
│   │   └── test_export.py
│   │
│   └── integration/
│       ├── test_single_sample.py
│       ├── test_training_step.py
│       └── test_nas_smoke.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── README.md
│
├── search_results/
│   ├── architectures/
│   ├── checkpoints/
│   ├── metrics/
│   └── reports/
│
├── checkpoints/
├── experiments/
├── notebooks/
└── docs/
    ├── architecture.md
    ├── nas_search_space.md
    ├── gp_reference.md
    ├── dataset.md
    ├── training.md
    ├── evaluation.md
    └── deployment.md
```

