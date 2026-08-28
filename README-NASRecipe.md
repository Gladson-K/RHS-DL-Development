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

## Explanation for NAS Recipe

* Define RHSAdaptiveBEamformingSearchSpace and search space should not decide arbitrary neural-network operations.

* It should choose among **RHS beamforming-specific primitives**

```
RHSAdaptiveBeamformingSearchSpace
│
├── Complex processing
│   ├── ComplexLinear
│   ├── RealImagLinear
│   ├── ComplexResidual
│   └── ComplexNorm
│
├── Spatial processing
│   ├── LocalRHS
│   ├── SpatialGNN
│   ├── GraphAttention
│   └── GlobalAttention
│
├── Efficient processing
│   ├── MBConv
│   ├── Bottleneck
│   ├── Depthwise
│   └── Residual
│
├── Configuration conditioning
│   ├── Add
│   ├── FiLM
│   ├── CrossAttention
│   └── Concatenate
│
├── Depth
│   ├── 2
│   ├── 4
│   ├── 6
│   └── 8
│
├── Hidden dimension
│   ├── 128
│   ├── 192
│   ├── 256
│   └── 384
│
└── Heads
    ├── M
    ├── w
    └── rate
```

* Optimize Path for NASNet

```
RHSAdaptiveBeamformingSearchSpace
│
├── Complex processing
│   ├── ComplexLinear
│   ├── RealImagLinear
│   ├── ComplexResidual
│   └── ComplexNorm
│
├── Spatial processing
│   ├── LocalRHS
│   ├── SpatialGNN
│   ├── GraphAttention
│   └── GlobalAttention
│
├── Efficient processing
│   ├── MBConv
│   ├── Bottleneck
│   ├── Depthwise
│   └── Residual
│
├── Configuration conditioning
│   ├── Add
│   ├── FiLM
│   ├── CrossAttention
│   └── Concatenate
│
├── Depth
│   ├── 2
│   ├── 4
│   ├── 6
│   └── 8
│
├── Hidden dimension
│   ├── 128
│   ├── 192
│   ├── 256
│   └── 384
│
└── Heads
    ├── M
    ├── w
    └── rate
```


**PyTorch's multi-objective NAS material explicitly describes the common deployment trade-off between performance and competing objectives such as latency, power and model size**

```
NAS Objective
=
α × Rate
+
β × GP similarity
-
γ × Latency
-
δ × Parameters
-
ε × ConstraintViolation
```

Formally in terms of simplied version as 

```python
score = (
    alpha * normalized_rate
    + beta * beamforming_similarity
    - gamma * latency_ms
    - delta * parameter_million
    - epsilon * constraint_violation
)
```

# NAS Architecture Search Flow

```
                 Dataset
                    │
                    ▼
              GP teacher
                    │
                    ▼
            Training samples
                    │
                    ▼
        RHSAdaptiveBeamforming
              Model Space
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
      DARTS        ENAS       Random
        │           │           │
        └───────────┼───────────┘
                    ▼
             Candidate models
                    │
                    ▼
              Fast evaluator
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
      Rate       Latency      Parameters
        │           │            │
        └───────────┼────────────┘
                    ▼
             Pareto frontier
                    │
                    ▼
              Best architecture
                    │
                    ▼
              Full retraining
                    │
                    ▼
             Final benchmark
```

# NAS Implementation

* Stage 1 - Random NAS 

* Seach from 20 upto 50 Architectures

Purpose:
* Validate Search Space

* Stage 2 - One-Shot NAS

Use
```
ENAS/rRandomOneShot or DARTS
```

* NNI currently supports DARTS, ENAS and other one-shot strategies, and documents the distinction between multi-trial and weight-sharing one-shot search.

* Stage 3 - Pareto/Evolutionary Search

Search

```
rate
latency
parameters
```
* This is where you obtain 

```
Accuracy-optimal
Balanced
Latency-optimal
```
### NAS Evaluation

* NAS Evaluator 500 - 1000 respective samples and choose the best 5 architecture setup.
* use 5 architectures for full 15K training and full test validation

```
GP:
10K labels

NAS:
500–1000 representative scenarios

Final training:
15K

Final test:
held-out test set
```

### Dataset Sampling for NAS

create

```bash
data/splits/nas_search.csv
```

which is a stratified subset

```
frequency
SNR
channel seed
user configuration
geometry
```

### NAS Result Format

Every NAS Search Should Generate

```
search_results/
└── 2026-08-28_holobeam_nas/
    │
    ├── config.yaml
    ├── search_space.yaml
    ├── search.log
    │
    ├── architectures/
    │   ├── arch_0001.json
    │   ├── arch_0002.json
    │   └── best.json
    │
    ├── metrics/
    │   ├── candidates.csv
    │   ├── pareto.csv
    │   └── best.json
    │
    ├── checkpoints/
    │   └── supernet.pt
    │
    └── plots/
        ├── rate_vs_latency.png
        ├── params_vs_rate.png
        └── pareto_frontier.png
```

### Example Architecture Result

'best.json'

```json
{
  "architecture": {
    "hidden_dim": 256,
    "rhs_depth": 4,
    "spatial_operator": [
      "gnn",
      "gnn",
      "graph_attention",
      "complex_residual"
    ],
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

* It will be the research artifact

### NAS Export 

Search Must end with :

```python
python scripts/export_architecture.py \
    --search-result search_results/... \
    --output configs/model/nas_best.yaml
```

and then

```python
python scripts/retrain_architecture.py \
    --architecture configs/model/nas_best.yaml \
    --dataset data/processed/v1
```

This gives us a clean separation

```
python scripts/retrain_architecture.py \
    --architecture configs/model/nas_best.yaml \
    --dataset data/processed/v1
```

After CI/CD Test

New Developer needs

```bash
make dev
make test
```


## Development Lifecycle

```
                  Git Repository
                        │
                        ▼
                 .devcontainer
                        │
                        ▼
                  pytest + lint
                        │
                        ▼
                 GP validation
                        │
                        ▼
              20-sample debug data
                        │
                        ▼
                  10K GP dataset
                        │
                        ▼
             ┌───────────────────┐
             │ RHS NAS Search    │
             └─────────┬─────────┘
                       │
           ┌───────────┼────────────┐
           ▼           ▼            ▼
         DARTS       ENAS        Evolution
           │           │            │
           └───────────┼────────────┘
                       ▼
                Pareto frontier
                       │
                       ▼
               Top 3–5 models
                       │
                       ▼
                 Full retrain
                       │
                       ▼
                 GP comparison
                       │
           ┌───────────┼────────────┐
           ▼           ▼            ▼
        Rate       Latency       Params
           │           │            │
           └───────────┼────────────┘
                       ▼
                  Best model
                       │
                       ▼
                 Export / ONNX
                       │
                       ▼
                  Deployment
```