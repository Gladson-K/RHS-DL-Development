# RHS-DL-Development

## RHS Adaptive Beamforming — Deep Learning, HaloBeamGNNNet & NAS Research Framework

This repository contains the unified development environment for **Reconfigurable Holographic Surface (RHS) adaptive beamforming using Deep Learning**, including the **HaloBeamGNNNet beamforming architecture** and the **NAS-based RHS Adaptive Beamforming architecture search framework**.

The repository intentionally combines both development tracks into a single codebase so that the same physics, channel models, dataset generation, configuration system, testing infrastructure, experiments, and evaluation pipeline can be reused across architectures.

---

## Repository Scope

The project is organized around two complementary research setups:

### 1. HaloBeamGNNNet

The primary manually designed Deep Learning architecture for RHS adaptive beamforming.

It provides the baseline architecture and establishes the complete pipeline for:

```text
Channel / System Configuration
            ↓
       GP / Reference
            ↓
      Training Dataset
            ↓
       HaloBeamGNNNet
            ↓
      Beamforming Prediction
            ↓
       Rate Evaluation
            ↓
    GP vs DL Comparison
```

Complete implementation details, architecture design, configuration, training, testing, dataset generation, and execution procedures are maintained in:

**`README-HaloBeamGnnNet.md`**

---

### 2. RHS Adaptive Beamforming NAS Recipe

The NAS framework extends the HaloBeamGNNNet approach by automatically searching for an architecture optimized specifically for RHS adaptive beamforming.

The NAS search space is designed around communication-aware neural-network primitives rather than generic image-classification operations.

The search can consider:

* Complex-valued feature processing
* Real/imaginary representations
* RHS spatial processing
* GNN operations
* Graph attention
* Spatial attention
* Efficient blocks
* Residual blocks
* Configuration conditioning
* FiLM / cross-attention
* Network depth
* Hidden dimension
* Beamforming output heads
* Model size
* Inference latency
* Achievable rate
* Physical/beamforming constraints

Complete NAS architecture, search-space definition, search strategies, evaluation methodology, dataset sampling, result format, export and retraining procedures are maintained in:

**`README-NASRecipe.md`**

---

# Repository Structure

The repository is intentionally kept as a unified structure.

```text
RHS-DL-Development/
│
├── README.md
│   └── This master repository reference
│
├── README-HaloBeamGnnNet.md
│   └── Complete HaloBeamGNNNet setup and workflow
│
├── README-NASRecipe.md
│   └── Complete NAS Recipe setup and workflow
│
├── pyproject.toml
│   └── Python package and development configuration
│
├── Makefile.halobeamgnnNet
│   └── HaloBeamGNNNet workflow commands
│
├── Makefile.nasarc
│   └── NAS workflow commands
│
├── .devcontainer/
│   └── Development container configuration
│
├── .github/
│   └── GitHub Actions / CI-CD configuration
│
├── .dockerignore
├── .gitignore
├── .gitattributes
│
├── docker/
│   └── Docker-related environment/configuration
│
├── docker-compose.yaml
│   └── Containerized development/runtime configuration
│
├── configs/
│   └── System, dataset, model, training,
│       NAS and experiment configurations
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   └── Main RHS beamforming Python package
│
├── scripts/
│   └── Dataset generation, training,
│       NAS search, export and evaluation scripts
│
├── tests/
│   ├── unit/
│   ├── model/
│   ├── nas/
│   └── integration/
│
├── notebooks/
│   └── Research and experimental notebooks
│
├── experiments/
│   └── Experiment configurations and runs
│
├── checkpoints/
│   └── Model checkpoints
│
├── reports/
│   └── Experimental results and generated reports
│
└── docs/
    └── Supporting technical documentation
```

---

# Two-Track Development Model

Both research tracks share the same underlying project infrastructure.

```text
                       RHS-DL-Development
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
       HaloBeamGNNNet                   NAS Recipe
              │                               │
              │                        Custom RHS NAS
              │                        Search Space
              │                               │
              ▼                               ▼
       Fixed Architecture              Candidate Architectures
              │                               │
              └───────────────┬───────────────┘
                              ▼
                     Common Evaluation
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
             Rate          Latency        Parameters
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                       Final Comparison
```

The objective is therefore not to maintain two unrelated projects.

Instead:

```text
Common Physics
      +
Common Dataset
      +
Common Reference/GP
      +
HaloBeamGNNNet Baseline
      +
NAS Architecture Search
      ↓
RHS Adaptive Beamforming DL Framework
```

---

# Authoritative Documentation

The root README intentionally does **not duplicate the complete implementation documentation**.

Use the following documents as the authoritative references.

| Document                   | Purpose                                |
| -------------------------- | -------------------------------------- |
| `README.md`                | Repository overview and navigation     |
| `README-HaloBeamGnnNet.md` | Complete HaloBeamGNNNet implementation |
| `README-NASRecipe.md`      | Complete NAS Recipe implementation     |

### HaloBeamGNNNet Reference

See:

```text
README-HaloBeamGnnNet.md
```

for the complete HaloBeamGNNNet setup.

This document should be used when working on:

* HaloBeamGNNNet architecture
* Model components
* RHS/GNN processing
* Dataset generation
* Training
* Configuration
* Evaluation
* Experiments
* Baseline comparisons

### NAS Recipe Reference

See:

```text
README-NASRecipe.md
```

for the complete NAS setup.

This document should be used when working on:

* RHS-specific NAS search space
* NAS primitives
* Search strategies
* Architecture candidates
* Multi-objective evaluation
* Rate-aware architecture selection
* Latency-aware search
* Architecture export
* Architecture retraining
* NAS experiments
* Pareto analysis

---

# Configuration Philosophy

The repository uses configuration-driven experimentation.

Configurations should be preferred over hard-coded values wherever possible.

The major configurable components include:

```text
System
  ├── Nr
  ├── Nx
  ├── Nt
  ├── L
  ├── RFno
  └── Frequency

Channel
  ├── Channel model
  ├── SNR
  ├── Users
  └── Random seed

Dataset
  ├── Number of samples
  ├── Training split
  ├── Validation split
  ├── Test split
  └── NAS search subset

Model
  ├── Architecture
  ├── Hidden dimension
  ├── Depth
  ├── Attention
  ├── GNN configuration
  └── Output representation

Training
  ├── Epochs
  ├── Batch size
  ├── Learning rate
  ├── Loss weights
  └── Scheduler

NAS
  ├── Search space
  ├── Search strategy
  ├── Number of trials
  ├── Search dataset
  ├── Objective weights
  └── Model constraints
```

The exact configuration files and their usage are documented in the respective setup README files.

---

# Common Research Pipeline

Both approaches ultimately follow the same communication-system workflow.

```text
System Configuration
        ↓
Channel Generation
        ↓
Channel Validation
        ↓
MMSE / Reference Processing
        ↓
GP Reference Solution
        ↓
Training Dataset
        ↓
Deep Learning Model
        ↓
RHS Beamforming Prediction
        ↓
Beamforming Constraint / Normalization
        ↓
Achievable Rate Calculation
        ↓
Performance Evaluation
```

The NAS workflow adds an architecture-search stage:

```text
Training Dataset
       ↓
RHS NAS Model Space
       ↓
Architecture Search
       ↓
Candidate Architectures
       ↓
Fast Evaluation
       ↓
Pareto Selection
       ↓
Top Architectures
       ↓
Full Retraining
       ↓
Final Evaluation
```

---

# Research Comparison

The final research evaluation should allow comparison between:

```text
GP Reference
     │
     ├── HaloBeamGNNNet
     │
     └── NAS Optimized Model
```

with the major metrics including:

* Achievable rate
* Normalized rate
* GP rate ratio
* Beamforming prediction quality
* Physical/beamforming constraint violation
* Inference latency
* Parameter count
* Model size
* Generalization
* Robustness to different channel conditions

For NAS, the primary objective is **not simply neural-network prediction error**.

The architecture should ultimately be evaluated according to its usefulness for the RHS communication system.

---

# Development Entry Points

## HaloBeamGNNNet

Use:

```bash
make -f Makefile.halobeamgnnNet
```

Refer to:

```text
README-HaloBeamGnnNet.md
```

for the complete command reference.

---

## NAS

Use:

```bash
make -f Makefile.nasarc
```

Refer to:

```text
README-NASRecipe.md
```

for the complete NAS command reference.

---

# Testing

The repository contains separate test layers for the common physics/model infrastructure and NAS functionality.

```text
tests/
│
├── unit/
│   └── Fundamental functionality
│
├── model/
│   └── Model architecture tests
│
├── nas/
│   └── NAS search-space and evaluator tests
│
└── integration/
    └── End-to-end smoke tests
```

The testing philosophy is:

```text
Configuration
      ↓
Physics
      ↓
Data
      ↓
Model
      ↓
NAS
      ↓
Evaluation
```

Each stage should be independently testable before performing large-scale experiments.

---

# Development Environment

The project includes:

```text
.devcontainer/
```

for reproducible development environments.

Docker support is additionally provided through:

```text
docker/
docker-compose.yaml
```

The exact environment setup and dependencies should be taken from the repository configuration rather than manually recreating the environment.

---

# CI/CD

GitHub Actions configuration is maintained under:

```text
.github/workflows/
```

CI should validate:

```text
Code
 ↓
Lint
 ↓
Unit Tests
 ↓
Model Tests
 ↓
NAS Search-Space Tests
 ↓
Integration Smoke Tests
```

Large NAS searches and full training runs should remain separate from normal pull-request CI.

They should be executed as explicit research experiments using the NAS configuration and experiment infrastructure.

---

# Data and Experiment Management

Large generated datasets, checkpoints and experiment outputs should not be treated as source-code components.

The repository separates them into:

```text
data/
checkpoints/
experiments/
reports/
```

The source implementation remains under:

```text
src/
```

while executable research workflows are maintained under:

```text
scripts/
```

This separation allows experiments to be reproduced without coupling the implementation to a particular generated dataset or checkpoint.

---

# Recommended Research Lifecycle

The recommended development sequence is:

```text
1. Validate configuration
        ↓
2. Validate physics
        ↓
3. Validate GP/reference implementation
        ↓
4. Generate small debug dataset
        ↓
5. Run unit + integration tests
        ↓
6. Validate HaloBeamGNNNet
        ↓
7. Generate production dataset
        ↓
8. Train HaloBeamGNNNet baseline
        ↓
9. Validate NAS search space
        ↓
10. Run NAS search
        ↓
11. Select Pareto-optimal architectures
        ↓
12. Retrain selected architectures
        ↓
13. Evaluate against GP
        ↓
14. Compare HaloBeamGNNNet vs NAS
        ↓
15. Benchmark latency/model size
        ↓
16. Select final architecture
```

---

# Important Repository Principle

The repository contains two architecture-development paths, but **the underlying communication-system physics must remain common**.

The architecture may change:

```text
MLP
GNN
HaloBeamGNNNet
NAS-discovered architecture
```

but the following should remain consistently defined:

```text
System model
Channel model
RHS geometry
Reference/GP formulation
Beamforming constraints
Rate calculation
Evaluation methodology
```

This ensures that architecture comparisons remain scientifically meaningful.

---

# NAS Architecture Principle

The NAS framework should search **RHS-adaptive-beamforming-specific neural operations**, rather than treating the problem as a generic image-classification NAS task.

The intended search space includes:

```text
Complex Processing
       +
RHS Spatial Processing
       +
GNN
       +
Graph/Spatial Attention
       +
Efficient Blocks
       +
Configuration Conditioning
       +
Beamforming Output Heads
       +
Depth / Width
       ↓
RHS Adaptive Beamforming Architecture
```

The complete search-space specification is maintained in:

```text
README-NASRecipe.md
```

---

# Reference Architecture Relationship

The repository should be viewed as:

```text
                         GP
                          │
                          ▼
                  Reference Solution
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
       HaloBeamGNNNet          NAS Model Space
                │                   │
                │             ┌─────┼─────┐
                │             ▼     ▼     ▼
                │           DARTS ENAS Random
                │             │     │     │
                │             └─────┼─────┘
                │                   ▼
                │            Candidate Models
                │                   │
                │                   ▼
                │             Pareto Search
                │                   │
                └───────────┬───────┘
                            ▼
                   Final Evaluation
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
            Rate         Latency       Parameters
```

---

# Current Repository Status

This repository represents the combined development branch:

```text
DL-RHS-HoloBeamGNNNet-dev
```

The repository currently contains both:

```text
HaloBeamGNNNet
```

and

```text
NAS Recipe
```

development structures.

The root README is intentionally maintained as a **navigation and project-level reference**, while detailed implementation instructions remain in their dedicated README files.

---

# Quick Reference

```text
Repository
    │
    ├── HaloBeamGNNNet
    │      └── README-HaloBeamGnnNet.md
    │
    └── NAS Recipe
           └── README-NASRecipe.md
```

### Start with HaloBeamGNNNet

```text
README-HaloBeamGnnNet.md
```

### Start with NAS

```text
README-NASRecipe.md
```

### Source code

```text
src/
```

### Configuration

```text
configs/
```

### Scripts

```text
scripts/
```

### Tests

```text
tests/
```

### Containers

```text
.devcontainer/
docker/
docker-compose.yaml
```

### CI/CD

```text
.github/
```

### Experiments

```text
experiments/
reports/
checkpoints/
```

---

# Final Project Objective

The long-term objective of this repository is to develop a **fast, accurate and configurable Deep Learning framework for RHS adaptive beamforming**, capable of replacing expensive iterative optimization during inference while maintaining communication-system performance close to the reference solution.

The development therefore progresses from:

```text
Classical Optimization
        ↓
HaloBeamGNNNet
        ↓
NAS-Optimized RHS Architecture
        ↓
Rate / Latency / Complexity Optimization
        ↓
Deployment-Ready Adaptive Beamforming
```

The two detailed implementation documents remain the authoritative references for their respective workflows.


## Debugging

## First validation

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/generate_dataset.py --samples 20
python scripts/search_nas.py --max-trials 2 --epochs 1 --samples 20
```

## Important

Before production experiments:

1. Preserve the validated project-specific GP/reference implementation.
2. Replace the generic synthetic dataset with the actual channel/data pipeline.
3. Connect the actual RHS/GNN formulation from `README-HaloBeamGnnNet.md`.
4. Connect the complete NAS search space from `README-NASRecipe.md`.
5. Validate all tensor dimensions before large dataset generation.
6. Do not use the scaffold's generic rate function as a substitute for the
   project's exact system-rate equation without validation.
