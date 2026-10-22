# RHS Beamforming Product - README

## Overview

This product integrates two approaches for Reconfigurable Holographic Surface (RHS) beamforming:

1. **Gradient Projection (GP) Algorithm** - Classical iterative optimization approach
2. **Deep Learning Model** - Trained `hybrid_jev` model (GNN + Transformer + JEV) on the full 15K dataset

## Architecture

```
rhs_product/
├── cpp/                    # C++ implementation of GP algorithm
│   ├── include/           # Header files
│   ├── src/               # Source files
│   ├── cmake/             # CMake build files
│   └── build/             # Compiled binaries
├── python/                # Python integration and API
│   ├── rhs_gp.py         # Python GP implementation
│   └── rhs_beamforming_product.py  # Unified product system
├── config/               # Configuration files
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## Features

### GP Algorithm (C++ Implementation)
- **Rayleigh Channel Generation**: Generates realistic fading channels
- **MMSE Beamforming**: Minimum Mean Square Error receive combining
- **Excitation Matrix Generation**: RHS excitation patterns
- **Gradient Projection Optimization**: Iterative optimization with convergence criteria
- **Sum Capacity Computation**: Achievable rate calculation

### Deep Learning Model
- **Hybrid Architecture**: GNN + Transformer with JEV (eigen) decision head
- **Trained on Full Dataset**: 15,000 samples with JEV spectral-consistency regularization
- **Performance**: Test M cos 0.777, rate R² 0.804, inference ~8-10ms
- **Real-time Inference**: Fast prediction for adaptive beamforming

### Unified Product System
- **Coordinate-based Input**: Accepts Base Station and User coordinates
- **Adaptive Beamforming**: Real-time beamforming for multiple users
- **Comparison Mode**: Compare GP and DL approaches side by side
- **Flexible API**: Easy integration for deployment

## Quick Start

### 1. Test C++ GP Implementation
```bash
cd rhs_product/cpp/build
./rhs_gp_test
```

### 2. Run Python Demo
```bash
cd rhs_product/python
PYTHONPATH=../../src python3 rhs_beamforming_product.py
```

### 3. Use as API
```python
from rhs_beamforming_product import RHSBeamformingProduct

# Initialize
rhs = RHSBeamformingProduct()

# Define positions
bs_coords = (0, 0, 10)
user_coords = [(50, 30, 1.5)]

# DL-based beamforming (fast, real-time)
result = rhs.adaptive_beamforming(bs_coords, user_coords, use_method='dl')
print(f"Rate: {result['users'][0]['rate']:.2f} bps/Hz")

# GP-based beamforming (iterative)
result = rhs.adaptive_beamforming(bs_coords, user_coords, use_method='gp', Nx=64)
print(f"Rate: {result['users'][0]['rate']:.2f} bps/Hz")

# Compare both
result = rhs.adaptive_beamforming(bs_coords, user_coords, use_method='compare')
```

## Performance Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **DL (hybrid_jev)** | ~8-10 ms | M cos 0.777, rate R² 0.804 | Real-time adaptive beamforming, edge deployment |
| **GP (iterative)** | ~50-500 ms | Near-optimal (iterative convergence) | Offline optimization, performance benchmarking, ground truth |

## System Requirements

- Python 3.10+
- PyTorch
- NumPy
- Eigen3 (for C++)
- CMake (for building C++)

## Configuration

The system uses the trained model from:
`search_results/mimo64_full_data/checkpoints/hybrid_jev.pt`

Features are normalized using scaler from:
`data/processed/mimo64_full/scaler.json`

## Next Steps for Production

1. **ONNX Export**: Export DL model to ONNX for faster C++ inference
2. **TensorRT**: Optimize for GPU deployment
3. **Real-time Integration**: Connect to SDR/BS hardware interfaces
4. **Channel Estimation**: Integrate with real channel estimation
5. **Multi-user Coordination**: Extend for MU-MIMO scenarios

## License

Research/Development use - RHS-DL-Development Project
