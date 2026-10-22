"""RHS Adaptive Beamforming NAS package.

This package contains the Neural Architecture Search (NAS) framework
for discovering optimal architectures for RHS adaptive beamforming.

Structure
---------
nas/
    search_space/   RHS-specific NAS primitives and model space
    strategies/     Search strategies (random, DARTS, ENAS, evolution)
    evaluators/     Multi-objective evaluators (rate, latency, constraints)
    objectives/     Individual NAS objective functions
    export/         Architecture export and fixed-model deployment
    runner.py       Top-level NAS search orchestrator
"""

__version__ = "0.1.0"

from rhs_beamform_nas.nas.runner import NASRunner

__all__ = ["NASRunner"]
