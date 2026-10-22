"""NAS package for RHS adaptive beamforming architecture search.

Sub-packages
------------
search_space    RHS-specific model space and primitives
strategies      Search algorithms (random, DARTS, ENAS, evolution)
evaluators      Architecture evaluators (rate, latency, constraints)
objectives      Individual and combined NAS objectives
export          Architecture export and fixed-model utilities
"""

from rhs_beamform_nas.nas.runner import NASRunner

__all__ = ["NASRunner"]
