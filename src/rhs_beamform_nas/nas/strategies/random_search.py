"""Random search strategy for RHS NAS.

Stage 1 in the NAS Recipe:
Randomly samples architecture configurations from the search space to
validate the search space and establish baseline distributions (20-50 trials).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from rhs_beamform_nas.nas.search_space.model_space import (
    DEPTH_CHOICES,
    HIDDEN_DIM_CHOICES,
    CONDITIONING_CHOICES,
    M_REPR_CHOICES,
    W_REPR_CHOICES,
)
from rhs_beamform_nas.nas.search_space.rhs_adaptive_cell import DEFAULT_CANDIDATES


class RandomSearchStrategy:
    """Random architecture generator and explorer for RHS NAS.

    Generates random architecture specifications conforming to the RHS
    search space definitions.
    """

    def __init__(
        self,
        num_trials: int = 30,
        seed: Optional[int] = None,
        candidate_ops: Optional[List[str]] = None,
        depth_choices: Optional[List[int]] = None,
        hidden_dim_choices: Optional[List[int]] = None,
    ) -> None:
        self.num_trials = num_trials
        self.seed = seed
        self.candidate_ops = candidate_ops or DEFAULT_CANDIDATES
        self.depth_choices = depth_choices or DEPTH_CHOICES
        self.hidden_dim_choices = hidden_dim_choices or HIDDEN_DIM_CHOICES
        if seed is not None:
            random.seed(seed)

    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a single random architecture configuration dictionary."""
        depth = random.choice(self.depth_choices)
        hidden_dim = random.choice(self.hidden_dim_choices)
        conditioning = random.choice(CONDITIONING_CHOICES)
        m_repr = random.choice(M_REPR_CHOICES)
        w_repr = random.choice(W_REPR_CHOICES)

        # Sample operator for each cell up to max depth or chosen depth
        spatial_operators = [random.choice(self.candidate_ops) for _ in range(depth)]

        return {
            "hidden_dim": hidden_dim,
            "rhs_depth": depth,
            "spatial_operator": spatial_operators,
            "conditioning": conditioning,
            "M_representation": m_repr,
            "w_representation": w_repr,
        }

    def generate_trials(self) -> List[Dict[str, Any]]:
        """Generate a list of unique or sampled architecture trials."""
        trials = []
        for _ in range(self.num_trials):
            trials.append(self.sample_architecture())
        return trials
