"""Fast smoke test for the multi-model NAS search (in-memory cache, tiny space)."""

import json

import numpy as np

from rhs_beamforming.data.mimo64 import OUTPUT_DIM
from rhs_beamforming.nas.real_objective import ScoreWeights
from rhs_beamforming.nas.real_search import MultiModelSearch, SearchConfig
from rhs_beamforming.nas.real_space import RealSearchSpace


def _tiny_cache(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10)).astype(np.float32)
    Y = rng.standard_normal((n, OUTPUT_DIM)).astype(np.float32)
    rate = rng.uniform(20, 60, size=n).astype(np.float32)
    Y[:, -1] = rate
    pos = np.arange(n)
    return {
        "X": X,
        "Y": Y,
        "rate": rate,
        "rate_mean": np.array([float(rate.mean())], dtype=np.float64),
        "rate_std": np.array([float(rate.std() or 1.0)], dtype=np.float64),
        "train": pos[:36],
        "val": pos[36:42],
        "test": pos[42:],
    }


def test_multi_model_search_ranks_and_exports(tmp_path):
    space = RealSearchSpace(
        hidden_layer_choices=[(64, 32), (48,)],
        activation_choices=["relu"],
        dropout_choices=[0.0],
        batchnorm_choices=[False],
        residual_choices=[False],
        head_choices=["dense"],
    )
    cfg = SearchConfig(
        strategy="random",
        num_candidates=2,
        budget_epochs=1,
        final_epochs=1,
        top_k=1,
        batch_size=16,
        halving=False,
        early_stop_patience=1,
        seed=0,
        output_dir=str(tmp_path / "search"),
        score=ScoreWeights(),
    )
    result = MultiModelSearch(space, _tiny_cache(), cfg).run()

    out = tmp_path / "search"
    assert (out / "architectures" / "best.json").exists()
    assert (out / "metrics" / "candidates.csv").exists()
    assert (out / "best.yaml").exists()

    best = json.loads((out / "architectures" / "best.json").read_text())
    assert best["architecture"]["output_dim"] == OUTPUT_DIM
    assert result["best"]["metrics"]["parameters"] > 0
