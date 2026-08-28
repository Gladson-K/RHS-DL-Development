import torch
from rhs_beamforming.physics.beamforming import normalize_beamformer, power_constraint


def test_power_normalization():
    w = torch.ones(2, 4, dtype=torch.cfloat)
    out = normalize_beamformer(w, power=1.0)
    assert power_constraint(out, power=1.0)
