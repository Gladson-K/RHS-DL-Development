import torch
from rhs_beamforming.physics.rate import calculate_sum_rate


def test_rate_is_finite():
    H = torch.ones(1, 2, 4, dtype=torch.cfloat)
    W = torch.zeros(1, 4, 2, dtype=torch.cfloat)
    W[:, 0, 0] = 1
    W[:, 1, 1] = 1
    rate = calculate_sum_rate(H, W)
    assert torch.isfinite(rate)
    assert rate >= 0
