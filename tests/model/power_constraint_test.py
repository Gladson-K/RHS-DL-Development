import torch

def test_beamformer_power_constraint():

    w = torch.randn(
        2,
        4,
        256,
        dtype=torch.cfloat,
    )

    w_normalized = normalize_beamformer(
        w,
        power=1.0,
    )

    power = (
        w_normalized.abs() ** 2
    ).sum()

    assert torch.isclose(
        power,
        torch.tensor(1.0),
        atol=1e-4,
    )