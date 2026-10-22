import torch


def test_nas_candidate_forward():
    from rhs_beamforming.nas.search_space.model_space import (
        RHSAdaptiveBeamformingModelSpace,
    )
    model = RHSAdaptiveBeamformingModelSpace(
        Nr=2,
        Nx=16,
        L=4,
        RFno=1,
    )

    batch = create_dummy_batch()

    output = model(batch)

    assert "M" in output
    assert "w" in output

    assert torch.isfinite(
        output["M"]
    ).all()

    assert torch.isfinite(
        output["w"]
    ).all()