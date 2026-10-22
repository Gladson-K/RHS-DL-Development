def test_rhs_search_space():

    from rhs_beamforming.nas.search_space.model_space import (
        RHSAdaptiveBeamformingModelSpace,
    )

    model = RHSAdaptiveBeamformingModelSpace(
        Nr=2,
        Nx=16,
        L=4,
        RFno=1,
    )

    assert model is not None