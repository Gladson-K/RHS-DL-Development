from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.models import HaloBeamGNNNet


def test_holobeam_forward():
    data = generate_debug_dataset(samples=2)
    model = HaloBeamGNNNet()
    out = model(data["H"])
    assert out["M"].shape == (2, 256, 1)
    assert out["w"].shape == (2, 256, 1)
