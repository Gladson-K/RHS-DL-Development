from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.models import HaloBeamGNNNet


def test_single_sample_pipeline():
    data = generate_debug_dataset(samples=1)
    model = HaloBeamGNNNet()
    out = model(data["H"])
    assert out["M"].numel() > 0
    assert out["w"].numel() > 0
