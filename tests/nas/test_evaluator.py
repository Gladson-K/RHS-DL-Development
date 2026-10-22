from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.nas.model_space import RHSAdaptiveBeamformingModelSpace
from rhs_beamforming.nas.evaluator import evaluate_model


def test_evaluator():
    data = generate_debug_dataset(samples=2)
    model = RHSAdaptiveBeamformingModelSpace(hidden_dim=64, depth=2)
    metrics = evaluate_model(model, data["H"])
    assert metrics["parameters"] > 0
    assert metrics["latency_ms"] >= 0
