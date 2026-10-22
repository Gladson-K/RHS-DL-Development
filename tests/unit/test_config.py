from rhs_beamforming.config.loader import merge_configs


def test_merge_configs():
    a = {"model": {"hidden": 128}, "seed": 1}
    b = {"model": {"depth": 4}}
    out = merge_configs(a, b)
    assert out["model"] == {"hidden": 128, "depth": 4}
    assert out["seed"] == 1
