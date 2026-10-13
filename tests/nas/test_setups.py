"""Tests for the five RHS architecture setups on the real target contract."""

import torch

from rhs_beamforming.nas.real_space import OUTPUT_DIM
from rhs_beamforming.nas.setups import (
    SETUP_VARIANTS,
    build_setup_by_name,
    setup_table,
)


def test_all_setups_build_and_forward():
    x = torch.randn(2, 10)
    for name in SETUP_VARIANTS:
        model = build_setup_by_name(name)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, OUTPUT_DIM), (name, out.shape)
        assert torch.isfinite(out).all(), name


def test_setup_table_reports_parameter_counts():
    rows = {r["setup"]: r for r in setup_table()}
    assert set(rows) == {"mlp", "gnn", "transformer", "hybrid", "hybrid_jev"}
    for name, row in rows.items():
        assert row["parameters"] > 0, name
    # token models are larger than the compact MLP baseline
    assert rows["gnn"]["parameters"] > rows["mlp"]["parameters"]


def test_hybrid_jev_has_eigen_knob():
    assert SETUP_VARIANTS["hybrid_jev"].family == "hybrid_jev"
    assert SETUP_VARIANTS["hybrid_jev"].num_eig >= 1
    assert SETUP_VARIANTS["mlp"].family == "mlp"


def test_setups_are_trainable_step():
    model = build_setup_by_name("hybrid")
    model.train()
    x = torch.randn(4, 10)
    y = torch.randn(4, OUTPUT_DIM)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
