"""Tests for the five RHS architecture setups on the real target contract."""

import torch
from torch import nn

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


def spectral_target(y, k: int = 4, device=None):

    E_slice = (y.shape[1] - 1) // 2 if y.shape[1] % 2 == 1 else y.shape[1] // 2
    re = y[:, :E_slice].reshape(y.shape[0], -1, 10)
    im = y[:, E_slice : 2 * E_slice].reshape(y.shape[0], -1, 10)
    mc = torch.complex(re, im)
    gram = mc.conj().transpose(-1, -2) @ mc
    evals = torch.linalg.eigvalsh(gram)
    return torch.softmax(evals[:, -k:], dim=-1).to(device)


def test_jev_spectral_regularizer_gradients():
    """The consistency loss must backprop through the eigh decomposition."""
    torch.manual_seed(0)
    model = build_setup_by_name("hybrid_jev")
    model.train()
    x = torch.randn(3, 10)
    y = torch.randn(3, OUTPUT_DIM)
    out = model(x)
    main = nn.functional.mse_loss(out, y)
    head = model.head
    k = head.last_eigenvalues.shape[-1]
    w_model = torch.softmax(head.last_eigenvalues, dim=-1)
    target = spectral_target(y, k)
    spec = nn.functional.mse_loss(w_model, target)
    loss = main + 0.1 * spec
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
