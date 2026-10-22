"""Shape / gradient tests for the Transformer, GNN and JEV blocks."""

import torch

from rhs_beamforming.models.gnn_block import GraphMessagePassingBlock
from rhs_beamforming.models.jev_head import JEVDecisionHead
from rhs_beamforming.models.transformer import (
    MultiHeadSelfAttention,
    TokenStem,
    TransformerEncoderBlock,
)

B, D, T = 3, 32, 8


def test_token_stem_shape():
    stem = TokenStem(input_dim=10, hidden_dim=D, num_tokens=T)
    out = stem(torch.randn(B, 10))
    assert out.shape == (B, T, D)


def test_transformer_block_residual_shape():
    block = TransformerEncoderBlock(D, heads=4)
    x = torch.randn(B, T, D)
    assert block(x).shape == x.shape


def test_multihead_self_attention_shape():
    attn = MultiHeadSelfAttention(D, heads=4)
    x = torch.randn(B, T, D)
    assert attn(x).shape == x.shape


def test_gnn_block_shape():
    block = GraphMessagePassingBlock(D, num_tokens=T)
    x = torch.randn(B, T, D)
    assert block(x).shape == x.shape


def test_jev_head_output_shape_and_eig_knob():
    head = JEVDecisionHead(dim=D, out_dim=50, num_eig=4)
    out = head(torch.randn(B, T, D))
    assert out.shape == (B, 50)
    assert head.last_eigenvalues.shape == (B, 4)


def test_jev_head_gradients_are_finite():
    torch.manual_seed(0)
    head = JEVDecisionHead(dim=D, out_dim=20, num_eig=4)
    out = head(torch.randn(B, T, D))
    out.pow(2).mean().backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_jev_eigenvalues_are_sorted_ascending():
    head = JEVDecisionHead(dim=D, out_dim=5, num_eig=4)
    head(torch.randn(B, T, D))
    ev = head.last_eigenvalues
    assert torch.all(ev[:, 1:] >= ev[:, :-1] - 1e-5)
