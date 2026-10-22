import torch
from torch import nn
from .primitives import ComplexResidualBlock, RHSBottleneck


class RHSAdaptiveBeamformingModelSpace(nn.Module):
    """
    Lightweight dependency-safe NAS model space.

    The candidate operator list can be replaced by NNI LayerChoice when the
    NNI search backend is enabled.
    """

    def __init__(
        self,
        Nr=2,
        Nt=256,
        L=4,
        RFno=1,
        hidden_dim=256,
        depth=3,
        operator="bottleneck",
    ):
        super().__init__()
        self.Nr, self.Nt, self.L, self.RFno = Nr, Nt, L, RFno
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.input = nn.Linear(2 * L * Nr, hidden_dim)

        if operator == "complex_residual":
            block = lambda: ComplexResidualBlock(hidden_dim)
        elif operator == "bottleneck":
            block = lambda: RHSBottleneck(hidden_dim)
        else:
            raise ValueError(f"Unknown operator: {operator}")

        self.blocks = nn.Sequential(*[block() for _ in range(depth)])
        self.M_head = nn.Linear(hidden_dim, 2 * Nt * RFno)
        self.w_head = nn.Linear(hidden_dim, 2 * Nt * RFno)

    def forward(self, H):
        if H.ndim != 4:
            raise ValueError("H must be [B,L,Nr,Nt].")

        x = H.mean(dim=-1)  # [B,L,Nr]
        x = torch.cat([x.real, x.imag], dim=-1).reshape(H.shape[0], -1)
        x = self.input(x)
        x = self.blocks(x)

        M = self.M_head(x).view(-1, self.Nt, self.RFno, 2)
        w = self.w_head(x).view(-1, self.Nt, self.RFno, 2)

        return {
            "M": torch.complex(M[..., 0], M[..., 1]),
            "w": torch.complex(w[..., 0], w[..., 1]),
        }
