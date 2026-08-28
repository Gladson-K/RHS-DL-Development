import torch
from torch import nn


class ComplexFeatureStem(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        xr = torch.cat([x.real, x.imag], dim=-1)
        return self.proj(xr)


class HaloBeamGNNNet(nn.Module):
    """
    Development baseline.

    This intentionally provides a stable tensor contract while the detailed
    RHS/GNN formulation remains governed by README-HaloBeamGnnNet.md.
    """

    def __init__(self, Nr=2, Nt=256, L=4, RFno=1, hidden_dim=256):
        super().__init__()
        self.Nr, self.Nt, self.L, self.RFno = Nr, Nt, L, RFno

        self.encoder = ComplexFeatureStem(Nr * L, hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.M_head = nn.Linear(hidden_dim, 2 * Nt * RFno)
        self.w_head = nn.Linear(hidden_dim, 2 * Nt * RFno)

    def forward(self, H):
        if H.ndim != 4:
            raise ValueError("H must have shape [B,L,Nr,Nt].")
        if H.shape[1:] != (self.L, self.Nr, self.Nt):
            raise ValueError(
                f"Expected [B,{self.L},{self.Nr},{self.Nt}], got {tuple(H.shape)}"
            )

        # Aggregate RHS/spatial dimension while preserving complex information.
        x = H.permute(0, 3, 1, 2).reshape(H.shape[0], self.Nt, self.L * self.Nr)
        x = self.encoder(x).mean(dim=1)
        x = self.backbone(x)

        M = self.M_head(x).view(-1, self.Nt, self.RFno, 2)
        w = self.w_head(x).view(-1, self.Nt, self.RFno, 2)

        M = torch.complex(M[..., 0], M[..., 1])
        w = torch.complex(w[..., 0], w[..., 1])

        return {"M": M, "w": w}
