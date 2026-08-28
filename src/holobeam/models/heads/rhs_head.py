import torch
from torch import nn


class RHSHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        RFno: int,
    ):
        super().__init__()

        self.RFno = RFno

        self.output = nn.Linear(
            hidden_dim,
            2 * RFno,
        )

    def forward(self, x):

        raw = self.output(x)

        amplitude = torch.sigmoid(
            raw[..., : self.RFno]
        )

        phase = torch.pi * torch.tanh(
            raw[..., self.RFno:]
        )

        return amplitude, phase