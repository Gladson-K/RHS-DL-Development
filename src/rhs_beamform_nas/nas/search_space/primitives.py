from torch import nn


class RHSPrimitive(nn.Module):
    """
    Base class for RHS-specific NAS operators.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim


class ComplexResidualBlock(RHSPrimitive):

    def __init__(self, dim: int):
        super().__init__(dim)

        self.net = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, 2 * dim),
        )

    def forward(self, x):
        return x + self.net(x)


class RHSBottleneck(RHSPrimitive):

    def __init__(self, dim: int):
        super().__init__(dim)

        hidden = dim // 2

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)