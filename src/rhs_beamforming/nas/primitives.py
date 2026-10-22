from torch import nn


class ComplexResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, 2 * dim),
        )

    def forward(self, x):
        return x + self.net(x)


class RHSBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)
