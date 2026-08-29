from torch import nn


class RHSOperatorChoice(nn.Module):

    def __init__(self, dim):

        self.choices = nn.ModuleList([
            ComplexResidualBlock(dim),
            SpatialGNNBlock(dim),
            GraphAttentionBlock(dim),
            EfficientRHSBlock(dim),
        ])

