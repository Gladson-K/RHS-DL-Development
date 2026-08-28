from nni.nas.nn.pytorch import ModelSpace, LayerChoice


class RHSAdaptiveCell(ModelSpace):

    def __init__(self, dim):

        self.op = LayerChoice(
            [
                ComplexResidualBlock(dim),
                RHSBottleneck(dim),
                SpatialGNNBlock(dim),
                SpatialAttentionBlock(dim),
            ],
            label="rhs_operator"
        )

    def forward(self, x):
        return self.op(x)