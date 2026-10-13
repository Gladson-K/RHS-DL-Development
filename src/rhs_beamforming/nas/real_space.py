"""RHS-specific NAS search space for the tabular config -> (M, rate) task.

The reference ver1.0-CNN maps 10 configuration features to
``[Re(M), Im(M), sum_rate]`` (25601 outputs). This module exposes a small but
meaningful NAS space around that architecture family: depth/width, activation,
regularisation, residual connections, and output-head style.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from torch import nn

ELEMENTS_PER_SAMPLE = 20 * 64 * 10  # 12800
OUTPUT_DIM = 2 * ELEMENTS_PER_SAMPLE + 1  # 25601
INPUT_DIM = 10

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


@dataclass
class Architecture:
    """A candidate architecture descriptor (JSON-serialisable)."""

    hidden_layers: tuple[int, ...] = (1024, 512, 256, 128, 64)
    activation: str = "leaky_relu"
    dropout: float = 0.2
    batchnorm: bool = True
    residual: bool = True
    head: str = "deep"  # dense | residual | deep
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_layers"] = list(self.hidden_layers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Architecture:
        d = dict(d)
        if "hidden_layers" in d:
            d["hidden_layers"] = tuple(d["hidden_layers"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def name(self) -> str:
        widths = "x".join(str(h) for h in self.hidden_layers)
        return f"h{widths}-{self.activation}-d{self.dropout}-{'bn' if self.batchnorm else 'nobn'}-{self.head}"


@dataclass
class RealSearchSpace:
    """Discrete search space over the tabular beamforming architecture."""

    hidden_layer_choices: list[tuple[int, ...]] = field(
        default_factory=lambda: [
            (1024, 512, 256, 128, 64),   # ver1.0-CNN reference
            (512, 256, 128),
            (768, 384, 192, 96),
            (1024, 768, 384, 192),
            (512, 512, 256, 128),
            (256, 256, 128, 64),
            (1024, 512, 256),
            (384, 192, 96, 48),
            (2048, 1024, 512, 256),
            (640, 320, 160, 80),
        ]
    )
    activation_choices: list[str] = field(
        default_factory=lambda: ["relu", "leaky_relu", "gelu", "prelu", "elu", "silu"]
    )
    dropout_choices: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    batchnorm_choices: list[bool] = field(default_factory=lambda: [True, False])
    residual_choices: list[bool] = field(default_factory=lambda: [True, False])
    head_choices: list[str] = field(default_factory=lambda: ["dense", "residual", "deep"])

    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM

    def sample(self, rng) -> Architecture:
        import random

        r = rng if rng is not None else random
        return Architecture(
            hidden_layers=r.choice(self.hidden_layer_choices),
            activation=r.choice(self.activation_choices),
            dropout=r.choice(self.dropout_choices),
            batchnorm=r.choice(self.batchnorm_choices),
            residual=r.choice(self.residual_choices),
            head=r.choice(self.head_choices),
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def mutate(self, arch: Architecture, rng) -> Architecture:
        import random

        r = rng if rng is not None else random
        d = arch.to_dict()
        field_names = ["hidden_layers", "activation", "dropout", "batchnorm", "residual", "head"]
        key = r.choice(field_names)
        if key == "hidden_layers":
            d[key] = r.choice(self.hidden_layer_choices)
        elif key == "activation":
            d[key] = r.choice(self.activation_choices)
        elif key == "dropout":
            d[key] = r.choice(self.dropout_choices)
        elif key == "batchnorm":
            d[key] = r.choice(self.batchnorm_choices)
        elif key == "residual":
            d[key] = r.choice(self.residual_choices)
        elif key == "head":
            d[key] = r.choice(self.head_choices)
        return Architecture.from_dict(d)

    def crossover(self, a: Architecture, b: Architecture, rng) -> Architecture:
        import random

        r = rng if rng is not None else random
        da, db = a.to_dict(), b.to_dict()
        child = {k: (da[k] if r.random() < 0.5 else db[k]) for k in da}
        return Architecture.from_dict(child)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str, dropout: float, batchnorm: bool):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity()
        self.act = ACTIVATIONS[activation]()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.drop(self.act(self.bn(self.fc(x))))


class RHSTabularNet(nn.Module):
    """Configurable MLP that predicts flattened Re/Im(M) and the sum-rate."""

    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch = arch
        layers: list[nn.Module] = []
        prev = arch.input_dim
        for i, h in enumerate(arch.hidden_layers):
            if arch.residual and i > 0 and h == prev:
                layers.append(ResidualMLPBlock(prev, h, arch.activation, arch.dropout, arch.batchnorm))
            else:
                if arch.residual and i > 0:
                    layers.append(ResidualMLPBlock(prev, h, arch.activation, arch.dropout, arch.batchnorm))
                else:
                    layers.append(nn.Linear(prev, h))
                    if arch.batchnorm:
                        layers.append(nn.BatchNorm1d(h))
                    layers.append(ACTIVATIONS[arch.activation]())
                    layers.append(nn.Dropout(arch.dropout))
            prev = h

        if arch.head == "deep":
            mid = max(prev // 2, 8)
            layers.append(nn.Linear(prev, mid))
            layers.append(ACTIVATIONS[arch.activation]())
            layers.append(nn.Dropout(max(arch.dropout * 0.5, 0.0)))
            layers.append(nn.Linear(mid, arch.output_dim))
        elif arch.head == "residual":
            layers.append(ResidualMLPBlock(prev, prev, arch.activation, arch.dropout, arch.batchnorm))
            layers.append(nn.Linear(prev, arch.output_dim))
        else:  # dense
            layers.append(nn.Linear(prev, arch.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(arch: Architecture) -> RHSTabularNet:
    return RHSTabularNet(arch)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
