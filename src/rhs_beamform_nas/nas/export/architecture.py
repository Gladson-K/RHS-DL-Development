"""Architecture specification descriptor and data models for RHS NAS.

Structures describing candidate and exported architectures with validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ArchitectureDescriptor:
    """Complete specification of a discovered or candidate RHS architecture."""

    hidden_dim: int = 256
    rhs_depth: int = 4
    spatial_operator: List[str] = field(default_factory=lambda: ["complex_residual", "spatial_gnn", "graph_attention", "residual"])
    attention_heads: int = 4
    conditioning: str = "film"
    M_representation: str = "amplitude_phase"
    w_representation: str = "real_imag"
    Nr: int = 64
    Nt: int = 4
    input_dim: int = 128
    config_dim: int = 16
    metrics: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary representation."""
        return {
            "architecture": {
                "hidden_dim": self.hidden_dim,
                "rhs_depth": self.rhs_depth,
                "spatial_operator": self.spatial_operator,
                "attention_heads": self.attention_heads,
                "conditioning": self.conditioning,
                "M_representation": self.M_representation,
                "w_representation": self.w_representation,
                "Nr": self.Nr,
                "Nt": self.Nt,
                "input_dim": self.input_dim,
                "config_dim": self.config_dim,
            },
            "metrics": self.metrics,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArchitectureDescriptor:
        """Create descriptor from dictionary or parsed JSON/YAML."""
        arch_data = data.get("architecture", data)
        metrics = data.get("metrics", {})
        score = data.get("score")

        return cls(
            hidden_dim=arch_data.get("hidden_dim", 256),
            rhs_depth=arch_data.get("rhs_depth", len(arch_data.get("spatial_operator", [None] * 4))),
            spatial_operator=arch_data.get("spatial_operator", ["complex_residual"] * 4),
            attention_heads=arch_data.get("attention_heads", 4),
            conditioning=arch_data.get("conditioning", "film"),
            M_representation=arch_data.get("M_representation", "amplitude_phase"),
            w_representation=arch_data.get("w_representation", "real_imag"),
            Nr=arch_data.get("Nr", 64),
            Nt=arch_data.get("Nt", 4),
            input_dim=arch_data.get("input_dim", 128),
            config_dim=arch_data.get("config_dim", 16),
            metrics=metrics,
            score=score,
        )

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Serialize to JSON string or write to file."""
        s = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Serialize to YAML string or write to file."""
        s = yaml.dump(self.to_dict(), sort_keys=False)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(s)
        return s
