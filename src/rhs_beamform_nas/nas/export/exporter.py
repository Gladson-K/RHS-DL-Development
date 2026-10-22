"""Architecture exporter for RHS NAS.

Exports discovered optimal architectures to YAML/JSON configs and ready-to-train
models according to the NAS Recipe artifact specification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from rhs_beamform_nas.nas.export.architecture import ArchitectureDescriptor


class ArchitectureExporter:
    """Manages serialization and artifact generation for NAS search results."""

    def __init__(self, output_dir: str = "search_results") -> None:
        self.output_dir = Path(output_dir)

    def export_best(
        self,
        architecture: Dict[str, Any] | ArchitectureDescriptor,
        metrics: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Path:
        """Save best architecture to JSON and optionally YAML config."""
        if isinstance(architecture, dict):
            desc = ArchitectureDescriptor.from_dict(architecture)
            if metrics:
                desc.metrics.update(metrics)
        else:
            desc = architecture

        if save_path is None:
            out_path = self.output_dir / "best.json"
        else:
            out_path = Path(save_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        desc.to_json(str(out_path))

        # Also write yaml companion
        yaml_path = out_path.with_suffix(".yaml")
        desc.to_yaml(str(yaml_path))

        return out_path

    def export_pareto_candidates(
        self,
        candidates: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
    ) -> List[Path]:
        """Save candidate architectures and summary CSV for Pareto frontier."""
        target_dir = Path(save_dir) if save_dir else self.output_dir / "architectures"
        target_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = []
        for i, cand in enumerate(candidates):
            desc = ArchitectureDescriptor.from_dict(cand)
            file_path = target_dir / f"arch_{i+1:04d}.json"
            desc.to_json(str(file_path))
            exported_paths.append(file_path)

        return exported_paths
