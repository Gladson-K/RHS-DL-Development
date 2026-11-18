"""Static-sample fixture access (configuration + GP ground truth)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class StaticSamples:
    """Loads the small fixture written by ``platform/scripts/make_fixture.py``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.available = self.path.exists()
        self.data: Any = np.load(self.path, allow_pickle=True) if self.available else None

    def ids(self) -> list[str]:
        if not self.available:
            return []
        return [str(value) for value in self.data["sample_ids"]]

    def get(self, sample_id: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        ids = self.ids()
        if sample_id not in ids:
            return None
        index = ids.index(sample_id)
        return self._row(index)

    def nearest(self, config: list[float], tolerance: float = 1e-2) -> dict[str, Any] | None:
        if not self.available:
            return None
        configs = np.asarray(self.data["config_raw"], dtype=np.float32)
        target = np.asarray(config, dtype=np.float32)
        distances = np.max(np.abs(configs - target), axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > tolerance:
            return None
        return self._row(index)

    def _row(self, index: int) -> dict[str, Any]:
        matrix = np.asarray(self.data["gp_M"][index], dtype=np.complex64)
        return {
            "sample_id": self.ids()[index],
            "config": np.asarray(self.data["config"][index], dtype=np.float32).tolist(),
            "config_raw": np.asarray(self.data["config_raw"][index], dtype=np.float32).tolist(),
            "gp_rate": float(np.asarray(self.data["gp_rate"])[index]),
            "gp_M_real": np.round(matrix.real, 6).tolist(),
            "gp_M_imag": np.round(matrix.imag, 6).tolist(),
            "gp_M": matrix,
            "global_index": int(np.asarray(self.data["global_indices"])[index]),
        }

    def first(self) -> dict[str, Any] | None:
        ids = self.ids()
        return self.get(ids[0]) if ids else None
