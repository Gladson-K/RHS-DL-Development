"""Model registry and inference engine (GP reference, DL models, JEV decision)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from protocol.jev import ModelProfile, decide, signature_from_gram, spectral_signature

E = 20 * 64 * 10  # 12800
M_SHAPE = (20, 64, 10)


class EngineUnavailable(RuntimeError):
    """Raised when models could not be loaded."""


class UnknownModel(KeyError):
    """Raised when a requested model is not registered."""


class GpTruthUnavailable(LookupError):
    """Raised when no GP ground truth exists for the requested configuration."""


@dataclass
class ModelEntry:
    name: str
    kind: str
    checkpoint: str
    model: Any = None
    latency_ms: float = 0.0
    params: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    profile: ModelProfile | None = None
    available: bool = False
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "checkpoint": self.checkpoint,
            "latency_ms": round(self.latency_ms, 4),
            "parameters": self.params,
            "metrics": self.metrics,
            "centroid": list(self.profile.centroid) if self.profile else [],
            "available": self.available,
            "error": self.error,
        }


class Engine:
    """Loads the DL checkpoints once and serves deterministic predictions."""

    def __init__(self, settings):
        self.settings = settings
        self.loaded = False
        self.load_error: str | None = None
        self.models: dict[str, ModelEntry] = {}
        self.active_model: str | None = None
        self.rate_mean = settings.rate_mean
        self.rate_std = settings.rate_std
        self._torch = None
        self._fixture: Any = None
        self._scaler_mean = np.zeros(10, dtype=np.float32)
        self._scaler_scale = np.ones(10, dtype=np.float32)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        try:
            self._load()
            self.loaded = True
        except Exception as exc:  # surfaced via /healthz
            self.load_error = f"{type(exc).__name__}: {exc}"
            if self.settings.strict:
                raise

    def _load(self) -> None:
        import torch

        self._torch = torch
        sys.path.insert(0, self.settings.repo_src)
        from rhs_beamforming.nas.real_space import Architecture, build_model, count_parameters
        from rhs_beamforming.nas.setups import build_setup_by_name

        self._fixture = np.load(self.settings.fixture, allow_pickle=True)
        self.rate_mean = float(np.asarray(self._fixture["rate_mean"]).reshape(-1)[0])
        self.rate_std = float(np.asarray(self._fixture["rate_std"]).reshape(-1)[0])
        meta_path = Path(self.settings.meta)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.rate_mean = float(meta.get("rate_mean", self.rate_mean))
            self.rate_std = float(meta.get("rate_std", self.rate_std))
        scaler_path = Path(self.settings.scaler)
        if scaler_path.exists():
            scaler = json.loads(scaler_path.read_text())
            self._scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
            self._scaler_scale = np.asarray(scaler["scale"], dtype=np.float32)
            self._scaler_scale = np.where(self._scaler_scale < 1e-6, 1.0, self._scaler_scale)

        hybrid = ModelEntry(
            "hybrid_jev", "dl", self.settings.hybrid_checkpoint, metrics=_read_json_metrics(
                Path(self.settings.hybrid_checkpoint).parents[1] / "results.json"
            )
        )
        try:
            model = build_setup_by_name("hybrid_jev")
            state = torch.load(self.settings.hybrid_checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            hybrid.model = model
            hybrid.params = count_parameters(model)
            hybrid.available = True
        except Exception as exc:  # noqa: BLE001
            hybrid.error = f"{type(exc).__name__}: {exc}"
        self.models["hybrid_jev"] = hybrid

        nas = ModelEntry(
            "nas_best",
            "dl",
            self.settings.nas_checkpoint,
            metrics=_read_json_metrics(
                Path(self.settings.nas_checkpoint).with_suffix(".metrics.json")
            ),
        )
        try:
            payload = torch.load(self.settings.nas_checkpoint, map_location="cpu")
            arch = Architecture.from_dict(payload["arch"])
            model = build_model(arch)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            nas.model = model
            nas.params = count_parameters(model)
            nas.available = True
        except Exception as exc:  # noqa: BLE001
            nas.error = f"{type(exc).__name__}: {exc}"
        self.models["nas_best"] = nas

        gp = ModelEntry("gp", "gp", self.settings.fixture, available=True)
        self.models["gp"] = gp

        for entry in self.models.values():
            if entry.kind == "dl" and not entry.available:
                continue
            if not entry.latency_ms:
                entry.latency_ms = self._measure_latency(entry)
            entry.profile = self._calibrate(entry)
        self.active_model = "hybrid_jev" if self.models["hybrid_jev"].available else "gp"

    def _calibrate(self, entry: ModelEntry) -> ModelProfile:
        config_raw = np.asarray(self._fixture["config_raw"][0], dtype=np.float32)
        try:
            if entry.kind == "gp":
                matrix = np.asarray(self._fixture["gp_M"][0], dtype=np.complex128)
            else:
                matrix, _ = self._forward(entry, config_raw)
            gram = _gram(matrix)
            centroid = signature_from_gram(gram, k=self.settings.jev_k)
        except Exception:  # calibration is best-effort  # noqa: BLE001
            centroid = [1.0 / self.settings.jev_k] * self.settings.jev_k
        return ModelProfile(
            name=entry.name,
            centroid=tuple(centroid),
            latency_ms=entry.latency_ms,
            kind=entry.kind,
        )

    def _measure_latency(self, entry: ModelEntry) -> float:
        if entry.kind == "gp" or entry.model is None:
            return 0.0
        config_raw = np.asarray(self._fixture["config_raw"][0], dtype=np.float32)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            self._forward(entry, config_raw)
            samples.append((time.perf_counter() - start) * 1000.0)
        return float(sorted(samples)[len(samples) // 2])

    # --------------------------------------------------------------- predict
    def predict(
        self,
        config: list[float],
        model: str = "auto",
        include_m: bool = True,
        probe: list[float] | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise EngineUnavailable(self.load_error or "engine not loaded")
        decision = None
        chosen = model
        if model == "auto":
            decision = self.jev_decide(config, probe=probe)
            chosen = str(decision["chosen_model"])
        if chosen == "gp":
            matrix, rate, neat = self._gp_truth(config)
            result = {
                "model": "gp",
                "kind": "gp",
                "rate": float(rate),
                "latency_ms": 0.0,
                "M_shape": list(M_SHAPE),
                "M_norm": float(np.linalg.norm(matrix)),
                "M_hash": _hash_matrix(matrix),
                "matrix": matrix,
                "_nearest": neat,
            }
        else:
            entry = self.models.get(chosen)
            if entry is None:
                raise UnknownModel(chosen)
            if not entry.available or entry.model is None:
                raise EngineUnavailable(entry.error or f"{chosen} unavailable")
            start = time.perf_counter()
            matrix, rate = self._forward(entry, np.asarray(config, dtype=np.float32))
            latency_ms = (time.perf_counter() - start) * 1000.0
            result = {
                "model": chosen,
                "kind": entry.kind,
                "rate": float(rate),
                "latency_ms": latency_ms,
                "M_shape": list(M_SHAPE),
                "M_norm": float(np.linalg.norm(matrix)),
                "M_hash": _hash_matrix(matrix),
                "matrix": matrix,
                "_nearest": None,
            }
        if decision is not None:
            result["decision"] = decision
        if include_m:
            result["M_real"] = _round_nested(matrix.real)
            result["M_imag"] = _round_nested(matrix.imag)
        result.pop("matrix", None)
        result.pop("_nearest", None)
        return result

    def _forward(self, entry: ModelEntry, config_raw: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self._torch
        x = (np.asarray(config_raw, dtype=np.float32) - self._scaler_mean) / self._scaler_scale
        with torch.no_grad():
            out = entry.model(torch.from_numpy(x).float().unsqueeze(0))[0].cpu().numpy()
        matrix = (out[:E] + 1j * out[E : 2 * E]).reshape(M_SHAPE)
        rate = float(out[2 * E] * self.rate_std + self.rate_mean)
        return matrix, rate

    # ------------------------------------------------------------------- GP
    def _gp_truth(self, config: list[float]) -> tuple[np.ndarray, float, float]:
        fixture = self._fixture
        configs = np.asarray(fixture["config_raw"], dtype=np.float32)
        target = np.asarray(config, dtype=np.float32)
        distances = np.max(np.abs(configs - target), axis=1)
        index = int(np.argmin(distances))
        nearest = float(distances[index])
        if nearest > 1e-2:
            raise GpTruthUnavailable(
                "GP ground truth is only available for published/static samples"
            )
        matrix = np.asarray(fixture["gp_M"][index], dtype=np.complex64)
        rate = float(np.asarray(fixture["gp_rate"])[index])
        return matrix, rate, nearest

    # ------------------------------------------------------------------ JEV
    def jev_decide(self, config: list[float], probe: list[float] | None = None) -> dict[str, Any]:
        signature = self._probe_signature(config, probe)
        try:
            gp_matrix, gp_rate, _ = self._gp_truth(config)
        except GpTruthUnavailable:
            gp_matrix, gp_rate = None, None
        profiles: dict[str, ModelProfile] = {}
        for name, entry in self.models.items():
            if entry.kind != "dl" or not entry.available:
                continue
            try:
                matrix, rate = self._forward(entry, np.asarray(config, dtype=np.float32))
            except Exception:  # a broken candidate must not block the decision  # noqa: BLE001, S112
                continue
            centroid = signature_from_gram(_gram(matrix), k=self.settings.jev_k)
            error = 0.0
            if gp_matrix is not None and gp_rate is not None:
                gp_norm = float(np.linalg.norm(gp_matrix))
                pred_norm = float(np.linalg.norm(matrix))
                cos = float(np.vdot(gp_matrix, matrix).real / (gp_norm * pred_norm + 1e-12))
                rate_error = abs(rate - gp_rate) / max(self.rate_std, 1e-6)
                error = (1.0 - cos) + 0.05 * rate_error
            profiles[name] = ModelProfile(
                name=name,
                centroid=tuple(centroid),
                latency_ms=entry.latency_ms,
                kind="dl",
                error=error,
            )
        if not profiles:
            if self.models["gp"].available:
                return {
                    "chosen_model": "gp",
                    "confidence": 1.0,
                    "scores": {"gp": 0.0},
                    "probe": signature,
                    "latency_weight": self.settings.latency_weight,
                    "spectral_weight": self.settings.spectral_weight,
                }
            raise EngineUnavailable("no models available for a JEV decision")
        return decide(
            signature,
            profiles,
            latency_weight=self.settings.latency_weight,
            spectral_weight=self.settings.spectral_weight,
        )

    def _probe_signature(self, config: list[float], probe: list[float] | None) -> list[float]:
        k = self.settings.jev_k
        if probe:
            values = [float(value) for value in probe]
            if len(values) == k and abs(sum(values) - 1.0) < 1e-3 and min(values) >= 0:
                return values
            return spectral_signature(values, k=k)
        try:
            matrix, _, _ = self._gp_truth(config)
            return signature_from_gram(_gram(matrix), k=k)
        except GpTruthUnavailable:
            entry = self.models["hybrid_jev"]
            if not entry.available:
                return [1.0 / k] * k
            matrix, _ = self._forward(entry, np.asarray(config, dtype=np.float32))
            return signature_from_gram(_gram(matrix), k=k)

    # -------------------------------------------------------------- registry
    def switch(self, model: str) -> dict[str, Any]:
        entry = self.models.get(model)
        if entry is None:
            raise UnknownModel(model)
        if not entry.available:
            raise EngineUnavailable(entry.error or f"{model} unavailable")
        self.active_model = model
        return {"active_model": model, "available": entry.available}

    def registry(self) -> list[dict[str, Any]]:
        return [entry.summary() for entry in self.models.values()]

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "load_error": self.load_error,
            "active_model": self.active_model,
            "rate_mean": self.rate_mean,
            "rate_std": self.rate_std,
            "jev_k": self.settings.jev_k,
            "models": [entry.summary() for entry in self.models.values()],
        }


def _gram(matrix: np.ndarray) -> np.ndarray:
    flat = np.asarray(matrix, dtype=np.complex128).reshape(-1, M_SHAPE[-1])
    return flat.conj().T @ flat


def _hash_matrix(matrix: np.ndarray) -> str:
    rounded = np.round(np.asarray(matrix), 6)
    payload = rounded.real.astype(np.float32).tobytes() + rounded.imag.astype(np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _round_nested(array: np.ndarray, digits: int = 6) -> list[Any]:
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def _read_json_metrics(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("test"), dict):
        return data["test"]
    test_metrics = {
        key.removeprefix("test_"): value
        for key, value in data.items()
        if key.startswith("test_")
    }
    return test_metrics or data
