"""Request/response models shared by the backend and inference services."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Transport = Literal["simulator", "usb", "wifi", "hdmi", "static"]


def _check_config(values: list[float]) -> list[float]:
    if len(values) != 10:
        raise ValueError("config must contain exactly 10 features")
    return [float(value) for value in values]


class SampleIn(BaseModel):
    """A published sample: configuration plus optional GP ground truth."""

    sample_id: str = Field(min_length=1, max_length=128)
    config: list[float]
    gp_rate: float | None = None
    gp_M: Any | None = None
    transport: Transport = "simulator"
    device_id: str = "sim-01"
    source_ts: float | None = None

    _config = field_validator("config")(_check_config)


class PredictRequest(BaseModel):
    """Inference request. ``model='auto'`` invokes the JEV decision first."""

    config: list[float]
    model: str = "auto"
    include_M: bool = True
    probe: list[float] | None = None
    include_decision: bool = True

    _config = field_validator("config")(_check_config)


class SwitchRequest(BaseModel):
    """Force the platform's active model."""

    model: str = Field(min_length=1, max_length=64)


class FrameIn(BaseModel):
    """Base64 encoded protocol frame."""

    frame: str = Field(min_length=1)


class FrameOut(BaseModel):
    frame: str
    seq: int
    type: str


class GpTruth(BaseModel):
    rate: float
    real: list[Any]
    imag: list[Any]


class DlOutput(BaseModel):
    model: str
    rate: float
    real: list[Any] | None = None
    imag: list[Any] | None = None
    latency_ms: float | None = None


class DiffReport(BaseModel):
    rate_delta: float
    rate_abs_delta: float
    m_cos: float
    m_nmse: float


class PublishResponse(BaseModel):
    cid: str
    sample_id: str
    decision: dict[str, Any]
    gp: dict[str, Any]
    dl: dict[str, Any]
    diff: DiffReport
    urls: dict[str, str]


def as_real_imag(value: Any) -> tuple[list[Any], list[Any]] | None:
    """Normalise a complex matrix into ``(real, imag)`` nested lists.

    Accepts either ``{"real": ..., "imag": ...}``, ``{"real_flat": ...,
    "imag_flat": ...}`` or a list of ``[real, imag]`` pairs.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if "real" in value and "imag" in value:
            return value["real"], value["imag"]
        if "real_flat" in value and "imag_flat" in value:
            return value["real_flat"], value["imag_flat"]
        raise ValueError("complex matrix dict needs real/imag keys")
    if isinstance(value, list):
        if not value:
            return [], []
        first = value[0]
        is_pair = isinstance(first, (list, tuple)) and len(first) == 2
        if is_pair and isinstance(first[0], (int, float)):
            real = [float(pair[0]) for pair in value]
            imag = [float(pair[1]) for pair in value]
            return real, imag
        if isinstance(first, dict) and "real" in first and "imag" in first:
            return [item["real"] for item in value], [item["imag"] for item in value]
    raise ValueError("unsupported complex matrix encoding")
