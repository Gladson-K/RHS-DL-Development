"""Source -> travel -> destination provenance and validation checks."""

from __future__ import annotations

import math
from typing import Any

from .metrics import matrix_hash


def source_hash(config: list[float], gp_real: Any, gp_imag: Any) -> str:
    """Hash of the configuration + GP beamformer captured at the source."""
    config_part = "|".join(f"{float(value):.6f}" for value in config)
    return f"{config_part}#{matrix_hash(gp_real, gp_imag)}"


def source_hop(
    cid: str,
    sample_id: str,
    config: list[float],
    gp_real: Any,
    gp_imag: Any,
    transport: str,
    device_id: str,
) -> dict[str, Any]:
    return {
        "cid": cid,
        "stage": "source",
        "sample_id": sample_id,
        "transport": transport,
        "device_id": device_id,
        "config": list(config),
        "source_hash": source_hash(config, gp_real, gp_imag),
    }


def travel_hop(
    cid: str,
    transport: str,
    frames: int,
    crc_ok: bool,
    latency_ms: float,
    retries: int,
) -> dict[str, Any]:
    return {
        "cid": cid,
        "stage": "travel",
        "transport": transport,
        "frames": int(frames),
        "crc_ok": bool(crc_ok),
        "latency_ms": round(float(latency_ms), 3),
        "retries": int(retries),
    }


def destination_hop(cid: str, dl: dict[str, Any], diff: dict[str, Any]) -> dict[str, Any]:
    return {
        "cid": cid,
        "stage": "destination",
        "dl_model": dl.get("model"),
        "dl_hash": diff.get("dl_hash"),
        "gp_hash": diff.get("gp_hash"),
        "rate_delta": diff.get("rate_delta"),
        "m_cos": diff.get("m_cos"),
        "m_nmse": diff.get("m_nmse"),
    }


def _config_gp_hash(config: list[float], gp_hash: Any) -> str:
    config_part = "|".join(f"{float(value):.6f}" for value in config)
    return f"{config_part}#{gp_hash}"


def validate(
    source: dict[str, Any],
    travel: dict[str, Any],
    destination: dict[str, Any],
    diff: dict[str, Any],
    decision: dict[str, Any] | None,
    ack_delivered: bool,
) -> dict[str, Any]:
    """Compute the pass/fail checks used by ``/validation/{cid}``."""
    destination_hash = _config_gp_hash(source.get("config", []), destination.get("gp_hash"))
    source_match = source.get("source_hash") == destination_hash
    rate = float(diff.get("gp_rate", 0.0))
    delta = float(diff.get("rate_delta", 0.0))
    checks = {
        "source_destination_hash_match": bool(source_match),
        "crc_ok": bool(travel.get("crc_ok", True)),
        "frames_delivered": int(travel.get("frames", 0)) >= 1,
        "ack_delivered": bool(ack_delivered),
        "dl_rate_finite": math.isfinite(float(diff.get("dl_rate", float("nan")))),
        "rate_delta_within_20pct": abs(delta) <= max(0.2 * abs(rate), 1e-6),
        "m_cos_above_0_5": float(diff.get("m_cos", 0.0)) >= 0.5,
        "m_nmse_below_1": float(diff.get("m_nmse", 1.0)) <= 1.0,
        "decision_matches_output": (
            decision is None
            or decision.get("chosen_model") == destination.get("dl_model")
        ),
    }
    passed = all(bool(value) for value in checks.values())
    score = sum(1 for value in checks.values() if value) / max(len(checks), 1)
    return {
        "checks": checks,
        "passed": passed,
        "score": round(score, 4),
        "source_hash": source.get("source_hash"),
        "destination_hash": destination_hash,
    }
