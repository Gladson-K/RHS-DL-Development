"""ACK state machine shared by the gateway simulator and the backend.

The handshake is::

    device  ── DATA_REQ ─────────────────▶ platform
    device  ◀───────────── RECV_OK ────── platform   (model + JEV decision)
    device  ── DATA_ACK ────────────────▶ platform   ("receive ok")
    device  ◀───────────── RESULT ─────── platform   (beamforming vector + diff)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .frames import Frame, FrameType, payload_json


class AckState(str, Enum):
    """Lifecycle of a single sequenced exchange."""

    IDLE = "idle"
    AWAIT_RECV_OK = "await_recv_ok"
    AWAIT_DATA_ACK = "await_data_ack"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class AckMachine:
    """Deterministic ACK tracker for one in-flight sequence number."""

    max_attempts: int = 3
    seq: int | None = None
    state: AckState = AckState.IDLE
    attempts: int = 0
    chosen_model: str | None = None
    confidence: float | None = None
    last_error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def data_request(self, seq: int) -> AckState:
        """Start a new exchange and wait for ``RECV_OK``."""
        self.seq = int(seq)
        self.state = AckState.AWAIT_RECV_OK
        self.attempts = 0
        self.chosen_model = None
        self.confidence = None
        self.last_error = None
        self._record("DATA_REQ")
        return self.state

    def on_frame(self, frame: Frame) -> AckState:
        """Advance the state machine with an inbound frame."""
        if frame.type == FrameType.ERROR:
            self.last_error = _error_text(frame)
            self.state = AckState.FAILED
            self._record("ERROR")
            return self.state
        if self.seq is not None and frame.seq != self.seq:
            # Stale or unrelated frame: record but do not advance.
            self._record("IGNORED")
            return self.state
        if frame.type == FrameType.RECV_OK and self.state == AckState.AWAIT_RECV_OK:
            data = payload_json(frame) or {}
            self.chosen_model = data.get("chosen_model") or data.get("model")
            confidence = data.get("confidence")
            self.confidence = float(confidence) if confidence is not None else None
            self.state = AckState.AWAIT_DATA_ACK
        elif frame.type == FrameType.MODEL_SWITCH:
            data = payload_json(frame) or {}
            self.chosen_model = data.get("chosen_model") or data.get("model") or self.chosen_model
        elif frame.type == FrameType.DATA_ACK and self.state == AckState.AWAIT_DATA_ACK:
            self.state = AckState.DELIVERED
        self._record(frame.type.name)
        return self.state

    def retry(self) -> bool:
        """Return ``True`` when the exchange should be retransmitted."""
        if self.state in (AckState.AWAIT_RECV_OK, AckState.AWAIT_DATA_ACK):
            if self.attempts + 1 >= self.max_attempts:
                self.attempts += 1
                self.state = AckState.FAILED
                self.last_error = "ack timeout"
                return False
            self.attempts += 1
            return True
        return False

    @property
    def complete(self) -> bool:
        return self.state == AckState.DELIVERED

    def _record(self, event: str) -> None:
        self.history.append({"event": event, "state": self.state.value, "seq": self.seq})

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "state": self.state.value,
            "attempts": self.attempts,
            "chosen_model": self.chosen_model,
            "confidence": self.confidence,
            "last_error": self.last_error,
            "history": self.history,
        }


def _error_text(frame: Frame) -> str:
    data = payload_json(frame)
    if isinstance(data, dict):
        return str(data.get("error") or data.get("detail") or data)
    return "remote error"
