"""Binary framing for the RHS streaming platform.

Wire format (big endian)::

    +--------+---------+-------+---------+---------+---------+--------+
    | magic  | version | type  | seq     | length  | payload | crc32  |
    | 4 B    | 1 B     | 1 B   | 4 B     | 4 B     | N B     | 4 B    |
    +--------+---------+-------+---------+---------+---------+--------+

The CRC32 covers every byte before the trailing CRC field and uses the same
polynomial as :func:`zlib.crc32` (IEEE 802.3), so the STM32 firmware and the
Python services agree bit-for-bit.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

MAGIC = b"RHS1"
VERSION = 1
MAX_PAYLOAD = 1 << 20  # 1 MiB

# magic(4s), version(B), type(B), seq(I), length(I)
HEADER = struct.Struct(">4sBBII")
CRC = struct.Struct(">I")
MIN_FRAME = HEADER.size + CRC.size


class FrameType(IntEnum):
    """Message kinds exchanged during the ACK/JEV handshake."""

    DATA_REQ = 1
    RECV_OK = 2
    DATA_ACK = 3
    JEV_DECISION = 4
    MODEL_SWITCH = 5
    RESULT = 6
    ERROR = 7


class FrameError(ValueError):
    """Base class for framing errors."""


class FrameIncomplete(FrameError):
    """Not enough bytes have arrived yet; call again after more data."""


class FrameCorrupt(FrameError):
    """The frame is malformed or its CRC does not match."""


@dataclass(frozen=True)
class Frame:
    """A decoded protocol frame."""

    type: FrameType
    seq: int
    payload: bytes = b""
    version: int = VERSION

    def encode(self) -> bytes:
        """Serialise the frame including header, payload and CRC."""
        return encode_frame(self.type, self.seq, self.payload, self.version)


def encode_frame(
    frame_type: FrameType | int,
    seq: int,
    payload: bytes = b"",
    version: int = VERSION,
) -> bytes:
    """Encode one frame.

    Args:
        frame_type: Message kind (``FrameType`` or raw integer).
        seq: 32-bit sequence number used for idempotency and ACK matching.
        payload: Raw payload bytes (JSON is used by the Python services).
        version: Protocol version byte.

    Raises:
        FrameCorrupt: If a field is out of range.
    """
    try:
        ftype = FrameType(int(frame_type))
    except (TypeError, ValueError) as exc:
        raise FrameCorrupt(f"unknown frame type: {frame_type!r}") from exc
    if not 0 <= seq <= 0xFFFFFFFF:
        raise FrameCorrupt(f"sequence out of range: {seq}")
    if not 0 <= version <= 0xFF:
        raise FrameCorrupt(f"version out of range: {version}")
    if len(payload) > MAX_PAYLOAD:
        raise FrameCorrupt(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    header = HEADER.pack(MAGIC, version, int(ftype), seq, len(payload))
    body = header + bytes(payload)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + CRC.pack(crc)


def decode_frame(data: bytes | bytearray | memoryview, offset: int = 0) -> tuple[Frame, int]:
    """Decode one frame from ``data`` starting at ``offset``.

    Returns:
        ``(frame, next_offset)`` so callers can iterate a byte stream.

    Raises:
        FrameIncomplete: When fewer than a full frame are available.
        FrameCorrupt: When the magic, version, length or CRC is invalid.
    """
    buf = bytes(data)
    if offset < 0:
        raise FrameCorrupt("negative offset")
    if len(buf) - offset < HEADER.size:
        raise FrameIncomplete("header incomplete")
    magic, version, raw_type, seq, length = HEADER.unpack_from(buf, offset)
    if magic != MAGIC:
        raise FrameCorrupt(f"bad magic: {magic!r}")
    if version != VERSION:
        raise FrameCorrupt(f"unsupported version: {version}")
    if length > MAX_PAYLOAD:
        raise FrameCorrupt(f"payload too large: {length}")
    end = offset + HEADER.size + length
    if len(buf) < end + CRC.size:
        raise FrameIncomplete("payload or crc incomplete")
    body = buf[offset:end]
    (expected,) = CRC.unpack_from(buf, end)
    actual = zlib.crc32(body) & 0xFFFFFFFF
    if actual != expected:
        raise FrameCorrupt(f"crc mismatch: got {actual:#010x} expected {expected:#010x}")
    try:
        ftype = FrameType(raw_type)
    except ValueError as exc:
        raise FrameCorrupt(f"unknown frame type: {raw_type}") from exc
    payload = buf[offset + HEADER.size : end]
    return Frame(type=ftype, seq=seq, payload=payload, version=version), end + CRC.size


def iter_frames(data: bytes | bytearray | memoryview) -> list[Frame]:
    """Decode every complete frame in ``data`` and ignore a trailing partial frame."""
    frames: list[Frame] = []
    offset = 0
    while offset < len(data):
        try:
            frame, offset = decode_frame(data, offset)
        except FrameIncomplete:
            break
        frames.append(frame)
    return frames


def json_frame(frame_type: FrameType | int, seq: int, obj: Any) -> bytes:
    """Encode a frame whose payload is a compact JSON document."""
    payload = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return encode_frame(frame_type, seq, payload)


def payload_json(frame: Frame) -> Any:
    """Decode a frame payload as JSON (empty payload returns ``None``)."""
    if not frame.payload:
        return None
    try:
        return json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameCorrupt("payload is not valid JSON") from exc
