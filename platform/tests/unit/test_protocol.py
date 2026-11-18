"""Framing, ACK and JEV protocol unit tests."""

from __future__ import annotations

import zlib

import pytest
from protocol.ack import AckMachine, AckState
from protocol.frames import (
    Frame,
    FrameCorrupt,
    FrameIncomplete,
    FrameType,
    decode_frame,
    encode_frame,
    iter_frames,
    json_frame,
    payload_json,
)
from protocol.jev import ModelProfile, decide, l1_distance, spectral_signature


def test_crc_reference_vector() -> None:
    assert zlib.crc32(b"123456789") & 0xFFFFFFFF == 0xCBF43926


def test_frame_roundtrip() -> None:
    encoded = encode_frame(FrameType.DATA_REQ, 42, b'{"sample_id":"s00001"}')
    frame, consumed = decode_frame(encoded)
    assert consumed == len(encoded)
    assert frame.type == FrameType.DATA_REQ
    assert frame.seq == 42
    assert payload_json(frame) == {"sample_id": "s00001"}


def test_frame_incomplete_and_corrupt() -> None:
    encoded = encode_frame(FrameType.DATA_ACK, 7, b"abc")
    with pytest.raises(FrameIncomplete):
        decode_frame(encoded[:-1])
    corrupted = bytearray(encoded)
    corrupted[8] ^= 0xFF
    with pytest.raises(FrameCorrupt):
        decode_frame(bytes(corrupted))
    with pytest.raises(FrameCorrupt):
        decode_frame(b"XXXX" + encoded[4:])


def test_iter_frames_ignores_partial_tail() -> None:
    stream = encode_frame(FrameType.DATA_REQ, 1, b"a") + encode_frame(
        FrameType.DATA_ACK, 2, b"b"
    )
    assert [frame.seq for frame in iter_frames(stream)] == [1, 2]
    assert [frame.seq for frame in iter_frames(stream + b"partial")] == [1, 2]


def test_json_frame_helper() -> None:
    encoded = json_frame(FrameType.RECV_OK, 3, {"chosen_model": "hybrid_jev"})
    frame, _ = decode_frame(encoded)
    assert payload_json(frame)["chosen_model"] == "hybrid_jev"


def test_ack_handshake() -> None:
    machine = AckMachine()
    assert machine.data_request(9) == AckState.AWAIT_RECV_OK
    encoded = json_frame(FrameType.RECV_OK, 9, {"chosen_model": "gp"})
    recv, _ = decode_frame(encoded)
    assert machine.on_frame(recv) == AckState.AWAIT_DATA_ACK
    assert machine.chosen_model == "gp"
    assert machine.on_frame(Frame(FrameType.DATA_ACK, 9)) == AckState.DELIVERED
    assert machine.complete


def test_ack_ignores_stale_sequence() -> None:
    machine = AckMachine()
    machine.data_request(4)
    assert machine.on_frame(Frame(FrameType.RECV_OK, 5)) == AckState.AWAIT_RECV_OK


def test_ack_retry_limit() -> None:
    machine = AckMachine(max_attempts=2)
    machine.data_request(1)
    assert machine.retry() is True
    assert machine.retry() is False
    assert machine.state == AckState.FAILED


def test_spectral_signature_sums_to_one() -> None:
    signature = spectral_signature([5.0, 3.0, 1.0, 0.5, 0.1], k=4)
    assert len(signature) == 4
    assert pytest.approx(sum(signature), rel=1e-9) == 1.0
    assert signature[0] > signature[-1]


def test_jev_decision_prefers_matching_centroid() -> None:
    profiles = {
        "gp": ModelProfile("gp", (0.7, 0.1, 0.1, 0.1), latency_ms=0.0, kind="gp"),
        "dl": ModelProfile("dl", (0.1, 0.7, 0.1, 0.1), latency_ms=5.0, kind="dl"),
    }
    decision = decide((0.1, 0.7, 0.1, 0.1), profiles)
    assert decision["chosen_model"] == "dl"
    assert 0.0 <= decision["confidence"] <= 1.0
    assert l1_distance((1, 0, 0, 0), (0, 1, 0, 0)) == 2.0
