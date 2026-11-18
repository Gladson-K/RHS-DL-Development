#!/usr/bin/env python3
"""Edge gateway for the RHS streaming platform.

The gateway is the host-side half of the STM32 intermediate layer.  It can:

* replay the static fixture through the ACK/JEV handshake (``--mode simulator``),
* bridge a USB CDC/FTDI serial port (``--mode serial``),
* bridge an ESP32/ESP-01 Wi-Fi link over MQTT (``--mode wifi``),
* ingest an HDMI capture device (``--mode hdmi``).

Only ``httpx`` and ``numpy`` are required for the simulator; the hardware modes
import their optional dependencies lazily so the platform can run without a
board attached.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

from protocol.frames import Frame, FrameType, decode_frame, json_frame, payload_json  # noqa: E402, RUF100


class GatewayClient:
    """Thin HTTP client used by every transport."""

    def __init__(self, backend: str, timeout: float = 60.0) -> None:
        self.backend = backend.rstrip("/")
        self.client = httpx.Client(base_url=self.backend, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def health(self) -> dict[str, Any]:
        response = self.client.get("/healthz")
        return response.json()

    def publish(self, sample: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post("/api/samples", json=sample)
        response.raise_for_status()
        return response.json()

    def send_frame(self, cid: str, frame_bytes: bytes) -> Frame:
        encoded = base64.b64encode(frame_bytes).decode("ascii")
        response = self.client.post(f"/api/streams/{cid}/frames", json={"frame": encoded})
        response.raise_for_status()
        data = response.json()
        frame, _ = decode_frame(base64.b64decode(data["frame"]))
        return frame

    def validation(self, cid: str) -> dict[str, Any]:
        response = self.client.get(f"/api/validation/{cid}")
        response.raise_for_status()
        return response.json()


def load_fixture(path: Path) -> dict[str, Any]:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    return {
        "sample_ids": [str(value) for value in data["sample_ids"]],
        "config": np.asarray(data["config_raw"], dtype=np.float32).tolist(),
        "gp_rate": np.asarray(data["gp_rate"], dtype=np.float32).tolist(),
    }


def run_simulator(args: argparse.Namespace) -> int:
    """Replay the static fixture, one ACK/JEV handshake per sample."""
    fixture = load_fixture(Path(args.fixture))
    client = GatewayClient(args.backend, args.timeout)
    print(f"[simulator] backend={args.backend} samples={len(fixture['sample_ids'])}")
    print(f"[simulator] health={json.dumps(client.health())}")
    seq = 0
    for iteration in range(args.iterations):
        for index, sample_id in enumerate(fixture["sample_ids"]):
            seq += 1
            cid = f"gw{seq:04d}-{sample_id}"
            probe = None
            request = json_frame(
                FrameType.DATA_REQ,
                seq,
                {
                    "sample_id": sample_id,
                    "transport": "simulator",
                    "device_id": "gateway-sim",
                    "source_ts": time.time(),
                    "probe": probe,
                },
            )
            recv = client.send_frame(cid, request)
            decision = payload_json(recv) or {}
            print(
                f"[{cid}] RECV_OK model={decision.get('chosen_model')} "
                f"confidence={decision.get('confidence')}"
            )
            ack = json_frame(
                FrameType.DATA_ACK,
                seq,
                {"cid": cid, "status": "receive_ok", "chosen_model": decision.get("chosen_model")},
            )
            result = client.send_frame(cid, ack)
            payload = payload_json(result) or {}
            diff = payload.get("diff", {})
            print(
                f"[{cid}] RESULT gp={payload.get('gp_rate')} dl={payload.get('dl_rate')} "
                f"delta={diff.get('rate_delta')} m_cos={diff.get('m_cos')}"
            )
            print(f"[{cid}] urls={payload.get('urls')}")
            if args.verbose:
                print(json.dumps(client.validation(cid), indent=2, default=str))
    client.close()
    return 0


def _find_serial_port(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/tty.usbmodem*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def run_serial(args: argparse.Namespace) -> int:
    """Bridge a USB serial port carrying the same frames."""
    try:
        import serial  # type: ignore
    except ImportError:
        print("pyserial is required for --mode serial (pip install pyserial)")
        return 2
    port = _find_serial_port(args.port)
    if not port:
        print("no USB serial device found")
        return 2
    client = GatewayClient(args.backend, args.timeout)
    cid = f"usb-{int(time.time())}"
    print(f"[serial] port={port} cid={cid}")
    with serial.Serial(port, args.baud, timeout=0.5) as link:
        deadline = time.time() + args.duration
        buffer = b""
        while time.time() < deadline:
            chunk = link.read(4096)
            if not chunk:
                continue
            buffer += chunk
            while buffer:
                try:
                    frame, consumed = decode_frame(buffer)
                except Exception:  # incomplete or corrupt  # noqa: BLE001
                    break
                buffer = buffer[consumed:]
                if frame.type == FrameType.DATA_REQ:  # noqa: SIM114
                    reply = client.send_frame(cid, frame.encode())
                    link.write(reply.encode())
                elif frame.type == FrameType.DATA_ACK:
                    reply = client.send_frame(cid, frame.encode())
                    link.write(reply.encode())
    client.close()
    return 0


def run_wifi(args: argparse.Namespace) -> int:
    """Bridge an ESP32 MQTT link to the backend frame endpoint."""
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except ImportError:
        print("paho-mqtt is required for --mode wifi (pip install paho-mqtt)")
        return 2
    client = GatewayClient(args.backend, args.timeout)
    cid = f"wifi-{int(time.time())}"
    broker = args.broker
    print(f"[wifi] broker={broker} cid={cid}")

    def on_message(_client, _userdata, message):  # type: ignore[no-untyped-def]
        try:
            frame, _ = decode_frame(message.payload)
        except Exception:  # noqa: BLE001
            return
        reply = client.send_frame(cid, frame.encode())
        mqtt_client.publish(args.topic_out, reply.encode())

    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message
    mqtt_client.connect(args.broker_host, args.broker_port, 60)
    mqtt_client.subscribe(args.topic_in)
    mqtt_client.loop_forever()
    client.close()
    return 0


def run_hdmi(args: argparse.Namespace) -> int:
    """Ingest an HDMI capture device and annotate the retro display stream."""
    try:
        import cv2  # type: ignore
    except ImportError:
        print("opencv-python-headless is required for --mode hdmi")
        return 2
    device = args.video_device
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        print(f"cannot open HDMI capture device {device}")
        return 2
    print(f"[hdmi] capturing from {device}; press Ctrl-C to stop")
    deadline = time.time() + args.duration
    try:
        while time.time() < deadline:
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            print(f"[hdmi] frame {width}x{height} at {time.time():.3f}")
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["simulator", "serial", "wifi", "hdmi", "auto"],
        default="simulator",
    )
    parser.add_argument("--backend", default="http://localhost:8088")
    parser.add_argument("--fixture", default=str(PLATFORM_ROOT / "fixtures/static_samples.npz"))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--broker", default="mqtt://localhost:1883")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic-in", default="rhs/rx")
    parser.add_argument("--topic-out", default="rhs/tx")
    parser.add_argument("--video-device", default="/dev/video0")
    args = parser.parse_args(argv)

    mode = args.mode
    if mode == "auto":
        if _find_serial_port(args.port):
            mode = "serial"
        elif Path(args.video_device).exists():
            mode = "hdmi"
        else:
            mode = "simulator"
    if mode == "simulator":
        return run_simulator(args)
    if mode == "serial":
        return run_serial(args)
    if mode == "wifi":
        return run_wifi(args)
    return run_hdmi(args)


if __name__ == "__main__":
    raise SystemExit(main())
