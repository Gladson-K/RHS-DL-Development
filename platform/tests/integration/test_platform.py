"""End-to-end tests against a running docker compose stack.

Run the stack first::

    cd platform && docker compose up -d --build

Then::

    PLATFORM_URL=http://localhost:8088 pytest platform/tests/integration -v
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from protocol.frames import FrameType, decode_frame, json_frame, payload_json

PLATFORM_URL = os.getenv("PLATFORM_URL", "http://localhost:8088").rstrip("/")
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/static_samples.npz"


def _platform_up() -> bool:
    try:
        response = httpx.get(f"{PLATFORM_URL}/", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _platform_up(), reason=f"platform not running at {PLATFORM_URL}"
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    data = np.load(FIXTURE, allow_pickle=True)
    matrix = np.asarray(data["gp_M"][0])
    return {
        "sample_id": str(data["sample_ids"][0]),
        "config": np.asarray(data["config_raw"][0], dtype=np.float32).tolist(),
        "gp_rate": float(np.asarray(data["gp_rate"])[0]),
        "gp_M": {
            "real": np.round(matrix.real, 6).tolist(),
            "imag": np.round(matrix.imag, 6).tolist(),
        },
    }


def test_pages_render(fixture: dict) -> None:
    for path in ("/", "/results", "/models"):
        response = httpx.get(f"{PLATFORM_URL}{path}", timeout=30.0)
        assert response.status_code == 200, path
        assert "RHS" in response.text


def test_publish_and_validation(fixture: dict) -> None:
    payload = {
        **fixture,
        "transport": "simulator",
        "device_id": "pytest",
    }
    response = httpx.post(f"{PLATFORM_URL}/api/samples", json=payload, timeout=120.0)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["cid"]
    assert "diff" in data
    assert data["diff"]["rate_abs_delta"] >= 0.0
    assert data["diff"]["m_cos"] > 0.0
    assert data["urls"]["validation"].endswith(data["cid"])
    cid = data["cid"]
    for path in (
        f"/validation/{cid}",
        f"/samples/{cid}",
        f"/streams/{cid}",
        f"/api/validation/{cid}",
        f"/api/samples/{cid}",
    ):
        page = httpx.get(f"{PLATFORM_URL}{path}", timeout=30.0)
        assert page.status_code == 200, path
    report = httpx.get(f"{PLATFORM_URL}/api/validation/{cid}", timeout=30.0).json()
    assert "checks" in report["validation"]
    assert "source" in report and "travel" in report and "destination" in report


def test_ack_jev_handshake(fixture: dict) -> None:
    cid = f"it-{int(time.time() * 1000)}"
    seq = 11
    request = json_frame(
        FrameType.DATA_REQ,
        seq,
        {
            "sample_id": fixture["sample_id"],
            "transport": "simulator",
            "device_id": "pytest",
        },
    )
    response = httpx.post(
        f"{PLATFORM_URL}/api/streams/{cid}/frames",
        json={"frame": base64.b64encode(request).decode("ascii")},
        timeout=120.0,
    )
    assert response.status_code == 200, response.text
    recv, _ = decode_frame(base64.b64decode(response.json()["frame"]))
    assert recv.type == FrameType.RECV_OK
    decision = payload_json(recv)
    assert decision["chosen_model"]
    assert 0.0 <= float(decision["confidence"]) <= 1.0

    ack = json_frame(FrameType.DATA_ACK, seq, {"status": "receive_ok"})
    response = httpx.post(
        f"{PLATFORM_URL}/api/streams/{cid}/frames",
        json={"frame": base64.b64encode(ack).decode("ascii")},
        timeout=120.0,
    )
    assert response.status_code == 200, response.text
    result, _ = decode_frame(base64.b64decode(response.json()["frame"]))
    assert result.type == FrameType.RESULT
    payload = payload_json(result)
    assert payload["chosen_model"] == decision["chosen_model"]
    assert "diff" in payload and "urls" in payload
    check = httpx.get(f"{PLATFORM_URL}/api/validation/{cid}", timeout=30.0)
    assert check.status_code == 200


def test_switch_endpoint() -> None:
    response = httpx.post(
        f"{PLATFORM_URL}/api/switch", json={"model": "hybrid_jev"}, timeout=30.0
    )
    assert response.status_code == 200, response.text
    assert response.json()["active_model"] == "hybrid_jev"


def test_published_artifacts_are_browsable() -> None:
    listing = httpx.get(f"{PLATFORM_URL}/results", timeout=30.0)
    assert listing.status_code == 200
    if "mimo64_nas" not in listing.text:
        pytest.skip("mimo64_nas artifacts are not mounted")
    run_page = httpx.get(f"{PLATFORM_URL}/results/mimo64_nas", timeout=30.0)
    assert run_page.status_code == 200
    figure = httpx.get(
        f"{PLATFORM_URL}/results/mimo64_nas/plots/pareto_frontier.png", timeout=30.0
    )
    assert figure.status_code == 200
    assert figure.headers["content-type"].startswith("image/")
