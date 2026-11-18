"""Platform service: publication, ACK/JEV handshake and validation reports."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

import httpx
from fastapi import HTTPException
from protocol.ack import AckMachine, AckState
from protocol.frames import (
    Frame,
    FrameCorrupt,
    FrameType,
    decode_frame,
    payload_json,
)
from protocol.schemas import SampleIn, as_real_imag

from . import metrics, provenance, registry
from .fixtures import StaticSamples


class PlatformService:
    """Coordinates MongoDB, Redis, the inference service and the artifact index."""

    def __init__(self, settings, store, samples: StaticSamples) -> None:
        self.settings = settings
        self.store = store
        self.samples = samples
        self.artifacts: dict[str, registry.Artifact] = {}
        self.runs: dict[str, list[registry.Artifact]] = {}
        self.client = httpx.AsyncClient(
            base_url=settings.inference_url, timeout=settings.inference_timeout
        )
        self._pending: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------- lifecycle
    async def close(self) -> None:
        await self.client.aclose()

    def refresh_artifacts(self) -> None:
        items = registry.scan(self.settings.artifact_dirs)
        self.artifacts = {item.key: item for item in items}
        self.runs = registry.by_run(items)
        for item in items:
            try:
                self.store.upsert("artifacts", "_id", item.key, item.to_dict())
            except Exception:  # noqa: BLE001, S110
                pass

    def get_artifact(self, run: str, name: str) -> registry.Artifact | None:
        return self.artifacts.get(f"{run}/{name}")

    # ------------------------------------------------------------- inference
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"inference unavailable: {exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"inference {path} failed ({response.status_code}): {response.text[:300]}",
            )
        return response.json()

    async def inference_models(self) -> dict[str, Any]:
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            return {"models": [], "active_model": None, "error": str(exc)}

    async def switch_model(self, model: str) -> dict[str, Any]:
        result = await self._post("/switch", {"model": model})
        active = {"model": result.get("active_model", model), "ts": time.time()}
        self.store.set_json("model:active", active)
        return result

    async def inference_health(self) -> dict[str, Any]:
        try:
            response = await self.client.get("/healthz")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": str(exc)}

    # ------------------------------------------------------------------- cid
    def new_cid(self) -> str:
        value = self.store.incr("cid:counter")
        if value:
            return f"c{value:06d}"
        return f"c{uuid.uuid4().hex[:10]}"

    # ------------------------------------------------------------- GP truth
    async def _gp_truth(self, sample: SampleIn):
        real: Any = None
        imag: Any = None
        gp_rate = sample.gp_rate
        if sample.gp_M is not None:
            parsed = as_real_imag(sample.gp_M)
            if parsed is not None:
                real, imag = parsed
        fixture = self.samples.get(sample.sample_id) or self.samples.nearest(sample.config)
        if fixture is not None:
            real = real if real is not None else fixture["gp_M_real"]
            imag = imag if imag is not None else fixture["gp_M_imag"]
            gp_rate = gp_rate if gp_rate is not None else fixture["gp_rate"]
        if real is None or imag is None or gp_rate is None:
            raise HTTPException(
                status_code=422,
                detail="GP ground truth required: send gp_M/gp_rate or a fixture sample_id",
            )
        return real, imag, float(gp_rate), fixture

    # ------------------------------------------------------------- publish
    async def publish(
        self,
        sample: SampleIn,
        model: str = "auto",
        decision: dict[str, Any] | None = None,
        frames: int = 1,
        transport: str | None = None,
        latency_ms: float = 0.0,
        cid: str | None = None,
    ) -> dict[str, Any]:
        gp_real, gp_imag, gp_rate, _ = await self._gp_truth(sample)
        payload = {
            "config": sample.config,
            "model": model,
            "include_M": True,
            "include_decision": True,
        }
        dl = await self._post("/predict", payload)
        decision = decision or dl.get("decision")
        dl_real, dl_imag = dl.get("M_real"), dl.get("M_imag")
        if dl_real is None or dl_imag is None:
            raise HTTPException(status_code=502, detail="inference returned no beamforming matrix")
        diff = metrics.diff_report(dl_real, dl_imag, gp_real, gp_imag, dl["rate"], gp_rate)
        cid = cid or self.new_cid()
        transport = transport or sample.transport
        source = provenance.source_hop(
            cid, sample.sample_id, sample.config, gp_real, gp_imag, transport, sample.device_id
        )
        travel = provenance.travel_hop(cid, transport, frames, True, latency_ms, 0)
        destination = provenance.destination_hop(cid, dl, diff)
        validation = provenance.validate(source, travel, destination, diff, decision, True)

        self.store.insert(
            "samples",
            {
                "cid": cid,
                "sample_id": sample.sample_id,
                "config": sample.config,
                "transport": transport,
                "device_id": sample.device_id,
                "source_ts": sample.source_ts,
            },
        )
        self.store.insert(
            "gp_truth",
            {
                "cid": cid,
                "sample_id": sample.sample_id,
                "rate": gp_rate,
                "M_real": gp_real,
                "M_imag": gp_imag,
                "m_hash": diff["gp_hash"],
            },
        )
        self.store.insert(
            "dl_outputs",
            {
                "cid": cid,
                "model": dl["model"],
                "rate": dl["rate"],
                "M_real": dl_real,
                "M_imag": dl_imag,
                "m_hash": diff["dl_hash"],
                "latency_ms": dl.get("latency_ms"),
                "decision": decision,
            },
        )
        if decision:
            self.store.insert("jev_decisions", {"cid": cid, **decision})
        for hop in (source, travel, destination):
            self.store.insert("provenance_hops", hop)

        report = {
            "cid": cid,
            "sample_id": sample.sample_id,
            "config": sample.config,
            "transport": transport,
            "device_id": sample.device_id,
            "source": source,
            "travel": travel,
            "destination": destination,
            "diff": diff,
            "decision": decision,
            "validation": validation,
            "dl": {key: value for key, value in dl.items() if key not in ("M_real", "M_imag")},
            "gp": {"rate": gp_rate},
            "dl_matrix": {"real": dl_real, "imag": dl_imag},
            "gp_matrix": {"real": gp_real, "imag": gp_imag},
        }
        self.store.upsert("validation_reports", "cid", cid, report)
        self.store.xadd(
            f"stream:{cid}",
            {
                "type": "RESULT",
                "seq": str(frames),
                "model": str(dl["model"]),
                "rate": str(dl["rate"]),
            },
        )
        self.store.set_json(f"cache:result:{cid}", self.summary(report))
        self.store.set_json(
            "model:active", {"model": dl["model"], "cid": cid, "ts": time.time()}
        )
        return self.summary(report, include_matrices=False) | {
            "urls": self.urls(cid),
            "dl_matrix": report["dl_matrix"],
            "gp_matrix": report["gp_matrix"],
            "diff": diff,
        }

    def urls(self, cid: str) -> dict[str, str]:
        base = self.settings.public_base_url.rstrip("/")
        return {
            "sample": f"{base}/samples/{cid}",
            "validation": f"{base}/validation/{cid}",
            "stream": f"{base}/streams/{cid}",
            "api": f"{base}/api/samples/{cid}",
        }

    def summary(self, report: dict[str, Any], include_matrices: bool = False) -> dict[str, Any]:
        summary = {
            "cid": report["cid"],
            "sample_id": report.get("sample_id"),
            "transport": report.get("transport"),
            "decision": report.get("decision"),
            "gp": report.get("gp"),
            "dl": {
                key: value
                for key, value in report.get("dl", {}).items()
                if key not in ("M_real", "M_imag", "decision")
            },
            "destination": report.get("destination"),
            "validation": report.get("validation"),
        }
        if include_matrices:
            summary["dl_matrix"] = report.get("dl_matrix")
            summary["gp_matrix"] = report.get("gp_matrix")
        return summary

    def get_report(self, cid: str) -> dict[str, Any] | None:
        return self.store.find_one("validation_reports", "cid", cid)

    def list_samples(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.store.find("samples", limit=limit)

    def list_stream(self, cid: str) -> list[dict[str, Any]]:
        return self.store.xrange(f"stream:{cid}", count=self.settings.max_stream_frames)

    def active_model(self) -> dict[str, Any] | None:
        return self.store.get_json("model:active")

    # ------------------------------------------------------------- frames
    async def handle_frame(self, cid: str, encoded: str) -> Frame:
        try:
            raw = base64.b64decode(encoded, validate=True)
            frame, _ = decode_frame(raw)
        except (ValueError, FrameCorrupt) as exc:
            raise HTTPException(status_code=422, detail=f"bad frame: {exc}") from exc
        self.store.xadd(
            f"stream:{cid}",
            {
                "type": frame.type.name,
                "seq": str(frame.seq),
                "payload": frame.payload.decode("utf-8", errors="replace")[:2000],
            },
        )
        if frame.type == FrameType.DATA_REQ:
            return await self._on_data_request(cid, frame)
        if frame.type == FrameType.DATA_ACK:
            return await self._on_data_ack(cid, frame)
        if frame.type == FrameType.MODEL_SWITCH:
            data = payload_json(frame) or {}
            model = str(data.get("model") or "")
            if model:
                try:
                    await self._post("/switch", {"model": model})
                    active = {"model": model, "cid": cid, "ts": time.time()}
                    self.store.set_json("model:active", active)
                except HTTPException:
                    pass
            return Frame(type=FrameType.RECV_OK, seq=frame.seq, payload=frame.payload)
        raise HTTPException(status_code=400, detail=f"unsupported frame type: {frame.type.name}")

    async def _on_data_request(self, cid: str, frame: Frame) -> Frame:
        data = payload_json(frame) or {}
        sample_id = str(data.get("sample_id") or "")
        fixture = self.samples.get(sample_id) if sample_id else None
        config = data.get("config") or (fixture["config_raw"] if fixture else None)
        if config is None:
            raise HTTPException(status_code=422, detail="DATA_REQ needs sample_id or config")
        sample = SampleIn(
            sample_id=sample_id or "adhoc",
            config=list(config),
            gp_rate=data.get("gp_rate"),
            gp_M=data.get("gp_M"),
            transport=str(data.get("transport") or "simulator"),
            device_id=str(data.get("device_id") or "sim-01"),
            source_ts=data.get("source_ts"),
        )
        decision = await self._post(
            "/jev_decide", {"config": sample.config, "probe": data.get("probe")}
        )
        pending = {
            "sample": sample.model_dump(),
            "decision": decision,
            "created": time.time(),
            "seq": frame.seq,
        }
        self._pending[cid] = pending
        self.store.set_json(f"pending:{cid}", pending)
        ack = AckMachine()
        ack.data_request(frame.seq)
        recv = Frame(
            type=FrameType.RECV_OK,
            seq=frame.seq,
            payload=_json_bytes(
                {
                    "chosen_model": decision["chosen_model"],
                    "confidence": decision["confidence"],
                    "scores": decision.get("scores"),
                    "probe": decision.get("probe"),
                    "cid": cid,
                }
            ),
        )
        ack.on_frame(recv)
        self.store.set_json(f"ack:{cid}", ack.to_dict())
        return recv

    async def _on_data_ack(self, cid: str, frame: Frame) -> Frame:
        pending = self._pending.pop(cid, None) or self.store.get_json(f"pending:{cid}")
        if not pending:
            raise HTTPException(status_code=409, detail=f"no pending sample for cid {cid}")
        ack_data = self.store.get_json(f"ack:{cid}") or {}
        ack = _ack_from_dict(ack_data)
        ack.on_frame(frame)
        self.store.set_json(f"ack:{cid}", ack.to_dict())
        sample = SampleIn(**pending["sample"])
        latency_ms = max(0.0, (time.time() - float(pending["created"])) * 1000.0)
        result = await self.publish(
            sample,
            model=str(pending["decision"]["chosen_model"]),
            decision=pending["decision"],
            frames=2,
            transport=sample.transport,
            latency_ms=latency_ms,
            cid=cid,
        )
        return Frame(
            type=FrameType.RESULT,
            seq=frame.seq,
            payload=_json_bytes(
                {
                    "cid": cid,
                    "chosen_model": pending["decision"]["chosen_model"],
                    "dl_rate": result["dl"]["rate"],
                    "gp_rate": result["gp"]["rate"],
                    "diff": result["diff"],
                    "urls": result["urls"],
                    "ack": ack.to_dict(),
                }
            ),
        )


def encode_out(frame: Frame) -> str:
    return base64.b64encode(frame.encode()).decode("ascii")


def _json_bytes(data: dict[str, Any]) -> bytes:
    import json

    return json.dumps(data, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")


def _ack_from_dict(data: dict[str, Any]) -> AckMachine:
    try:
        state = AckState(data.get("state", AckState.IDLE.value))
    except ValueError:
        state = AckState.IDLE
    return AckMachine(
        seq=data.get("seq"),
        state=state,
        attempts=int(data.get("attempts", 0)),
        chosen_model=data.get("chosen_model"),
        confidence=data.get("confidence"),
        last_error=data.get("last_error"),
        history=list(data.get("history", [])),
    )
