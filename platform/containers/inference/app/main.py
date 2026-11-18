"""FastAPI inference service: GP reference, DL models and JEV decisions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from protocol.schemas import PredictRequest, SwitchRequest

from .config import SETTINGS
from .engine import Engine, EngineUnavailable, GpTruthUnavailable, UnknownModel

engine = Engine(SETTINGS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.load()
    yield


app = FastAPI(
    title="RHS Inference Server",
    version="0.1.0",
    description="GP reference, DL beamforming models and JEV ACK-driven model switching.",
    lifespan=lifespan,
)


def _guard(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except UnknownModel as exc:
        raise HTTPException(status_code=404, detail=f"unknown model: {exc}") from exc
    except GpTruthUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "rhs-inference",
        "version": "0.1.0",
        "endpoints": [
            "/healthz",
            "/models",
            "/predict",
            "/predict_dl",
            "/predict_gp",
            "/jev_decide",
            "/switch",
        ],
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    status = engine.status()
    code = "ok" if engine.loaded else "degraded"
    return {"status": code, **status}


@app.get("/models")
def models() -> dict[str, Any]:
    return {"models": engine.registry(), "active_model": engine.active_model}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    return _guard(
        engine.predict,
        request.config,
        model=request.model,
        include_m=request.include_M,
        probe=request.probe,
    )


@app.post("/predict_dl")
def predict_dl(request: PredictRequest) -> dict[str, Any]:
    model = "hybrid_jev" if request.model == "auto" else request.model
    if model == "gp":
        raise HTTPException(status_code=400, detail="use /predict_gp for the GP reference")
    return _guard(engine.predict, request.config, model=model, include_m=request.include_M)


@app.post("/predict_gp")
def predict_gp(request: PredictRequest) -> dict[str, Any]:
    return _guard(engine.predict, request.config, model="gp", include_m=request.include_M)


@app.post("/jev_decide")
def jev_decide(request: PredictRequest) -> dict[str, Any]:
    return _guard(engine.jev_decide, request.config, probe=request.probe)


@app.post("/switch")
def switch(request: SwitchRequest) -> dict[str, Any]:
    return _guard(engine.switch, request.model)
