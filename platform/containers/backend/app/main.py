"""Backend FastAPI app: ingestion, validation, artifact browser and retro UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from protocol.schemas import FrameIn, SampleIn, SwitchRequest

from .config import SETTINGS
from .db import Store
from .fixtures import StaticSamples
from .registry import media_type
from .service import PlatformService, encode_out

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Store(SETTINGS.mongo_uri, SETTINGS.mongo_db, SETTINGS.redis_url)
    store.connect()
    samples = StaticSamples(SETTINGS.fixture)
    service = PlatformService(SETTINGS, store, samples)
    service.refresh_artifacts()
    app.state.store = store
    app.state.samples = samples
    app.state.service = service
    yield
    await service.close()


app = FastAPI(
    title="RHS Streaming Validation Platform",
    version="0.1.0",
    description="Publish GP/DL beamforming outputs, validate provenance and visualise results.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _service(request: Request) -> PlatformService:
    return request.app.state.service


def _store(request: Request):
    return request.app.state.store


def _render(request: Request, name: str, **context: Any) -> HTMLResponse:
    context.setdefault("active", "")
    context.setdefault("title", "RHS Platform")
    return templates.TemplateResponse(request, name, context)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    service = _service(request)
    store = _store(request)
    models = await service.inference_models()
    return _render(
        request,
        "index.html",
        active="dashboard",
        stats={
            "samples": store.count("samples"),
            "reports": store.count("validation_reports"),
            "artifacts": len(service.artifacts),
            "runs": len(service.runs),
        },
        runs=sorted(service.runs.items()),
        samples=service.list_samples(12),
        models=models,
        active_model_service=service.active_model(),
        db=store.status(),
    )


@app.get("/results", response_class=HTMLResponse)
def results(request: Request) -> HTMLResponse:
    service = _service(request)
    return _render(
        request,
        "results.html",
        active="results",
        runs=sorted(service.runs.items()),
        artifacts=service.artifacts,
    )


@app.get("/results/{run}", response_class=HTMLResponse)
def run_detail(request: Request, run: str) -> HTMLResponse:
    service = _service(request)
    items = service.runs.get(run)
    if not items:
        raise HTTPException(status_code=404, detail=f"run not found: {run}")
    return _render(
        request,
        "run.html",
        active="results",
        run=run,
        artifacts=items,
        images=[item for item in items if item.kind == "image"][:12],
    )


@app.get("/results/{run}/{name:path}")
def artifact_file(request: Request, run: str, name: str):
    service = _service(request)
    artifact = service.get_artifact(run, name)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = Path(artifact.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact file missing")
    return FileResponse(path, media_type=media_type(artifact), filename=path.name)


@app.get("/samples/{cid}", response_class=HTMLResponse)
def sample_page(request: Request, cid: str) -> HTMLResponse:
    service = _service(request)
    report = service.get_report(cid)
    if report is None:
        raise HTTPException(status_code=404, detail=f"sample not found: {cid}")
    return _render(
        request,
        "sample.html",
        active="samples",
        report=report,
        summary=service.summary(report, include_matrices=False),
        urls=service.urls(cid),
        frames=service.list_stream(cid),
    )


@app.get("/validation/{cid}", response_class=HTMLResponse)
def validation_page(request: Request, cid: str) -> HTMLResponse:
    service = _service(request)
    report = service.get_report(cid)
    if report is None:
        raise HTTPException(status_code=404, detail=f"sample not found: {cid}")
    return _render(
        request,
        "validation.html",
        active="validation",
        report=report,
        summary=service.summary(report, include_matrices=False),
        urls=service.urls(cid),
        checks=report.get("validation", {}).get("checks", {}),
        diff=report.get("diff", {}),
    )


@app.get("/streams/{cid}", response_class=HTMLResponse)
def stream_page(request: Request, cid: str) -> HTMLResponse:
    service = _service(request)
    report = service.get_report(cid)
    return _render(
        request,
        "stream.html",
        active="streams",
        cid=cid,
        frames=service.list_stream(cid),
        report=report,
        urls=service.urls(cid) if report else None,
    )


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    service = _service(request)
    models = await service.inference_models()
    return _render(
        request,
        "models.html",
        active="models",
        models=models,
        active_model_service=service.active_model(),
    )


@app.post("/api/samples")
async def publish_sample(request: Request, sample: SampleIn) -> JSONResponse:
    service = _service(request)
    result = await service.publish(sample)
    return JSONResponse(result)


@app.get("/api/samples/{cid}")
def get_sample(request: Request, cid: str) -> JSONResponse:
    service = _service(request)
    report = service.get_report(cid)
    if report is None:
        raise HTTPException(status_code=404, detail=f"sample not found: {cid}")
    payload = service.summary(report, include_matrices=True) | {"urls": service.urls(cid)}
    return JSONResponse(payload)


@app.post("/api/streams/{cid}/frames")
async def ingest_frame(request: Request, cid: str, body: FrameIn) -> JSONResponse:
    service = _service(request)
    frame = await service.handle_frame(cid, body.frame)
    return JSONResponse({"frame": encode_out(frame), "seq": frame.seq, "type": frame.type.name})


@app.get("/api/validation/{cid}")
def get_validation(request: Request, cid: str) -> JSONResponse:
    service = _service(request)
    report = service.get_report(cid)
    if report is None:
        raise HTTPException(status_code=404, detail=f"sample not found: {cid}")
    return JSONResponse(report | {"urls": service.urls(cid)})


@app.get("/api/models")
async def get_models(request: Request) -> JSONResponse:
    service = _service(request)
    return JSONResponse(await service.inference_models())


@app.post("/api/switch")
async def switch_model(request: Request, body: SwitchRequest) -> JSONResponse:
    service = _service(request)
    return JSONResponse(await service.switch_model(body.model))


@app.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    store = _store(request)
    service = _service(request)
    db = store.status()
    inference = await service.inference_health()
    ok = db["mongo"] and db["redis"] and inference.get("status") == "ok"
    return JSONResponse(
        {
            "status": "ok" if ok else "degraded",
            "database": db,
            "inference": inference,
            "artifacts": len(service.artifacts),
        },
        status_code=200 if ok else 503,
    )


@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException) -> HTMLResponse:
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": exc.detail}, status_code=404)
    return templates.TemplateResponse(
        request,
        "404.html",
        {"active": "", "title": "Not found", "detail": exc.detail},
        status_code=404,
    )
