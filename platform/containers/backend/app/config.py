"""Backend configuration (environment driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the backend container."""

    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "rhs_platform")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    inference_url: str = os.getenv("INFERENCE_URL", "http://localhost:8091")
    fixture: str = os.getenv("FIXTURE", "/app/fixtures/static_samples.npz")
    artifact_dirs: tuple[str, ...] = tuple(
        item for item in os.getenv("ARTIFACT_DIRS", "").split(":") if item
    )
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8088")
    inference_timeout: float = float(os.getenv("INFERENCE_TIMEOUT", "60"))
    max_stream_frames: int = int(os.getenv("MAX_STREAM_FRAMES", "200"))


SETTINGS = Settings()
