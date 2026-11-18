"""Artifact discovery for already-published NAS results, figures and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

KIND_BY_SUFFIX = {
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".svg": "image",
    ".pdf": "pdf",
    ".html": "html",
    ".tex": "tex",
    ".yaml": "yaml",
    ".yml": "yaml",
}

MEDIA_TYPES = {
    "json": "application/json",
    "csv": "text/csv",
    "markdown": "text/markdown",
    "image": "image/png",
    "pdf": "application/pdf",
    "html": "text/html",
    "tex": "text/plain",
    "yaml": "text/plain",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints"}
MAX_INDEX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    """One file exposed under ``/results/{run}/{name}``."""

    run: str
    name: str
    source: str
    path: str
    kind: str
    size: int
    mtime: float

    @property
    def key(self) -> str:
        return f"{self.run}/{self.name}"

    @property
    def url(self) -> str:
        return f"/results/{quote(self.run)}/{quote(self.name, safe='/')}"

    def to_dict(self) -> dict[str, object]:
        return {
            "_id": self.key,
            "run": self.run,
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "kind": self.kind,
            "size": self.size,
            "mtime": self.mtime,
        }


def scan(dirs: tuple[str, ...] | list[str]) -> list[Artifact]:
    """Walk the mounted artifact roots and index supported files."""
    artifacts: dict[str, Artifact] = {}
    for raw_root in dirs:
        root = Path(raw_root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            suffix = path.suffix.lower()
            kind = KIND_BY_SUFFIX.get(suffix)
            if kind is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > MAX_INDEX_BYTES:
                continue
            rel = path.relative_to(root)
            if len(rel.parts) >= 2:
                run = rel.parts[0]
                name = "/".join(rel.parts[1:])
            else:
                run = root.name
                name = rel.name
            artifact = Artifact(
                run=run,
                name=name,
                source=root.name,
                path=str(path),
                kind=kind,
                size=int(stat.st_size),
                mtime=float(stat.st_mtime),
            )
            artifacts.setdefault(artifact.key, artifact)
    return sorted(artifacts.values(), key=lambda item: (item.run, item.name))


def by_run(artifacts: list[Artifact]) -> dict[str, list[Artifact]]:
    runs: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        runs.setdefault(artifact.run, []).append(artifact)
    return runs


def media_type(artifact: Artifact) -> str:
    if artifact.kind == "image":
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(Path(artifact.path).suffix.lower(), "application/octet-stream")
    return MEDIA_TYPES.get(artifact.kind, "application/octet-stream")
