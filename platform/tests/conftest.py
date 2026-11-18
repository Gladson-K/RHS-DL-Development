"""Test path setup: make ``protocol`` and the backend ``app`` importable."""

from __future__ import annotations

import sys
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
BACKEND = PLATFORM / "containers/backend"
for path in (PLATFORM, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
