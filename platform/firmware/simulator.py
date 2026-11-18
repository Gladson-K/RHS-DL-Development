#!/usr/bin/env python3
"""Static fixture simulator entry point (no STM32 board required).

If you do not have a board attached, run this on the host::

    python3 platform/firmware/simulator.py --backend http://localhost:8088
"""

from __future__ import annotations

import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

from gateway.bridge import main  # noqa: E402, RUF100

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
