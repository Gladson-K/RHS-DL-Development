#!/usr/bin/env python3
"""Validate the real 64x64 MIMO HDF5 dataset (shapes, ranges, physics sanity)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhs_beamforming.data.mimo64 import validation_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    report = validation_report(args.h5)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"saved {args.out}")

    ok = (
        report["metadata_cols"] >= 15
        and report["datasets"].get("M", {}).get("shape", [0])[1:] == [20, 64, 10]
        and report["nan_free"]
        and 6.5 <= report["freq_ghz_min"] and report["freq_ghz_max"] <= 15.5
    )
    print("VALID" if ok else "INVALID")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
