#!/usr/bin/env python3
"""Extract the 64x64 MIMO HDF5 dataset from its zip archive.

The zip stores a single ~18 GB HDF5 file. Extraction is streamed in 16 MB
chunks so it works on memory-constrained machines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhs_beamforming.data.mimo64 import extract_h5_from_zip, resolve_zip, validation_report

DEFAULT_SEARCH_DIRS = [
    "/mnt/d/WorkSpace_Glad/64x64_MIMO_Dataset",
    "/mnt/d/Workspace_Glad/64x64_MIMO_Dataset",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", default=None, help="path to 64x64_MIMO_beamforming_dataset.zip")
    p.add_argument("--dest", default="data/raw", help="destination directory for the .h5")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true", help="print a validation report after extraction")
    args = p.parse_args()

    zip_path = resolve_zip(args.zip, extra_dirs=DEFAULT_SEARCH_DIRS)
    print(f"zip: {zip_path}")
    h5 = extract_h5_from_zip(zip_path, args.dest, overwrite=args.overwrite)
    print(f"extracted: {h5} ({h5.stat().st_size / 1e9:.2f} GB)")
    if args.validate:
        report = validation_report(h5)
        import json

        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
