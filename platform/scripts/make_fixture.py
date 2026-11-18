#!/usr/bin/env python3
"""Extract a small, self-contained static-sample fixture for the platform.

The platform must boot and be testable without mounting the 1.5 GB full cache,
so a handful of held-out samples (configuration, GP beamformer, GP rate) are
extracted from the 5K study cache.

Usage::

    python3 platform/scripts/make_fixture.py --count 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/processed/mimo64_5000.npz"
DEFAULT_OUTPUT = ROOT / "platform/fixtures/static_samples.npz"


def build_fixture(source: Path, output: Path, count: int) -> None:
    cache = np.load(source, allow_pickle=True)
    indices = np.asarray(cache["test"])[:count]
    if indices.size == 0:
        raise SystemExit("no test samples found in the cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        config=np.asarray(cache["X"])[indices].astype(np.float32),
        config_raw=np.asarray(cache["X_raw"])[indices].astype(np.float32),
        gp_M=np.asarray(cache["M"])[indices].astype(np.complex64),
        gp_rate=np.asarray(cache["rate"])[indices].astype(np.float32),
        global_indices=np.asarray(cache["global_indices"])[indices].astype(np.int64),
        sample_ids=np.asarray([f"s{int(i):05d}" for i in indices]),
        rate_mean=np.asarray(cache["rate_mean"], dtype=np.float32),
        rate_std=np.asarray(cache["rate_std"], dtype=np.float32),
        feature_names=np.asarray(cache["feature_names"]),
    )
    print(f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB, {len(indices)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    build_fixture(args.source, args.output, args.count)


if __name__ == "__main__":
    main()
