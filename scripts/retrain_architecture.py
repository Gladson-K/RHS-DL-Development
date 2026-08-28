import argparse
import json
from pathlib import Path
import torch
from rhs_beamforming.models import HaloBeamGNNNet
from rhs_beamforming.nas.model_space import RHSAdaptiveBeamformingModelSpace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", required=True)
    p.add_argument("--output", default="checkpoints/nas_best.pt")
    args = p.parse_args()

    arch = json.loads(Path(args.architecture).read_text(encoding="utf-8"))

    model = RHSAdaptiveBeamformingModelSpace(
        hidden_dim=128,
        depth=arch.get("depth", 3),
        operator=arch.get("operator", "bottleneck"),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "architecture": arch}, out)
    print(f"Saved initial architecture checkpoint: {out}")


if __name__ == "__main__":
    main()
