import argparse
import json
from pathlib import Path
import torch

from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.nas.model_space import RHSAdaptiveBeamformingModelSpace
from rhs_beamforming.nas.evaluator import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/nas/random.yaml")
    p.add_argument("--max-trials", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--output", default="search_results/smoke")
    args = p.parse_args()

    data = generate_debug_dataset(samples=args.samples)
    H = data["H"]

    candidates = []
    operators = ["bottleneck", "complex_residual"]
    depths = [2, 3, 4]

    for trial in range(args.max_trials):
        operator = operators[trial % len(operators)]
        depth = depths[trial % len(depths)]
        model = RHSAdaptiveBeamformingModelSpace(
            hidden_dim=128,
            depth=depth,
            operator=operator,
        )

        metrics = evaluate_model(model, H[:2], normalized_rate=0.0)
        metrics.update({
            "trial": trial,
            "operator": operator,
            "depth": depth,
        })
        candidates.append(metrics)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
