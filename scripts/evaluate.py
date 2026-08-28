import argparse
import torch
from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.models import HaloBeamGNNNet
from rhs_beamforming.nas.evaluator import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=20)
    args = p.parse_args()

    data = generate_debug_dataset(samples=args.samples)
    model = HaloBeamGNNNet()
    metrics = evaluate_model(model, data["H"][:2])
    print(metrics)


if __name__ == "__main__":
    main()
