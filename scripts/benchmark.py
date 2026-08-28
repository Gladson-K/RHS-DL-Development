import argparse
from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.models import HaloBeamGNNNet
from rhs_beamforming.nas.evaluator import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=2)
    args = p.parse_args()

    data = generate_debug_dataset(samples=args.samples)
    model = HaloBeamGNNNet()
    print(evaluate_model(model, data["H"]))


if __name__ == "__main__":
    main()
