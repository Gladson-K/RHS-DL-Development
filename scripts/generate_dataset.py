import argparse

from holobeam.config.loader import load_config
from holobeam.data.generator import DatasetGenerator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    generator = DatasetGenerator(
        config
    )

    generator.generate(
        output_dir=args.output
    )


if __name__ == "__main__":
    main()