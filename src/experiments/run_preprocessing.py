"""CLI entry point for Stage 4 MORPH preprocessing."""

import argparse

from src.experiments.preprocess_morph import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run_preprocessing(force=args.force))


if __name__ == "__main__":
    main()
