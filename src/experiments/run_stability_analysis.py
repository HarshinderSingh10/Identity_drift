"""CLI entry point for Stage 8 identity stability analysis."""

from __future__ import annotations

import argparse

from src.experiments.stability_index import run_stage8


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 individualized identity stability analysis")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "features", "index", "pre-failure", "persistence", "evaluation", "all"],
        default="all",
    )
    args = parser.parse_args()
    run_stage8(args.experiment)


if __name__ == "__main__":
    main()
