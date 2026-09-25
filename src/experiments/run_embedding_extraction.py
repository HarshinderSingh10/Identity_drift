"""CLI for Stage 5 MORPH embedding extraction."""

from __future__ import annotations

import argparse
import json

from src.experiments.embedding_extraction import run_full_extraction, run_smoke_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 ArcFace embedding extraction")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.smoke_test:
        result = run_smoke_test(sample_size=args.sample_size)
        print(json.dumps(result, indent=2))
        print(f"MODEL: {result['model']}")
        print("EMBEDDING DIMENSION: 512")
        print("NORMALIZATION: L2")
        print(f"EXECUTION PROVIDER: {result['active_providers']}")
        print(f"CUDA AVAILABLE: {result['cuda_available']}")
        print(f"GPU ACTUALLY USED: {'YES' if result['gpu_actually_used'] else 'NO'}")
        print(f"REPRODUCIBILITY: {result['reproducibility']}")
        return 0 if result["status"] == "PASS" else 1

    print(json.dumps(run_full_extraction(batch_size=args.batch_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

