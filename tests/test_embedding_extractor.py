#tests/test_embedding_extractor.py

"""Small initialization and optional sample-image demo for the embedding extractor."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedding_extractor import (  # noqa: E402
    FaceEmbeddingError,
    FaceEmbeddingExtractor,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo test."""

    parser = argparse.ArgumentParser(
        description="Initialize InsightFace and optionally extract one embedding."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a sample face image.",
    )
    parser.add_argument(
        "--face-selection",
        choices=["error", "largest", "highest_confidence"],
        default="error",
        help="How to handle multiple detected faces when --image is provided.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the extractor initialization check and optional extraction demo."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    print("Available ONNX Runtime providers:")
    print(FaceEmbeddingExtractor.available_providers())

    extractor = FaceEmbeddingExtractor(face_selection=args.face_selection)
    extractor.initialize()

    print("Requested providers:")
    print(extractor.providers)
    print("Active providers:")
    print(extractor.active_providers or "Provider details unavailable from model sessions.")
    print("Initialization successful.")

    if args.image is None:
        print("No sample image provided; skipping extraction.")
        return 0

    try:
        embedding = extractor.extract_embedding(args.image)
    except FaceEmbeddingError as exc:
        print(f"Embedding extraction failed: {exc}")
        return 1

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding L2 norm: {np.linalg.norm(embedding):.6f}")
    print(f"Observed embedding dimension: {extractor.embedding_dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
