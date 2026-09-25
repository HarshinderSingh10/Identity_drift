"""Diagnostic script for aligned 112x112 validation crops.

This script checks whether the current detector-based InsightFace path fails on
already-aligned validation crops and whether the ArcFace recognition model can be
used directly on the aligned crop without detector-based localization.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface import model_zoo

from src.embedding.embedding_extractor import FaceEmbeddingExtractor

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "validation"
DATASETS = [
    "agedb_30_112x112",
    "calfw_112x112",
    "cplfw_112x112",
    "lfw_112x112",
]


def main() -> None:
    extractor = FaceEmbeddingExtractor(normalize=True)
    try:
        extractor.initialize()
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"Extractor initialize fallback error: {type(exc).__name__}: {exc}")
    for dataset_name in DATASETS:
        dataset_dir = DATA_ROOT / dataset_name
        sample_paths = sorted(dataset_dir.glob("*.bmp"))[:2]
        print(f"DATASET {dataset_name}")
        for image_path in sample_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            print(f"  IMAGE {image_path.name}")
            if image is None:
                print("    OpenCV read failed")
                continue
            print(f"    shape={image.shape} dtype={image.dtype} channels={1 if image.ndim == 2 else image.shape[-1]} min={image.min()} max={image.max()}")

            try:
                embedding = extractor.extract_from_image(image)
                print(f"    detector path: success shape={embedding.shape} norm={np.linalg.norm(embedding):.6f}")
            except Exception as exc:  # pragma: no cover - diagnostic script
                print(f"    detector path: fail {type(exc).__name__}: {exc}")

            try:
                aligned = extractor.extract_aligned_embedding(image)
                print(f"    aligned-rec path: success shape={aligned.shape} norm={np.linalg.norm(aligned):.6f}")
            except Exception as exc:  # pragma: no cover - diagnostic script
                print(f"    aligned-rec path: fail {type(exc).__name__}: {exc}")

            try:
                rec_model = model_zoo.get_model("buffalo_l", providers=["CPUExecutionProvider"])
                feature = np.asarray(rec_model.get_feat(image), dtype=np.float32).reshape(-1)
                print(f"    raw ArcFace model: success shape={feature.shape} norm={np.linalg.norm(feature):.6f}")
            except Exception as exc:  # pragma: no cover - diagnostic script
                print(f"    raw ArcFace model: fail {type(exc).__name__}: {exc}")

        print()


if __name__ == "__main__":
    main()
