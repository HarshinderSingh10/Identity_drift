from __future__ import annotations

import cv2
import numpy as np
from insightface import model_zoo

from src.embedding.embedding_extractor import FaceEmbeddingExtractor


def test_extract_aligned_embedding_on_validation_crop_success() -> None:
    image_path = "data/raw/validation/lfw_112x112/0.bmp"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None

    extractor = FaceEmbeddingExtractor(normalize=True)
    extractor.providers = ["CPUExecutionProvider"]

    embedding = extractor.extract_aligned_embedding(image)

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, rtol=1e-4, atol=1e-4)


def test_extract_aligned_embedding_matches_direct_arcface_model() -> None:
    image_path = "data/raw/validation/cplfw_112x112/0.bmp"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None

    extractor = FaceEmbeddingExtractor(normalize=True)
    extractor.providers = ["CPUExecutionProvider"]

    emb_a = extractor.extract_aligned_embedding(image)
    model = model_zoo.get_model("buffalo_l", providers=["CPUExecutionProvider"])
    emb_b = np.asarray(model.get_feat(image), dtype=np.float32).reshape(-1)
    emb_b = emb_b / np.linalg.norm(emb_b)

    np.testing.assert_allclose(emb_a, emb_b, rtol=1e-5, atol=1e-5)
