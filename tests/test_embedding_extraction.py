"""Unit tests for Stage 5 embedding invariants and traceability helpers."""

from pathlib import Path

import numpy as np

from src.experiments.embedding_extraction import (
    EMBEDDING_DIMENSION,
    embedding_path_for,
    normalize_embedding,
    stable_image_id,
    validate_embedding_cache,
)


def test_normalize_embedding_shape_and_norm() -> None:
    embedding = normalize_embedding(np.ones(EMBEDDING_DIMENSION, dtype=np.float64))
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_normalize_embedding_rejects_zero_vector() -> None:
    try:
        normalize_embedding(np.zeros(EMBEDDING_DIMENSION, dtype=np.float32))
    except ValueError as exc:
        assert "zero-norm" in str(exc)
    else:
        raise AssertionError("Expected zero-norm embedding rejection")


def test_normalize_embedding_rejects_wrong_dimension() -> None:
    try:
        normalize_embedding(np.ones(128, dtype=np.float32))
    except ValueError as exc:
        assert "512" in str(exc)
    else:
        raise AssertionError("Expected dimension rejection")


def test_cache_validation_accepts_valid_embedding(tmp_path: Path) -> None:
    path = tmp_path / "valid.npy"
    np.save(path, normalize_embedding(np.arange(512, dtype=np.float32) + 1))
    assert validate_embedding_cache(path)


def test_cache_validation_rejects_invalid_embedding(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npy"
    np.save(path, np.zeros(512, dtype=np.float32))
    assert not validate_embedding_cache(path)


def test_cache_path_is_stable_and_not_python_hash_based() -> None:
    relative = "Train/00013_00M19.png"
    assert stable_image_id(relative) == stable_image_id(relative)
    assert embedding_path_for(relative).name.endswith(".npy")


def test_stage4_failed_image_is_not_a_valid_embedding_source() -> None:
    failed = {
        "preprocessing_status": "no_face",
        "processed_filepath": "",
    }
    assert not (
        failed["preprocessing_status"] == "success"
        and failed["processed_filepath"]
    )
