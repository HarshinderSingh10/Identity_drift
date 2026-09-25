"""Tests for mathematical embedding drift metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.drift_metrics import (  # noqa: E402
    batch_cosine_similarity,
    batch_identity_drift,
    cosine_distance,
    cosine_similarity,
    embedding_norm,
    euclidean_distance,
    identity_drift,
    is_normalized,
    summarize_drift,
    validate_embedding,
)


def test_identical_embeddings_have_zero_cosine_drift() -> None:
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert cosine_similarity(embedding, embedding) == pytest.approx(1.0)
    assert cosine_distance(embedding, embedding) == pytest.approx(0.0)
    assert identity_drift(embedding, embedding) == pytest.approx(0.0)


def test_orthogonal_embeddings_have_zero_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_embeddings_have_negative_one_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_non_normalized_embeddings_are_handled_correctly() -> None:
    assert cosine_similarity([2.0, 0.0], [10.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([2.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)


def test_zero_vector_raises_for_cosine_similarity() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("bad_embedding", [[np.nan, 1.0], [np.inf, 1.0]])
def test_nan_and_inf_raise_value_error(bad_embedding: list[float]) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_embedding(bad_embedding)


def test_dimension_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_batch_calculations_match_individual_calculations() -> None:
    reference = np.array([1.0, 0.0], dtype=np.float32)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    batch_similarities = batch_cosine_similarity(reference, embeddings)
    batch_drifts = batch_identity_drift(reference, embeddings)

    expected_similarities = np.array(
        [cosine_similarity(reference, row) for row in embeddings], dtype=np.float64
    )
    expected_drifts = np.array(
        [identity_drift(reference, row) for row in embeddings], dtype=np.float64
    )

    np.testing.assert_allclose(batch_similarities, expected_similarities)
    np.testing.assert_allclose(batch_drifts, expected_drifts)


def test_batch_dimension_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Batch embedding dimension"):
        batch_cosine_similarity([1.0, 0.0], np.array([[1.0, 0.0, 0.0]]))


def test_summary_statistics() -> None:
    drift_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    summary = summarize_drift(drift_values)

    assert summary["count"] == 5
    assert summary["mean"] == pytest.approx(0.2)
    assert summary["median"] == pytest.approx(0.2)
    assert summary["minimum"] == pytest.approx(0.0)
    assert summary["maximum"] == pytest.approx(0.4)
    assert summary["standard_deviation"] == pytest.approx(float(np.std(drift_values)))
    assert summary["percentile_25"] == pytest.approx(0.1)
    assert summary["percentile_75"] == pytest.approx(0.3)
    assert summary["percentile_95"] == pytest.approx(0.38)


def test_normalized_embedding_returns_true() -> None:
    assert is_normalized([1.0, 0.0, 0.0])


def test_slightly_non_normalized_embedding_outside_tolerance_returns_false() -> None:
    assert not is_normalized([1.0001, 0.0, 0.0], tolerance=1e-5)


def test_embedding_norm_and_euclidean_distance() -> None:
    assert embedding_norm([3.0, 4.0]) == pytest.approx(5.0)
    assert euclidean_distance([1.0, 2.0], [4.0, 6.0]) == pytest.approx(5.0)
