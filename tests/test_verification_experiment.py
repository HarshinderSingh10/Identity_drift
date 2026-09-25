from __future__ import annotations

import numpy as np
import pytest

from src.experiments.verification_experiment import (
    calculate_roc_metrics,
    threshold_metrics,
)


def test_threshold_direction_and_confusion_metrics() -> None:
    result = threshold_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 0.5)
    assert result["TP"] == 2
    assert result["TN"] == 2
    assert result["FP"] == 0
    assert result["FN"] == 0
    assert result["FAR"] == 0.0
    assert result["FRR"] == 0.0
    assert result["TAR"] == 1.0
    assert result["TNR"] == 1.0
    assert result["accuracy"] == 1.0


def test_threshold_metrics_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        threshold_metrics([], [], 0.5)
    with pytest.raises(ValueError):
        threshold_metrics([0.1], [1, 0], 0.5)
    with pytest.raises(ValueError):
        threshold_metrics([np.nan], [1], 0.5)


def test_roc_auc_and_eer_for_perfect_separation() -> None:
    result = calculate_roc_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["eer"] == pytest.approx(0.0)


def test_roc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        calculate_roc_metrics([0.1, 0.2], [1, 1])


def test_normalized_embedding_euclidean_consistency() -> None:
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)
    cosine = float(np.dot(first, second))
    euclidean_squared = float(np.linalg.norm(first - second) ** 2)
    assert euclidean_squared == pytest.approx(2.0 * (1.0 - cosine))
