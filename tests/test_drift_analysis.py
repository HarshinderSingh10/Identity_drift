from pathlib import Path

import numpy as np
import pytest

from src.experiments.drift_analysis import (
    age_gap_group,
    bootstrap_identity_cluster,
    build_reference_observations,
    compute_cosine_distance,
    compute_cosine_similarity,
    compute_euclidean_distance,
)


def test_metric_relationship_for_normalized_vectors() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert compute_cosine_similarity(a, b) == 0.0
    assert compute_cosine_distance(a, b) == 1.0
    assert compute_euclidean_distance(a, b) ** 2 == pytest.approx(2.0)


def test_age_gap_groups_are_frozen() -> None:
    assert [age_gap_group(x) for x in (0, 2, 3, 4, 5, 9, 10, 19, 20)] == [
        "0-2", "0-2", "3-4", "3-4", "5-9", "5-9", "10-19", "10-19", "20+"
    ]


def test_reference_uses_age_then_filename_tie_break() -> None:
    observations = {
        "id": [
            {"filename": "b.JPG", "age": 18},
            {"filename": "a.JPG", "age": 18},
            {"filename": "c.JPG", "age": 19},
        ]
    }
    for values in observations.values():
        values.sort(key=lambda x: (x["age"], x["filename"]))
    assert build_reference_observations(observations)["id"]["filename"] == "a.JPG"


def test_identity_cluster_bootstrap_is_seeded() -> None:
    data = {"a": [0.1, 0.2], "b": [0.3]}
    first = bootstrap_identity_cluster(data, n_bootstrap=20, seed=42)
    second = bootstrap_identity_cluster(data, n_bootstrap=20, seed=42)
    assert first == second
