import numpy as np
import pytest

from src.experiments.recognition_analysis import (
    apply_frozen_threshold,
    construct_impostor_pairs,
    drift_bin,
    select_validation_threshold,
)


def test_drift_bins_are_deterministic() -> None:
    assert drift_bin(0.0) == "0.00-0.10"
    assert drift_bin(0.1) == "0.10-0.20"
    assert drift_bin(1.0) == ">1.00"


def test_validation_threshold_is_selected_from_impostors() -> None:
    rows = [
        {"label": 0, "cosine_similarity": 0.9},
        {"label": 0, "cosine_similarity": 0.8},
        {"label": 0, "cosine_similarity": 0.7},
        {"label": 1, "cosine_similarity": 0.85},
        {"label": 1, "cosine_similarity": 0.95},
    ]
    result = select_validation_threshold(rows, target_far=0.34)
    assert result["threshold_metric"] == "cosine_similarity"
    assert result["frozen_before_test"] is True


def test_apply_frozen_threshold_marks_only_genuine_errors() -> None:
    rows = [
        {"label": 1, "cosine_similarity": 0.5},
        {"label": 0, "cosine_similarity": 0.5},
    ]
    scored = apply_frozen_threshold(rows, 0.6)
    assert scored[0]["verification_error"] == 1
    assert scored[1]["verification_error"] == 0


def test_impostor_sampling_is_deterministic_and_within_split() -> None:
    metadata = {
        f"{i}.JPG": {
            "filename": f"{i}.JPG",
            "identity_id": str(i // 2),
            "filename_age": "20",
            "research_split": "research_validation",
        }
        for i in range(8)
    }
    first = construct_impostor_pairs(metadata, {"research_validation": 5}, seed=42)
    second = construct_impostor_pairs(metadata, {"research_validation": 5}, seed=42)
    assert first == second
    assert len(first) == 5
    assert all(row["identity_a"] != row["identity_b"] for row in first)
    assert len({tuple(sorted((r["image_a"], r["image_b"]))) for r in first}) == 5
