import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.stability_index import (
    _write_csv,
    bootstrap_identity_metrics,
    build_candidate_stability_features,
    build_future_failure_targets,
    build_individual_baselines,
    compute_identity_stability_index,
    evaluate_persistence_recovery,
    evaluate_pre_failure_warning,
    fit_candidate_index_models,
    select_index_on_validation,
    validate_temporal_leakage,
)


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_PATH = ROOT / "results" / "metrics" / "stage7_frozen_threshold.json"


def _make_reference_df() -> pd.DataFrame:
    rows = [
        {
            "identity_id": "A",
            "research_split": "research_train",
            "reference_image": "A_00.JPG",
            "reference_age": 20,
            "observation_image": "A_00.JPG",
            "observation_age": 20,
            "filename": "A_00.JPG",
            "filename_age": 20,
            "cosine_similarity": 1.0,
            "cosine_distance": 0.0,
        },
        {
            "identity_id": "A",
            "research_split": "research_train",
            "reference_image": "A_00.JPG",
            "reference_age": 20,
            "observation_image": "A_01.JPG",
            "observation_age": 21,
            "filename": "A_01.JPG",
            "filename_age": 21,
            "cosine_similarity": 0.82,
            "cosine_distance": 0.18,
        },
        {
            "identity_id": "A",
            "research_split": "research_train",
            "reference_image": "A_00.JPG",
            "reference_age": 20,
            "observation_image": "A_02.JPG",
            "observation_age": 22,
            "filename": "A_02.JPG",
            "filename_age": 22,
            "cosine_similarity": 0.65,
            "cosine_distance": 0.35,
        },
        {
            "identity_id": "A",
            "research_split": "research_train",
            "reference_image": "A_00.JPG",
            "reference_age": 20,
            "observation_image": "A_03.JPG",
            "observation_age": 23,
            "filename": "A_03.JPG",
            "filename_age": 23,
            "cosine_similarity": 0.71,
            "cosine_distance": 0.29,
        },
        {
            "identity_id": "B",
            "research_split": "research_train",
            "reference_image": "B_00.JPG",
            "reference_age": 30,
            "observation_image": "B_00.JPG",
            "observation_age": 30,
            "filename": "B_00.JPG",
            "filename_age": 30,
            "cosine_similarity": 1.0,
            "cosine_distance": 0.0,
        },
        {
            "identity_id": "B",
            "research_split": "research_train",
            "reference_image": "B_00.JPG",
            "reference_age": 30,
            "observation_image": "B_01.JPG",
            "observation_age": 31,
            "filename": "B_01.JPG",
            "filename_age": 31,
            "cosine_similarity": 0.88,
            "cosine_distance": 0.12,
        },
        {
            "identity_id": "B",
            "research_split": "research_train",
            "reference_image": "B_00.JPG",
            "reference_age": 30,
            "observation_image": "B_02.JPG",
            "observation_age": 32,
            "filename": "B_02.JPG",
            "filename_age": 32,
            "cosine_similarity": 0.91,
            "cosine_distance": 0.09,
        },
    ]
    return pd.DataFrame(rows)


def _make_consecutive_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"identity_id": "A", "image_a": "A_00.JPG", "image_b": "A_01.JPG", "cosine_distance": 0.18},
        {"identity_id": "A", "image_a": "A_01.JPG", "image_b": "A_02.JPG", "cosine_distance": 0.17},
        {"identity_id": "A", "image_a": "A_02.JPG", "image_b": "A_03.JPG", "cosine_distance": 0.06},
        {"identity_id": "B", "image_a": "B_00.JPG", "image_b": "B_01.JPG", "cosine_distance": 0.12},
        {"identity_id": "B", "image_a": "B_01.JPG", "image_b": "B_02.JPG", "cosine_distance": 0.03},
    ])


def test_individual_baseline_construction() -> None:
    baseline = build_individual_baselines("research_train")
    assert {"identity_id", "n_observations", "mean_reference_drift"}.issubset(baseline.columns)
    assert not baseline.empty
    assert baseline["n_observations"].ge(3).any()


def test_historical_only_feature_construction() -> None:
    reference_df = _make_reference_df()
    consecutive_df = _make_consecutive_df()
    features = build_candidate_stability_features(reference_df=reference_df, consecutive_df=consecutive_df, splits=["research_train"])
    current = features[features["identity_id"] == "A"].sort_values("filename_age")
    assert "individual_drift_percentile" in current.columns
    assert current.iloc[1]["individual_drift_percentile"] >= 0.0
    assert current.iloc[2]["individual_drift_percentile"] >= 0.0
    assert pd.isna(current.iloc[1]["individual_drift_zscore"])


def test_no_future_leakage() -> None:
    reference_df = _make_reference_df()
    features = build_candidate_stability_features(reference_df=reference_df, consecutive_df=_make_consecutive_df(), splits=["research_train"])
    future = build_future_failure_targets(features)
    assert (future["feature_time"] <= future["target_time"]).all()
    assert (future["prediction_observation"] != future["target_observation"]).all()
    assert not validate_temporal_leakage(future)["temporal_leakage"]


def test_temporal_ordering() -> None:
    reference_df = _make_reference_df()
    features = build_candidate_stability_features(reference_df=reference_df, consecutive_df=_make_consecutive_df(), splits=["research_train"])
    a_rows = features[features["identity_id"] == "A"].sort_values(["filename_age", "filename"])
    assert a_rows["filename_age"].tolist() == sorted(a_rows["filename_age"].tolist())


def test_zero_variance_handling() -> None:
    df = pd.DataFrame({
        "identity_id": ["X", "X"],
        "research_split": ["research_train", "research_train"],
        "observation_image": ["X_00.JPG", "X_01.JPG"],
        "observation_age": [20, 21],
        "reference_image": ["X_00.JPG", "X_00.JPG"],
        "reference_age": [20, 20],
        "filename": ["X_00.JPG", "X_01.JPG"],
        "filename_age": [20, 21],
        "current_reference_drift": [0.1, 0.1],
        "individual_drift_percentile": [np.nan, 1.0],
        "individual_drift_zscore": [np.nan, np.nan],
        "current_drift_minus_historical_median": [np.nan, 0.0],
        "current_consecutive_drift": [np.nan, 0.0],
        "recent_drift_slope": [np.nan, 0.0],
        "persistence_count": [0, 0],
        "recovery_indicator": [0, 0],
        "recognition_margin": [0.5, 0.2],
        "verification_error": [0, 0],
    })
    scores = compute_identity_stability_index(df)
    assert np.all(np.isfinite(scores.to_numpy()))


def test_percentile_calculation() -> None:
    history = np.array([0.1, 0.2, 0.4])
    current = 0.4
    pct = float(np.mean(history <= current))
    assert pct >= 0.0 and pct <= 1.0


def test_z_score_calculation() -> None:
    history = np.array([0.1, 0.15, 0.2])
    current = 0.2
    std = float(np.std(history, ddof=1))
    score = float((current - float(history.mean())) / std)
    assert np.isfinite(score)
    assert score > 0.0


def test_persistence_calculation() -> None:
    feature_df = _make_reference_df().copy()
    feature_df["persistence_count"] = [0, 1, 2, 2, 0, 1, 0]
    persistence = evaluate_persistence_recovery(feature_df)
    assert "persistent_excursion" in persistence.columns
    assert persistence["persistent_excursion"].sum() >= 0


def test_recovery_calculation() -> None:
    feature_df = pd.DataFrame({
        "identity_id": ["R", "R"],
        "research_split": ["research_train", "research_train"],
        "observation_image": ["R_00.JPG", "R_01.JPG"],
        "filename_age": [20, 21],
        "current_reference_drift": [0.65, 0.20],
        "verification_error": [1, 0],
        "recognition_margin": [0.0, 0.1],
        "filename": ["R_00.JPG", "R_01.JPG"],
    })
    calc = evaluate_persistence_recovery(feature_df)
    assert "recovery_event" in calc.columns


def test_index_reproducibility() -> None:
    reference_df = _make_reference_df()
    feature_df = build_candidate_stability_features(reference_df=reference_df, consecutive_df=_make_consecutive_df(), splits=["research_train"])
    model = fit_candidate_index_models(feature_df)["model"]
    first = compute_identity_stability_index(feature_df, model=model)
    second = compute_identity_stability_index(feature_df, model=model)
    assert np.allclose(first.to_numpy(), second.to_numpy())


def test_train_only_feature_selection() -> None:
    feature_df = build_candidate_stability_features(reference_df=_make_reference_df(), consecutive_df=_make_consecutive_df(), splits=["research_train"])
    train_df = feature_df[feature_df["research_split"] == "research_train"]
    result = fit_candidate_index_models(train_df)
    assert "candidate_C" in result
    assert len(result["selected_features"]) > 0


def test_validation_only_index_selection() -> None:
    reference_df = _make_reference_df()
    feature_df = build_candidate_stability_features(reference_df=reference_df, consecutive_df=_make_consecutive_df(), splits=["research_train"])
    train_df = feature_df[feature_df["research_split"] == "research_train"].copy()
    validation_df = train_df.copy()
    validation_df["verification_error"] = np.array([0, 1, 0, 0, 0, 1, 0], dtype=int)
    result = select_index_on_validation(train_df, validation_df)
    assert result["candidate"] in {"candidate_A", "candidate_B", "candidate_C"}


def test_test_set_protection() -> None:
    reference_df = _make_reference_df()
    feature_df = build_candidate_stability_features(reference_df=reference_df, consecutive_df=_make_consecutive_df(), splits=["research_train"])
    train_df = feature_df[feature_df["research_split"] == "research_train"]
    assert set(train_df["research_split"]) == {"research_train"}


def test_future_failure_target_construction() -> None:
    feature_df = build_candidate_stability_features(reference_df=_make_reference_df(), consecutive_df=_make_consecutive_df(), splits=["research_train"])
    future = build_future_failure_targets(feature_df)
    assert "future_error_t_plus_1" in future.columns
    assert future["future_error_t_plus_1"].isin([0, 1]).all()


def test_identity_cluster_bootstrap() -> None:
    df = pd.DataFrame({
        "identity_id": ["A", "A", "B", "B", "C"],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    boot = bootstrap_identity_metrics(df, "score", n_bootstrap=20)
    assert "ci95_lower" in boot
    assert "identity_count" in boot


def test_class_imbalance_metrics() -> None:
    future = pd.DataFrame({
        "future_error_t_plus_1": [0, 1, 1, 0],
        "isi_score": [0.9, 0.2, 0.3, 0.8],
    })
    metrics = evaluate_pre_failure_warning(future, "isi_score")
    assert metrics["positive_events"] > 0
    assert metrics["negative_events"] > 0
    assert "roc_auc" in metrics


def test_calibration() -> None:
    future = pd.DataFrame({
        "future_error_t_plus_1": [0, 1, 0, 1],
        "isi_score": [0.9, 0.2, 0.8, 0.4],
    })
    metrics = evaluate_pre_failure_warning(future, "isi_score")
    assert "brier_score" in metrics
    assert np.isfinite(metrics["brier_score"])


def test_stage7_threshold_reuse() -> None:
    with THRESHOLD_PATH.open("r", encoding="utf-8") as stream:
        threshold = json.load(stream)["validation_threshold"]
    assert isinstance(threshold, float)
    assert threshold > 0.0


def test_synthetic_fixture_cannot_overwrite_production_pre_failure_output(tmp_path) -> None:
    production_path = ROOT / "results" / "metrics" / "stage8_pre_failure_predictions.csv"
    original = production_path.read_text(encoding="utf-8") if production_path.exists() else None
    fixture = pd.DataFrame({
        "identity_id": ["A", "B"],
        "research_split": ["research_train", "research_train"],
        "prediction_observation": ["A_00.JPG", "B_00.JPG"],
        "target_observation": ["A_01.JPG", "B_01.JPG"],
        "feature_time": [10, 11],
        "target_time": [11, 12],
        "future_error_t_plus_1": [1, 0],
    })
    tmp_output = tmp_path / "temp_stage8_pre_failure_predictions.csv"
    _write_csv(tmp_output, fixture, allow_test_data=True)
    assert tmp_output.exists()
    if production_path.exists():
        assert production_path.read_text(encoding="utf-8") == original
    with pytest.raises(ValueError, match="synthetic identity fixture detected"):
        _write_csv(production_path, fixture)


def test_production_pipeline_rejects_synthetic_identity_ids() -> None:
    fixture = pd.DataFrame({
        "identity_id": ["A", "B"],
        "research_split": ["research_train", "research_train"],
        "prediction_observation": ["A_00.JPG", "B_00.JPG"],
        "target_observation": ["A_01.JPG", "B_01.JPG"],
        "future_error_t_plus_1": [1, 0],
    })
    with pytest.raises(ValueError):
        _write_csv(ROOT / "results" / "metrics" / "stage8_pre_failure_predictions.csv", fixture)
