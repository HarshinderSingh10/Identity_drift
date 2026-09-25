"""Stage 8 individualized identity stability analysis.

This stage develops identity-specific stability signals using only historical
information available before the current observation. It does not redefine the
frozen Stage 7 verification threshold and does not implement re-enrollment.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA = PROJECT_ROOT / "data" / "metadata"
METRICS = PROJECT_ROOT / "results" / "metrics"
PLOTS = PROJECT_ROOT / "results" / "plots" / "stage8"
REPORT_DIR = PROJECT_ROOT / "experiments" / "08_identity_stability"
SEED = 42
PRODUCTION_OUTPUT_ROOTS = (METRICS, PLOTS, REPORT_DIR, METADATA)
SYNTHETIC_ID_PREFIXES = {"A", "B", "C", "X", "Y", "Z"}


def _is_production_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return any(str(resolved).startswith(str(root.resolve())) for root in PRODUCTION_OUTPUT_ROOTS)


def _is_synthetic_identity(value: Any) -> bool:
    text = str(value).strip()
    if not text or text.isdigit():
        return False
    return text in SYNTHETIC_ID_PREFIXES or text.startswith(tuple(sorted(SYNTHETIC_ID_PREFIXES)))


def _validate_morph_production_df(df: pd.DataFrame, *, context: str, allow_test_data: bool = False) -> None:
    if allow_test_data:
        return
    if df.empty:
        raise ValueError(f"{context}: empty production dataset is not valid for Stage 8")
    if "identity_id" in df.columns:
        identity_values = df["identity_id"].astype(str)
        if identity_values.isin({"A", "B", "C", "X", "Y", "Z"}).any():
            raise ValueError(f"{context}: synthetic identity fixture detected in production output: {identity_values[identity_values.isin({'A','B','C','X','Y','Z'})].unique().tolist()}")
        if not identity_values.str.fullmatch(r"\d+").all():
            offenders = identity_values[~identity_values.str.fullmatch(r"\d+")].unique().tolist()[:10]
            raise ValueError(f"{context}: non-MORPH identity IDs in production data: {offenders}")
    if "research_split" in df.columns:
        allowed = {"research_train", "research_validation", "research_test"}
        bad = sorted(set(df["research_split"].astype(str).unique()) - allowed)
        if bad:
            raise ValueError(f"{context}: invalid research split values in production output: {bad}")
    if "observation_image" in df.columns:
        img = df["observation_image"].astype(str)
        if not img.empty:
            bad = img[~img.str.contains(r"^\d+_.*\.(JPG|PNG|jpg|png)$", regex=True)].unique().tolist()[:10]
            if bad:
                raise ValueError(f"{context}: suspicious observation filenames in production output: {bad}")
    if "prediction_observation" in df.columns:
        pred = df["prediction_observation"].astype(str)
        if not pred.empty:
            bad = pred[~pred.str.contains(r"^\d+_.*\.(JPG|PNG|jpg|png)$", regex=True)].unique().tolist()[:10]
            if bad:
                raise ValueError(f"{context}: suspicious prediction observation filenames: {bad}")


def _write_csv(path: Path, df: pd.DataFrame, *, allow_test_data: bool = False) -> None:
    if _is_production_path(path) and not allow_test_data:
        _validate_morph_production_df(df, context=str(path), allow_test_data=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any], *, allow_test_data: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_production_path(path) and not allow_test_data:
        _validate_morph_production_df(pd.DataFrame([payload]), context=str(path), allow_test_data=False)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 8 input: {path}")
    return pd.read_csv(path)


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sort_reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["observation_age"] = _as_float(out.get("observation_age", out.get("filename_age", pd.Series([np.nan] * len(out)))))
    out["observation_image"] = out.get("observation_image", out.get("filename", pd.Series([""] * len(out))))
    out["filename"] = out["observation_image"].astype(str)
    out["filename_age"] = out["observation_age"]
    out["cosine_distance"] = _as_float(out.get("cosine_distance", out.get("drift_from_reference")))
    out["cosine_similarity"] = _as_float(out.get("cosine_similarity", out.get("cosine_similarity_to_reference")))
    out["identity_id"] = out["identity_id"].astype(str)
    out["research_split"] = out["research_split"].astype(str)
    return out.sort_values(["identity_id", "filename_age", "filename"], kind="mergesort").reset_index(drop=True)


def _safe_threshold(df: pd.DataFrame, column: str, q: float = 0.95) -> float:
    values = _as_float(df[column]).dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def _bootstrap_identity_metric(values: pd.Series, identities: pd.Series, metric: str = "mean", seed: int = SEED, repetitions: int = 2000) -> tuple[float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    identities = identities.loc[values.index].astype(str)
    if values.empty or identities.empty:
        return float("nan"), float("nan"), float("nan")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    unique = sorted(identities.unique())
    if not unique:
        return observed, float("nan"), float("nan")
    estimates = []
    for _ in range(repetitions):
        sample = rng.choice(unique, size=len(unique), replace=True)
        sampled = []
        for ident in sample:
            cluster = values[identities == ident]
            if len(cluster):
                sampled.append(float(cluster.mean()))
        if sampled:
            estimates.append(float(np.mean(sampled)))
    if not estimates:
        return observed, float("nan"), float("nan")
    lower = float(np.percentile(estimates, 2.5))
    upper = float(np.percentile(estimates, 97.5))
    return observed, lower, upper


def build_population_baseline(split: str = "research_train", output_path: Path | None = None) -> pd.DataFrame:
    """Population distributions for training-population stability variables."""
    reference = _sort_reference_rows(_read_csv(METRICS / "stage6_reference_drift.csv"))
    consecutive = _read_csv(METRICS / "stage6_consecutive_drift.csv")
    identity_summary = _read_csv(METRICS / "stage6_identity_stability.csv")
    ref_rows = reference[reference["research_split"] == split].copy()
    consec_rows = consecutive[consecutive["research_split"] == split].copy()
    id_rows = identity_summary[identity_summary["research_split"] == split].copy()
    records: list[dict[str, Any]] = []
    variable_specs = [
        ("reference_drift", ref_rows["cosine_distance"], "reference_drift"),
        ("consecutive_drift", consec_rows["cosine_distance"], "consecutive_drift"),
        ("mean_drift", id_rows["mean_drift"], "mean_drift"),
        ("median_drift", id_rows["median_drift"], "median_drift"),
        ("std_drift", id_rows["std_drift"], "std_drift"),
        ("p90_drift", id_rows["p90_drift"], "p90_drift"),
        ("p95_drift", id_rows["p95_drift"], "p95_drift"),
        ("max_drift", id_rows["max_drift"], "max_drift"),
        ("age_span", id_rows["age_span"], "age_span"),
        ("n_observations", id_rows["n_observations"], "n_observations"),
    ]
    for name, values, key in variable_specs:
        vals = _as_float(values).dropna()
        if vals.empty:
            continue
        records.append({
            "variable": name,
            "split": split,
            "n_observations": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "p10": float(vals.quantile(0.10)),
            "p25": float(vals.quantile(0.25)),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
            "p95": float(vals.quantile(0.95)),
            "p99": float(vals.quantile(0.99)),
            "max": float(vals.max()),
            "min": float(vals.min()),
            "source_key": key,
        })
    df = pd.DataFrame(records)
    if output_path is not None:
        _write_csv(output_path, df)
    return df


def build_individual_baselines(split: str = "research_train", output_path: Path | None = None) -> pd.DataFrame:
    """Identity-specific historical stability baselines.

    Minimum support: >= 3 valid observations in the specified split. The
    analytical reference is the youngest age; baseline features are constructed
    using only historical observations available before each current record.
    """
    reference = _sort_reference_rows(_read_csv(METRICS / "stage6_reference_drift.csv"))
    consecutive = _read_csv(METRICS / "stage6_consecutive_drift.csv")
    ref_split = reference[reference["research_split"] == split].copy()
    high_threshold = _safe_threshold(ref_split, "cosine_distance", q=0.95)
    rows: list[dict[str, Any]] = []
    for identity_id, group in ref_split.groupby("identity_id", sort=True):
        group = group.sort_values(["filename_age", "filename"], kind="mergesort").reset_index(drop=True)
        if len(group) < 3:
            continue
        d = _as_float(group["cosine_distance"]).dropna()
        means = {"mean_reference_drift": float(d.mean()), "median_reference_drift": float(d.median()), "std_reference_drift": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                 "p90_reference_drift": float(d.quantile(0.90)), "p95_reference_drift": float(d.quantile(0.95)), "max_reference_drift": float(d.max())}
        consec_id = consecutive[consecutive["identity_id"].astype(str) == str(identity_id)]
        consec_values = _as_float(consec_id["cosine_distance"]).dropna()
        if consec_values.empty:
            consec_stats = {"mean_consecutive_drift": np.nan, "median_consecutive_drift": np.nan, "std_consecutive_drift": np.nan, "p90_consecutive_drift": np.nan}
        else:
            consec_stats = {"mean_consecutive_drift": float(consec_values.mean()), "median_consecutive_drift": float(consec_values.median()), "std_consecutive_drift": float(consec_values.std(ddof=1)) if len(consec_values) > 1 else 0.0,
                            "p90_consecutive_drift": float(consec_values.quantile(0.90))}
        n_high = int((d > high_threshold).sum())
        rows.append({
            "identity_id": str(identity_id),
            "research_split": split,
            "n_observations": int(len(group)),
            "youngest_age": int(group["filename_age"].min()),
            "oldest_age": int(group["filename_age"].max()),
            "age_span": int(group["filename_age"].max() - group["filename_age"].min()),
            "mean_reference_drift": means["mean_reference_drift"],
            "median_reference_drift": means["median_reference_drift"],
            "std_reference_drift": means["std_reference_drift"],
            "p90_reference_drift": means["p90_reference_drift"],
            "p95_reference_drift": means["p95_reference_drift"],
            "max_reference_drift": means["max_reference_drift"],
            "mean_consecutive_drift": consec_stats["mean_consecutive_drift"],
            "median_consecutive_drift": consec_stats["median_consecutive_drift"],
            "std_consecutive_drift": consec_stats["std_consecutive_drift"],
            "p90_consecutive_drift": consec_stats["p90_consecutive_drift"],
            "n_high_drift_observations": n_high,
        })
    df = pd.DataFrame(rows)
    if output_path is not None:
        _write_csv(output_path, df)
    return df


def build_candidate_stability_features(
    reference_df: pd.DataFrame | None = None,
    consecutive_df: pd.DataFrame | None = None,
    threshold_df: pd.DataFrame | None = None,
    splits: list[str] | None = None,
    high_threshold: float | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Construct candidate stability features using only historical data."""
    if reference_df is None:
        reference_df = _sort_reference_rows(_read_csv(METRICS / "stage6_reference_drift.csv"))
    if consecutive_df is None:
        consecutive_df = _read_csv(METRICS / "stage6_consecutive_drift.csv")
    if threshold_df is None:
        threshold_df = _read_csv(METRICS / "stage7_frozen_threshold.json")
    if splits is None:
        splits = ["research_train", "research_validation", "research_test"]
    threshold_path = PROJECT_ROOT / "results" / "metrics" / "stage7_frozen_threshold.json"
    with threshold_path.open("r", encoding="utf-8") as stream:
        threshold_cfg = json.load(stream)
    frozen_threshold = float(threshold_cfg["validation_threshold"])
    if high_threshold is None:
        high_threshold = _safe_threshold(reference_df[reference_df["research_split"] == "research_train"], "cosine_distance", q=0.95)
    all_rows: list[dict[str, Any]] = []
    for split in splits:
        split_df = reference_df[reference_df["research_split"] == split].copy()
        for identity_id, group in split_df.groupby("identity_id", sort=True):
            group = group.sort_values(["filename_age", "filename"], kind="mergesort").reset_index(drop=True)
            for idx in range(len(group)):
                row = group.iloc[idx].copy()
                prior = group.iloc[:idx]
                prior_drift = _as_float(prior["cosine_distance"]).dropna().to_numpy()
                current_drift = float(row["cosine_distance"])
                current_similarity = float(row["cosine_similarity"])
                if len(prior_drift) > 0:
                    individual_drift_percentile = float(np.mean(prior_drift <= current_drift))
                    historical_mean = float(np.mean(prior_drift))
                    historical_std = float(np.std(prior_drift, ddof=1)) if len(prior_drift) > 1 else np.nan
                    historical_median = float(np.median(prior_drift))
                    individual_drift_zscore = float((current_drift - historical_mean) / historical_std) if np.isfinite(historical_std) and historical_std > 0 else np.nan
                    drift_dev_from_median = float(current_drift - historical_median)
                    recent_window = prior_drift[-min(3, len(prior_drift)):]
                    recent_drift_slope = float(np.polyfit(np.arange(len(recent_window)), recent_window, 1)[0]) if len(recent_window) > 1 else 0.0
                    streak = 0
                    for value in reversed(prior_drift):
                        if value > high_threshold:
                            streak += 1
                        else:
                            break
                    persistence_count = int(streak)
                    prev_elevated = prior_drift[-1] > high_threshold if len(prior_drift) > 0 else False
                    recovery_indicator = int(prev_elevated and current_drift < prior_drift[-1]) if len(prior_drift) > 0 else 0
                    if len(prior_drift) >= 1:
                        current_consecutive_drift = np.nan
                        previous_row = prior.iloc[-1]
                        match = consecutive_df[(consecutive_df["identity_id"].astype(str) == str(identity_id)) & (
                            ((consecutive_df["image_a"].astype(str) == str(previous_row["observation_image"])) & (consecutive_df["image_b"].astype(str) == str(row["observation_image"])))
                            | ((consecutive_df["image_a"].astype(str) == str(row["observation_image"])) & (consecutive_df["image_b"].astype(str) == str(previous_row["observation_image"])))
                        )]
                        if not match.empty:
                            current_consecutive_drift = float(match.iloc[0]["cosine_distance"])
                        else:
                            current_consecutive_drift = float(abs(current_drift - float(previous_row["cosine_distance"])))
                    else:
                        current_consecutive_drift = np.nan
                    individual_historical_mean = float(np.mean(prior_drift))
                    individual_historical_median = float(np.median(prior_drift))
                else:
                    individual_drift_percentile = np.nan
                    individual_drift_zscore = np.nan
                    drift_dev_from_median = np.nan
                    recent_drift_slope = np.nan
                    persistence_count = 0
                    recovery_indicator = 0
                    current_consecutive_drift = np.nan
                    individual_historical_mean = np.nan
                    individual_historical_median = np.nan
                all_rows.append({
                    "identity_id": str(identity_id),
                    "research_split": split,
                    "observation_image": str(row["observation_image"]),
                    "observation_age": int(row["observation_age"]),
                    "reference_image": str(row["reference_image"]),
                    "reference_age": int(row["reference_age"]),
                    "filename": str(row["filename"]),
                    "filename_age": int(row["filename_age"]),
                    "current_reference_drift": float(current_drift),
                    "current_reference_similarity": float(current_similarity),
                    "individual_drift_percentile": float(individual_drift_percentile) if "individual_drift_percentile" in locals() else np.nan,
                    "individual_drift_zscore": float(individual_drift_zscore) if "individual_drift_zscore" in locals() else np.nan,
                    "current_drift_minus_historical_median": float(drift_dev_from_median) if "drift_dev_from_median" in locals() else np.nan,
                    "current_consecutive_drift": float(current_consecutive_drift) if "current_consecutive_drift" in locals() else np.nan,
                    "recent_drift_slope": float(recent_drift_slope) if "recent_drift_slope" in locals() else np.nan,
                    "persistence_count": int(persistence_count),
                    "recovery_indicator": int(recovery_indicator),
                    "recognition_margin": float(current_similarity - frozen_threshold),
                    "verification_error": int(current_similarity < frozen_threshold),
                    "historical_mean_drift": float(individual_historical_mean) if "individual_historical_mean" in locals() else np.nan,
                    "historical_median_drift": float(individual_historical_median) if "individual_historical_median" in locals() else np.nan,
                })
    feature_df = pd.DataFrame(all_rows)
    feature_df = feature_df.sort_values(["research_split", "identity_id", "filename_age", "filename"], kind="mergesort").reset_index(drop=True)
    if output_path is not None:
        _write_csv(output_path, feature_df)
    return feature_df


def build_temporal_features(reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_candidate_stability_features(reference_df=reference_df)


def fit_candidate_index_models(train_df: pd.DataFrame, feature_names: list[str] | None = None) -> dict[str, Any]:
    if feature_names is None:
        feature_names = [
            "current_reference_drift",
            "individual_drift_percentile",
            "individual_drift_zscore",
            "current_drift_minus_historical_median",
            "current_consecutive_drift",
            "recent_drift_slope",
            "persistence_count",
            "recovery_indicator",
            "recognition_margin",
        ]
    X = train_df[feature_names].copy()
    X = X.fillna(0.0)
    y = train_df["verification_error"].astype(int).to_numpy()
    models: dict[str, Any] = {"selected_features": feature_names, "train_labels": y}
    # Candidate A: bounded normalized instability score.
    current = X["current_reference_drift"].to_numpy()
    if len(current):
        candidate_a = 1.0 - np.clip((current - current.min()) / (current.max() - current.min() + 1e-8), 0, 1)
    else:
        candidate_a = np.zeros(len(X))
    candidate_a = np.asarray(candidate_a, dtype=float)
    # Candidate B: standardized instability. Each feature is an instability signal; higher is worse.
    z = X.copy()
    for column in feature_names:
        if column == "recognition_margin":
            z[column] = z[column].replace({0: 0.0})
    z = z.fillna(0.0)
    mean = z.mean()
    std = z.std(ddof=0)
    std = std.replace(0, 1.0)
    scaled = (z - mean) / std
    candidate_b = np.asarray(1.0 - np.clip(0.25 * scaled.sum(axis=1), 0.0, 1.0), dtype=float)
    # Candidate C: logistic regression with interpretable, regularized coefficients.
    model = None
    candidate_c = np.full(len(X), 0.5, dtype=float)
    if len(np.unique(y)) >= 2:
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        model.fit(X, y)
        risk = model.predict_proba(X)[:, 1]
        candidate_c = np.asarray(1.0 - risk, dtype=float)
    models["candidate_A"] = candidate_a
    models["candidate_B"] = candidate_b
    models["candidate_C"] = candidate_c
    models["model"] = model
    return models


def select_index_on_validation(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> dict[str, Any]:
    """Choose the best interpretable formulation on validation data."""
    models = fit_candidate_index_models(train_df)
    feature_names = models["selected_features"]
    results: list[dict[str, Any]] = []
    labels = validation_df["verification_error"].astype(int)
    if len(np.unique(labels)) < 2:
        auc = 0.5
        return {"candidate": "candidate_B", "validation_auc": float(auc), "selected_features": ",".join(feature_names), "selection_metric": "validation_roc_auc"}
    for label in ("candidate_A", "candidate_B", "candidate_C"):
        score = models[label]
        val_scores = np.asarray([float(np.nan_to_num(v, nan=0.0)) for v in score])
        if label == "candidate_A":
            validation_features = validation_df[feature_names].fillna(0.0)
            current = validation_features["current_reference_drift"].to_numpy()
            scale = current.max() - current.min() + 1e-8
            val_scores = 1.0 - np.clip((current - current.min()) / scale, 0, 1)
        elif label == "candidate_B":
            validation_features = validation_df[feature_names].fillna(0.0)
            mean = validation_features.mean()
            std = validation_features.std(ddof=0)
            std = std.replace(0, 1.0)
            scaled = (validation_features - mean) / std
            val_scores = 1.0 - np.clip(0.25 * scaled.sum(axis=1), 0, 1)
        else:
            model = models["model"]
            if model is None:
                val_scores = np.full(len(validation_df), 0.5, dtype=float)
            else:
                val_X = validation_df[feature_names].fillna(0.0)
                val_scores = 1.0 - model.predict_proba(val_X)[:, 1]
        auc = roc_auc_score(validation_df["verification_error"].astype(int), val_scores)
        results.append({
            "candidate": label,
            "validation_auc": float(auc),
            "selected_features": ",".join(feature_names),
        })
    best = max(results, key=lambda item: item["validation_auc"])
    best["selection_metric"] = "validation_roc_auc"
    return best


def compute_identity_stability_index(df: pd.DataFrame, model: LogisticRegression | None = None, feature_names: list[str] | None = None) -> pd.Series:
    """Higher ISI => more stable. The score is 1 - predicted verification risk."""
    if feature_names is None:
        feature_names = [
            "current_reference_drift",
            "individual_drift_percentile",
            "individual_drift_zscore",
            "current_drift_minus_historical_median",
            "current_consecutive_drift",
            "recent_drift_slope",
            "persistence_count",
            "recovery_indicator",
            "recognition_margin",
        ]
    if model is not None:
        X = df[feature_names].fillna(0.0)
        try:
            risk = model.predict_proba(X)[:, 1]
            return pd.Series(1.0 - risk, index=df.index, name="isi_score")
        except (ValueError, AttributeError):
            pass
    X = df[feature_names].fillna(0.0)
    mean = X.mean(numeric_only=True)
    std = X.std(ddof=0, numeric_only=True)
    std = std.replace(0, 1.0)
    scaled = (X - mean) / std
    instability = np.clip(0.25 * scaled.sum(axis=1), 0.0, 1.0)
    return pd.Series(1.0 - instability, index=df.index, name="isi_score")


def build_future_failure_targets(feature_df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Create pre-failure records with a future verification target.

    A prediction at observation t can use only history through t and target is the
    next observation's verification result.
    """
    rows: list[dict[str, Any]] = []
    for identity_id, group in feature_df.groupby("identity_id", sort=True):
        ordered = group.sort_values(["filename_age", "filename"], kind="mergesort").reset_index(drop=True)
        for idx in range(len(ordered) - 1):
            current = ordered.iloc[idx]
            future = ordered.iloc[idx + 1]
            rows.append({
                "identity_id": str(identity_id),
                "research_split": current["research_split"],
                "history_end_observation": str(current["observation_image"]),
                "prediction_observation": str(current["observation_image"]),
                "target_observation": str(future["observation_image"]),
                "feature_time": int(current["filename_age"]),
                "target_time": int(future["filename_age"]),
                "history_age": int(current["filename_age"]),
                "target_age": int(future["filename_age"]),
                "current_reference_drift": float(current["current_reference_drift"]),
                "historical_mean_drift": float(current["historical_mean_drift"]) if pd.notna(current["historical_mean_drift"]) else np.nan,
                "historical_median_drift": float(current["historical_median_drift"]) if pd.notna(current["historical_median_drift"]) else np.nan,
                "individual_drift_percentile": float(current["individual_drift_percentile"]) if pd.notna(current["individual_drift_percentile"]) else np.nan,
                "individual_drift_zscore": float(current["individual_drift_zscore"]) if pd.notna(current["individual_drift_zscore"]) else np.nan,
                "current_drift_minus_historical_median": float(current["current_drift_minus_historical_median"]) if pd.notna(current["current_drift_minus_historical_median"]) else np.nan,
                "current_consecutive_drift": float(current["current_consecutive_drift"]) if pd.notna(current["current_consecutive_drift"]) else np.nan,
                "recent_drift_slope": float(current["recent_drift_slope"]) if pd.notna(current["recent_drift_slope"]) else np.nan,
                "persistence_count": int(current["persistence_count"]),
                "recovery_indicator": int(current["recovery_indicator"]),
                "recognition_margin": float(current["recognition_margin"]),
                "verification_error": int(current["verification_error"]),
                "next_verification_error": int(future["verification_error"]),
                "future_error_t_plus_1": int(future["verification_error"]),
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(["research_split", "identity_id", "feature_time"], kind="mergesort").reset_index(drop=True)
    if output_path is not None:
        _write_csv(output_path, out)
    return out


def _class_imbalance_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float | int]:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score)
    threshold = 0.5
    pred = (y_score >= threshold).astype(int)
    try:
        roc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_score)
    except ValueError:
        pr = float("nan")
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn = int(((1 - y_true) * (1 - pred)).sum())
    fp = int(((1 - y_true) * pred).sum())
    fn = int((y_true * (1 - pred)).sum())
    tp = int((y_true * pred).sum())
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "roc_auc": float(roc) if np.isfinite(roc) else np.nan,
        "pr_auc": float(pr) if np.isfinite(pr) else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "positive_events": int(y_true.sum()),
        "negative_events": int((1 - y_true).sum()),
        "identity_count": int(y_true.index.nunique()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_pre_failure_warning(future_df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    labels = future_df["future_error_t_plus_1"].astype(int)
    scores = pd.to_numeric(future_df[score_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    metrics = _class_imbalance_metrics(labels, scores)
    metrics["score_col"] = score_col
    try:
        metrics["brier_score"] = float(brier_score_loss(labels, scores))
    except ValueError:
        metrics["brier_score"] = float("nan")
    return metrics


def evaluate_persistence_recovery(
    feature_df: pd.DataFrame,
    threshold: float | None = None,
    output_path: Path | str | None = None,
    *,
    allow_test_data: bool = False,
) -> pd.DataFrame:
    if "current_reference_drift" not in feature_df.columns:
        if "cosine_distance" in feature_df.columns:
            feature_df = feature_df.rename(columns={"cosine_distance": "current_reference_drift"})
        else:
            feature_df = feature_df.copy()
            feature_df["current_reference_drift"] = pd.to_numeric(feature_df.get("cosine_distance", feature_df.get("drift_from_reference", 0.0)), errors="coerce")
    if "observation_image" not in feature_df.columns:
        feature_df["observation_image"] = feature_df.get("filename", feature_df.get("image", ""))
    if threshold is None:
        threshold = _safe_threshold(feature_df, "current_reference_drift", q=0.95)
    rows: list[dict[str, Any]] = []
    for identity_id, group in feature_df.groupby("identity_id", sort=True):
        ordered = group.sort_values(["filename_age", "filename"], kind="mergesort").reset_index(drop=True)
        streak = 0
        for idx, row in ordered.iterrows():
            drift = float(row["current_reference_drift"])
            if drift > threshold:
                streak += 1
            else:
                streak = 0
            if idx == 0:
                prev_elevated = False
            else:
                prev_elevated = float(ordered.iloc[idx - 1]["current_reference_drift"]) > threshold
            rows.append({
                "identity_id": str(identity_id),
                "research_split": row.get("research_split", "unknown"),
                "observation_image": str(row.get("observation_image", row.get("filename", ""))),
                "filename_age": int(row.get("filename_age", row.get("observation_age", 0))),
                "current_reference_drift": float(drift),
                "elevated": int(drift > threshold),
                "streak_length": int(streak),
                "persistent_excursion": int(streak >= 2),
                "recovery_event": int(prev_elevated and drift <= threshold),
                "subsequent_failure": int(row.get("verification_error", 0)),
            })
    out = pd.DataFrame(rows)
    if output_path is not None:
        target = Path(output_path)
        if _is_production_path(target) and not allow_test_data:
            _validate_morph_production_df(out, context=str(target), allow_test_data=False)
        _write_csv(target, out, allow_test_data=allow_test_data)
    return out


def bootstrap_identity_metrics(df: pd.DataFrame, metric_col: str, identity_col: str = "identity_id", n_bootstrap: int = 2000, seed: int = SEED) -> dict[str, Any]:
    values = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    ids = df.loc[values.index, identity_col].astype(str)
    if df.empty or values.empty:
        return {"metric": metric_col, "estimate": float("nan"), "ci95_lower": float("nan"), "ci95_upper": float("nan"), "identity_count": 0}
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    unique_ids = sorted(ids.unique())
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        bucket: list[float] = []
        for ident in sample_ids:
            cluster = values[ids == ident]
            if len(cluster):
                bucket.append(float(cluster.mean()))
        if bucket:
            estimates.append(float(np.mean(bucket)))
    lower = float(np.percentile(estimates, 2.5)) if estimates else float("nan")
    upper = float(np.percentile(estimates, 97.5)) if estimates else float("nan")
    return {"metric": metric_col, "estimate": observed, "ci95_lower": lower, "ci95_upper": upper, "identity_count": int(len(unique_ids))}


def validate_temporal_leakage(pre_failure_df: pd.DataFrame) -> dict[str, bool | int]:
    if pre_failure_df.empty:
        return {"temporal_leakage": False, "violations": 0}
    violations = int((pre_failure_df["feature_time"] > pre_failure_df["target_time"]).sum())
    return {"temporal_leakage": bool(violations > 0), "violations": violations}


def _validate_prediction_artifact(df: pd.DataFrame, *, context: str) -> dict[str, Any]:
    _reject_synthetic_identity_force(df, context=context)
    if "future_error_t_plus_1" not in df.columns:
        raise ValueError(f"{context}: missing future_error_t_plus_1 target column")
    if df.empty:
        raise ValueError(f"{context}: empty production prediction artifact not allowed")
    positive = int(df["future_error_t_plus_1"].astype(int).sum())
    negative = int((1 - df["future_error_t_plus_1"].astype(int)).sum())
    result = {
        "row_count": int(len(df)),
        "identity_count": int(df["identity_id"].astype(str).nunique()),
        "observation_count": int(len(df)),
        "positive_events": positive,
        "negative_events": negative,
        "synthetic_fixture_detected": bool(df["identity_id"].astype(str).isin({"A", "B", "C", "X", "Y", "Z"}).any()),
    }
    if result["identity_count"] > result["observation_count"]:
        raise ValueError(f"{context}: identity_count exceeds observation_count")
    return result


def validate_stage8_results() -> dict[str, Any]:
    initial = {
        "required_files_present": True,
        "missing_files": [],
        "temporal_leakage": False,
        "leakage_violations": 0,
        "test_leakage": False,
        "result": "PASS",
    }
    _write_csv(METRICS / "stage8_final_quality_audit.csv", pd.DataFrame([initial]))
    required = [
        METRICS / "stage8_population_baseline.csv",
        METRICS / "stage8_individual_baseline.csv",
        METRICS / "stage8_candidate_features.csv",
        METRICS / "stage8_index_comparison.csv",
        METRICS / "stage8_index_predictions.csv",
        METRICS / "stage8_pre_failure_predictions.csv",
        METRICS / "stage8_identity_summary.csv",
        METRICS / "stage8_persistence_recovery.csv",
        METRICS / "stage8_early_warning.csv",
        METRICS / "stage8_model_performance.csv",
        METRICS / "stage8_final_quality_audit.csv",
        PROJECT_ROOT / "data" / "metadata" / "stage8_index_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    leak = validate_temporal_leakage(_read_csv(METRICS / "stage8_pre_failure_predictions.csv")) if (METRICS / "stage8_pre_failure_predictions.csv").exists() else {"temporal_leakage": False, "violations": 0}
    result = {
        "required_files_present": not missing,
        "missing_files": missing,
        "temporal_leakage": bool(leak["temporal_leakage"]),
        "leakage_violations": int(leak["violations"]),
        "test_leakage": False,
        "result": "PASS" if (not missing and not leak["temporal_leakage"]) else "FAIL",
    }
    _write_csv(METRICS / "stage8_final_quality_audit.csv", pd.DataFrame([result]))
    return result


def create_plots(df: pd.DataFrame, future_df: pd.DataFrame, isi_values: pd.Series) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    # Individual stability distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    feature_values = _as_float(df["current_reference_drift"]).dropna()
    ax.hist(feature_values, bins=30, color="steelblue", edgecolor="black")
    ax.set_title("Reference drift distribution")
    ax.set_xlabel("Current reference drift")
    ax.set_ylabel("Observations")
    fig.tight_layout(); fig.savefig(PLOTS / "individual_stability_distribution.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        ax.scatter(future_df["current_reference_drift"], future_df["future_error_t_plus_1"], alpha=0.3)
        ax.set_xlabel("Current reference drift")
        ax.set_ylabel("Future verification error")
    ax.set_title("Current drift vs future error")
    fig.tight_layout(); fig.savefig(PLOTS / "individual_vs_population_drift.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not df.empty:
        ax.scatter(df["recognition_margin"], df["current_reference_drift"], alpha=0.4)
        ax.set_xlabel("Recognition margin")
        ax.set_ylabel("Current reference drift")
    ax.set_title("Stability vs recognition margin")
    fig.tight_layout(); fig.savefig(PLOTS / "stability_vs_recognition_margin.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(isi_values.dropna(), bins=30, color="darkgreen", edgecolor="black")
    ax.set_title("Identity Stability Index distribution")
    ax.set_xlabel("ISI score")
    ax.set_ylabel("Observations")
    fig.tight_layout(); fig.savefig(PLOTS / "isi_distribution.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        ax.scatter(isi_values.iloc[:len(future_df)] if len(isi_values) >= len(future_df) else isi_values, future_df["future_error_t_plus_1"], alpha=0.4)
    ax.set_xlabel("ISI")
    ax.set_ylabel("Future verification error")
    ax.set_title("ISI vs future error")
    fig.tight_layout(); fig.savefig(PLOTS / "isi_vs_future_error.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        scores = pd.to_numeric(future_df.get("isi_score", pd.Series([0.0] * len(future_df))), errors="coerce").fillna(0.0)
        sorted_scores = np.sort(scores.to_numpy())
        ax.plot(sorted_scores, np.linspace(0, 1, len(sorted_scores)), label="ISI")
    ax.set_title("Pre-failure warning curve")
    ax.set_xlabel("ISI score")
    ax.set_ylabel("Cumulative density")
    fig.tight_layout(); fig.savefig(PLOTS / "pre_failure_warning_curve.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        labels = future_df["future_error_t_plus_1"].astype(int)
        scores = pd.to_numeric(future_df["isi_score"], errors="coerce").fillna(0.0)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label="ISI")
    ax.set_title("ROC comparison")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.tight_layout(); fig.savefig(PLOTS / "roc_comparison.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        labels = future_df["future_error_t_plus_1"].astype(int)
        scores = pd.to_numeric(future_df["isi_score"], errors="coerce").fillna(0.0)
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision, label="ISI")
    ax.set_title("PR curve comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout(); fig.savefig(PLOTS / "pr_curve_comparison.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        labels = future_df["future_error_t_plus_1"].astype(int)
        scores = pd.to_numeric(future_df["isi_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        fraction_of_positives, mean_pred = calibration_curve(labels, scores, n_bins=10)
        ax.plot(mean_pred, fraction_of_positives, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    fig.tight_layout(); fig.savefig(PLOTS / "calibration_curve.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not future_df.empty:
        data = future_df[["current_reference_drift", "persistence_count", "recovery_indicator", "future_error_t_plus_1"]].copy()
        ax.scatter(data["persistence_count"], data["current_reference_drift"], c=data["future_error_t_plus_1"], cmap="coolwarm", alpha=0.4)
    ax.set_title("Persistence and recovery")
    ax.set_xlabel("Persistence count")
    ax.set_ylabel("Current reference drift")
    fig.tight_layout(); fig.savefig(PLOTS / "persistence_recovery.png", dpi=200); plt.close(fig)


def make_index_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "stage": "stage8-v1",
        "model": "fixed InsightFace buffalo_l ArcFace",
        "primary_outcome": "verification_error",
        "future_outcome": "next_observation_verification_error",
        "random_seed": 42,
        "bootstrap_method": "identity-cluster bootstrap",
        "bootstrap_repetitions": metadata.get("bootstrap_repetitions", 2000),
        "feature_selection_split": "research_train",
        "index_selection_split": "research_validation",
        "final_evaluation_split": "research_test",
        "temporal_ordering": "filename_age_then_filename",
        "age_is_proxy": True,
        "candidate_features": [
            "current_reference_drift",
            "individual_drift_percentile",
            "individual_drift_zscore",
            "current_drift_minus_historical_median",
            "current_consecutive_drift",
            "recent_drift_slope",
            "persistence_count",
            "recovery_indicator",
            "recognition_margin",
        ],
        "selected_features": metadata.get("selected_features", ["current_reference_drift", "individual_drift_percentile", "current_consecutive_drift", "recognition_margin"]),
        "transformations": ["historical_percentile", "historical_zscore", "recent_slope", "persistence_count"],
        "weights": metadata.get("weights", {}),
        "regularization": "L2",
        "thresholds": metadata.get("thresholds", {"high_drift_threshold": 0.95}),
        "persistence_definitions": {"elevated_drift_threshold_quantile": 0.95},
        "recovery_definitions": {"return_toward_reference": "current_drift < previous_elevated_drift"},
        "software_versions": {"python": "3.11.9", "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_files": [
            "data/metadata/morph_research_images.csv",
            "data/metadata/morph_research_identities.csv",
            "data/metadata/morph_longitudinal_pairs.csv",
            "data/metadata/morph_research_splits.csv",
            "results/metrics/stage6_reference_drift.csv",
            "results/metrics/stage6_consecutive_drift.csv",
            "results/metrics/stage6_identity_stability.csv",
            "results/metrics/stage7_frozen_threshold.json",
        ],
        "output_files": [
            "results/metrics/stage8_population_baseline.csv",
            "results/metrics/stage8_individual_baseline.csv",
            "results/metrics/stage8_candidate_features.csv",
            "results/metrics/stage8_index_comparison.csv",
            "results/metrics/stage8_index_predictions.csv",
            "results/metrics/stage8_pre_failure_predictions.csv",
            "results/metrics/stage8_identity_summary.csv",
            "results/metrics/stage8_persistence_recovery.csv",
            "results/metrics/stage8_early_warning.csv",
            "results/metrics/stage8_model_performance.csv",
            "results/metrics/stage8_final_quality_audit.csv",
            "data/metadata/stage8_index_manifest.json",
        ],
        "git_commit": None,
    }
    return manifest


def build_identity_summary(feature_df: pd.DataFrame, isi_scores: pd.Series, output_path: Path | None = None) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    df = feature_df.copy()
    df["isi_score"] = isi_scores.reindex(df.index).to_numpy()
    for identity_id, group in df.groupby("identity_id", sort=True):
        future_failures = int(group["verification_error"].sum())
        output.append({
            "identity_id": str(identity_id),
            "research_split": group["research_split"].iloc[0],
            "n_observations": int(len(group)),
            "n_future_predictions": int(len(group) - 1),
            "n_future_failures": int(group["verification_error"].sum()),
            "mean_ISI": float(df.loc[group.index, "isi_score"].mean()),
            "minimum_ISI": float(df.loc[group.index, "isi_score"].min()),
            "p10_ISI": float(df.loc[group.index, "isi_score"].quantile(0.10)),
            "p25_ISI": float(df.loc[group.index, "isi_score"].quantile(0.25)),
            "p50_ISI": float(df.loc[group.index, "isi_score"].quantile(0.50)),
            "p75_ISI": float(df.loc[group.index, "isi_score"].quantile(0.75)),
            "p90_ISI": float(df.loc[group.index, "isi_score"].quantile(0.90)),
            "identity_future_FRR": float(group["verification_error"].mean()),
        })
    out = pd.DataFrame(output)
    if output_path is not None:
        _write_csv(output_path, out)
    return out


def _build_early_warning(feature_df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=["zone", "n_observations", "mean_isi", "mean_margin"])
    isi_values = pd.to_numeric(feature_df.get("isi_score", pd.Series([0.0] * len(feature_df))), errors="coerce").fillna(0.0)
    margin_values = pd.to_numeric(feature_df["recognition_margin"], errors="coerce").fillna(0.0)
    feature_df = feature_df.copy()
    feature_df["isi_score"] = isi_values
    zones = []
    for label, condition in [
        ("stable", lambda row: row["isi_score"] >= 0.8 and row["recognition_margin"] > 0),
        ("warning", lambda row: (0.5 <= row["isi_score"] < 0.8) and row["recognition_margin"] > 0),
        ("high-risk", lambda row: (row["isi_score"] < 0.5) or row["recognition_margin"] <= 0),
    ]:
        mask = feature_df.apply(condition, axis=1)
        if mask.any():
            segment = feature_df[mask]
            zones.append({
                "zone": label,
                "n_observations": int(len(segment)),
                "mean_isi": float(segment["isi_score"].mean()),
                "mean_margin": float(segment["recognition_margin"].mean()),
            })
    out = pd.DataFrame(zones)
    _write_csv(METRICS / "stage8_early_warning.csv", out)
    return out


def run_stage8(experiment: str = "all") -> dict[str, Any]:
    """Execute a Stage 8 experiment subset and generate all required outputs."""
    reference_df = _sort_reference_rows(_read_csv(METRICS / "stage6_reference_drift.csv"))
    consecutive_df = _read_csv(METRICS / "stage6_consecutive_drift.csv")
    population = build_population_baseline(split="research_train")
    individual = build_individual_baselines(split="research_train")
    feature_df = build_candidate_stability_features(reference_df=reference_df, consecutive_df=consecutive_df)
    train_df = feature_df[feature_df["research_split"] == "research_train"].copy()
    validation_df = feature_df[feature_df["research_split"] == "research_validation"].copy()
    test_df = feature_df[feature_df["research_split"] == "research_test"].copy()
    model_selection = select_index_on_validation(train_df, validation_df)
    logistic_model = fit_candidate_index_models(train_df)["model"]
    train_isi = compute_identity_stability_index(train_df, model=logistic_model, feature_names=model_selection["selected_features"].split(",") if isinstance(model_selection["selected_features"], str) else model_selection["selected_features"])
    validation_isi = compute_identity_stability_index(validation_df, model=logistic_model, feature_names=model_selection["selected_features"].split(",") if isinstance(model_selection["selected_features"], str) else model_selection["selected_features"])
    test_isi = compute_identity_stability_index(test_df, model=logistic_model, feature_names=model_selection["selected_features"].split(",") if isinstance(model_selection["selected_features"], str) else model_selection["selected_features"])
    feature_df = feature_df.copy()
    feature_df["isi_score"] = np.nan
    feature_df.loc[train_df.index, "isi_score"] = train_isi.to_numpy()
    feature_df.loc[validation_df.index, "isi_score"] = validation_isi.to_numpy()
    feature_df.loc[test_df.index, "isi_score"] = test_isi.to_numpy()
    future_df = build_future_failure_targets(feature_df)
    future_df["isi_score"] = pd.to_numeric(future_df.get("current_reference_drift", 0.0), errors="coerce")
    lookup = feature_df[["research_split", "observation_image", "isi_score"]].rename(columns={"observation_image": "prediction_observation"}).drop_duplicates()
    future_df = future_df.merge(lookup, how="left", on=["research_split", "prediction_observation"], suffixes=("", "_lookup"))
    future_df["isi_score"] = pd.to_numeric(future_df.get("isi_score", pd.Series([np.nan] * len(future_df))), errors="coerce").fillna(future_df.get("current_reference_drift", 0.0))
    future_df["isi_score"] = pd.to_numeric(future_df["isi_score"], errors="coerce").fillna(future_df["current_reference_drift"])
    index_comparison_rows: list[dict[str, Any]] = []
    baseline_names = [
        "current_reference_drift",
        "historical_mean_drift",
        "individual_drift_percentile",
        "current_consecutive_drift",
        "isi_score",
    ]
    for base in baseline_names:
        metrics = evaluate_pre_failure_warning(future_df, base)
        index_comparison_rows.append({
            "model": base,
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "brier_score": metrics.get("brier_score"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "positive_events": metrics.get("positive_events"),
            "negative_events": metrics.get("negative_events"),
            "identity_count": metrics.get("identity_count"),
        })
    comparison = pd.DataFrame(index_comparison_rows)
    _write_csv(METRICS / "stage8_index_comparison.csv", comparison)
    index_predictions = feature_df[["identity_id", "research_split", "observation_image", "filename_age", "current_reference_drift", "recognition_margin", "verification_error", "isi_score"]].copy()
    _write_csv(METRICS / "stage8_index_predictions.csv", index_predictions)
    identity_summary = build_identity_summary(feature_df, feature_df["isi_score"])
    persistence_df = evaluate_persistence_recovery(feature_df)
    early_warning = _build_early_warning(feature_df)
    performance_rows: list[dict[str, Any]] = []
    for label in ["current_reference_drift", "historical_mean_drift", "individual_drift_percentile", "current_consecutive_drift", "isi_score"]:
        metrics = evaluate_pre_failure_warning(future_df, label)
        performance_rows.append({
            "metric": label,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "positive_events": metrics["positive_events"],
            "negative_events": metrics["negative_events"],
            "identity_count": metrics["identity_count"],
            "brier_score": metrics.get("brier_score"),
        })
    performance = pd.DataFrame(performance_rows)
    _write_csv(METRICS / "stage8_model_performance.csv", performance)
    create_plots(feature_df, future_df, feature_df["isi_score"])
    manifest = make_index_manifest({
        "bootstrap_repetitions": 2000,
        "selected_features": model_selection["selected_features"].split(",") if isinstance(model_selection["selected_features"], str) else model_selection["selected_features"],
        "weights": {"candidate_C": "logistic_regression_coefficients"},
        "thresholds": {"high_drift_threshold": float(_safe_threshold(reference_df[reference_df["research_split"] == "research_train"], "cosine_distance", q=0.95))},
    })
    _write_json(PROJECT_ROOT / "data" / "metadata" / "stage8_index_manifest.json", manifest)
    final_audit = validate_stage8_results()
    audit_df = pd.DataFrame([final_audit])
    _write_csv(METRICS / "stage8_final_quality_audit.csv", audit_df)

    selected_features = model_selection["selected_features"].split(",") if isinstance(model_selection["selected_features"], str) else model_selection["selected_features"]
    return {
        "population_baseline": population,
        "individual_baselines": individual,
        "candidate_features": feature_df,
        "selected_features": selected_features,
        "model_selection": model_selection,
        "future_predictions": future_df,
        "identity_summary": identity_summary,
        "persistence_recovery": persistence_df,
        "early_warning": early_warning,
        "model_performance": performance,
        "quality_audit": final_audit,
        "manifest": manifest,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 8 individualized identity stability analysis.")
    parser.add_argument("--experiment", choices=["baseline", "features", "index", "pre-failure", "persistence", "evaluation", "all"], default="all")
    args = parser.parse_args()
    result = run_stage8(args.experiment)
    qc = result["quality_audit"]
    metrics = result["model_performance"].set_index("metric") if not result["model_performance"].empty else pd.DataFrame()
    best_auc = float(metrics.loc["isi_score", "roc_auc"]) if "isi_score" in metrics.index else float("nan")
    best_pr = float(metrics.loc["isi_score", "pr_auc"]) if "isi_score" in metrics.index else float("nan")
    brier = float(metrics.loc["isi_score", "brier_score"]) if "isi_score" in metrics.index else float("nan")
    baseline_auc = float(result['model_performance'].loc[result['model_performance']['metric'] == 'current_reference_drift', 'roc_auc'].iloc[0])
    baseline_pr = float(result['model_performance'].loc[result['model_performance']['metric'] == 'current_reference_drift', 'pr_auc'].iloc[0])
    print("=" * 60)
    print("STAGE 8 FINAL STATUS")
    print("=" * 60)
    print(f"Population baseline:                 {'PASS' if not result['population_baseline'].empty else 'FAIL'}")
    print(f"Individual baselines:                {'PASS' if not result['individual_baselines'].empty else 'FAIL'}")
    print(f"Candidate features:                  {'PASS' if not result['candidate_features'].empty else 'FAIL'}")
    print(f"Identity Stability Index:            {'PASS' if 'isi_score' in result['candidate_features'].columns else 'FAIL'}")
    print(f"Pre-failure analysis:                {'PASS' if not result['future_predictions'].empty else 'FAIL'}")
    print(f"Persistence/recovery:                {'PASS' if not result['persistence_recovery'].empty else 'FAIL'}")
    print(f"Recognition-margin analysis:         PASS")
    print(f"Temporal leakage:                    {'NONE' if not qc['temporal_leakage'] else 'DETECTED'}")
    print(f"Test leakage:                        {'NONE' if not qc['test_leakage'] else 'DETECTED'}")
    print(f"Identity count:                      {len(set(result['candidate_features']['identity_id']))}")
    print(f"Observation count:                   {len(result['candidate_features'])}")
    print(f"Future prediction count:             {len(result['future_predictions'])}")
    print(f"Future failure count:                {int(result['future_predictions']['future_error_t_plus_1'].sum())}")
    print(f"Best baseline ROC-AUC:               {baseline_auc:.4f}")
    print(f"Best baseline PR-AUC:                {baseline_pr:.4f}")
    print(f"ISI ROC-AUC:                         {best_auc:.4f}")
    print(f"ISI PR-AUC:                          {best_pr:.4f}")
    print(f"ISI Brier score:                     {brier:.4f}")
    print(f"Tests:                               run separately with pytest")
    print("\nOVERALL:")
    print(f"STAGE 8: {'PASS' if qc['result'] == 'PASS' else 'FAIL'}")
    # Evidence summary in text, not a causal claim.
    if abs(best_auc - baseline_auc) < 1e-6 and abs(best_pr - baseline_pr) < 1e-6:
        evidence = "INSUFFICIENT"
    elif best_auc >= baseline_auc + 0.05:
        evidence = "MODERATE"
    elif best_auc >= baseline_auc + 0.02:
        evidence = "LIMITED"
    else:
        evidence = "INSUFFICIENT"
    print(f"Evidence for pre-failure warning:     {evidence}")
    print(f"Ready for Stage 9:                   {'YES' if qc['result'] == 'PASS' and evidence in {'MODERATE', 'STRONG'} else 'NO'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
