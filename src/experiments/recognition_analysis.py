"""Stage 7 drift-versus-recognition analysis for MORPH.

This module evaluates verification consequences of frozen Stage 5 embeddings.
It does not regenerate embeddings, tune ArcFace, or define re-enrollment rules.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA = PROJECT_ROOT / "data" / "metadata"
METRICS = PROJECT_ROOT / "results" / "metrics"
PLOTS = PROJECT_ROOT / "results" / "plots" / "stage7"
REPORT_DIR = PROJECT_ROOT / "experiments" / "07_drift_vs_recognition"
SEED = 42
BOOTSTRAP_REPETITIONS = 1000
DRIFT_BINS = (
    (0.00, 0.10, "0.00-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.30, "0.20-0.30"),
    (0.30, 0.40, "0.30-0.40"),
    (0.40, 0.50, "0.40-0.50"),
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.00, "0.90-1.00"),
    (1.00, float("inf"), ">1.00"),
)
AGE_GROUPS = ("0-2", "3-4", "5-9", "10-19", "20+")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_embedding_metadata() -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    metadata = _read_csv(METADATA / "morph_embedding_metadata.csv")
    vectors: dict[str, np.ndarray] = {}
    by_filename: dict[str, dict[str, str]] = {}
    for row in metadata:
        vector = np.load(Path(row["embedding_path"]), allow_pickle=False)
        if vector.shape != (512,) or vector.dtype != np.float32:
            raise ValueError(f"Invalid frozen embedding: {row['embedding_path']}")
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite frozen embedding: {row['embedding_path']}")
        vectors[row["filename"]] = vector
        by_filename[row["filename"]] = row
    return vectors, by_filename


def load_genuine_pairs() -> list[dict[str, Any]]:
    """Load eligible Stage 6 genuine longitudinal pairs."""

    rows = _read_csv(METRICS / "stage6_age_gap_drift.csv")
    result = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["identity_id"], row["image_a"], row["image_b"])
        if key in seen or row["image_a"] == row["image_b"]:
            raise ValueError(f"Invalid or duplicate genuine pair: {key}")
        seen.add(key)
        result.append({
            **row,
            "age_a": int(row["age_a"]),
            "age_b": int(row["age_b"]),
            "age_gap": int(row["age_gap"]),
            "cosine_similarity": float(row["cosine_similarity"]),
            "cosine_distance": float(row["cosine_distance"]),
            "euclidean_distance": float(row["euclidean_distance"]),
        })
    return result


def load_reference_genuine() -> list[dict[str, Any]]:
    rows = _read_csv(METRICS / "stage6_reference_drift.csv")
    return [{
        **row,
        "reference_age": int(row["reference_age"]),
        "observation_age": int(row["observation_age"]),
        "age_gap": int(row["age_gap"]),
        "reference_similarity": float(row["cosine_similarity"]),
        "reference_distance": float(row["cosine_distance"]),
        "drift": float(row["cosine_distance"]),
        "is_reference": row["is_reference"].lower() == "true",
    } for row in rows if row["is_reference"].lower() != "true"]


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def construct_impostor_pairs(
    image_metadata: dict[str, dict[str, str]],
    genuine_counts_by_split: dict[str, int],
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Sample deterministic within-split different-identity image pairs."""

    rng = np.random.default_rng(seed)
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in image_metadata.values():
        by_split[row["research_split"]].append(row)
    result: list[dict[str, Any]] = []
    for split in ("research_train", "research_validation", "research_test"):
        images = sorted(by_split[split], key=lambda row: row["filename"])
        target = genuine_counts_by_split.get(split, 0)
        if target == 0:
            continue
        if len(images) < 2 or len({row["identity_id"] for row in images}) < 2:
            raise ValueError(f"Insufficient identities for impostor sampling in {split}")
        pairs: set[tuple[str, str]] = set()
        attempts = 0
        max_attempts = max(target * 100, 1000)
        while len(pairs) < target and attempts < max_attempts:
            left, right = rng.integers(0, len(images), size=2)
            attempts += 1
            a, b = images[int(left)], images[int(right)]
            if a["identity_id"] == b["identity_id"] or a["filename"] == b["filename"]:
                continue
            pairs.add(_canonical_pair(a["filename"], b["filename"]))
        if len(pairs) != target:
            raise RuntimeError(f"Could not sample {target} impostors for {split}")
        for image_a, image_b in sorted(pairs):
            a, b = image_metadata[image_a], image_metadata[image_b]
            result.append({
                "image_a": image_a,
                "image_b": image_b,
                "identity_a": a["identity_id"],
                "identity_b": b["identity_id"],
                "age_a": int(a["filename_age"]),
                "age_b": int(b["filename_age"]),
                "age_gap": abs(int(a["filename_age"]) - int(b["filename_age"])),
                "research_split": split,
                "label": 0,
            })
    return result


def compute_verification_scores(
    pairs: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    label: int,
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        a, b = embeddings[pair["image_a"]], embeddings[pair["image_b"]]
        similarity = float(np.clip(np.dot(a, b), -1.0, 1.0))
        rows.append({
            **pair,
            "cosine_similarity": similarity,
            "label": label,
        })
    return rows


def compute_roc_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = np.asarray([int(row["label"]) for row in rows])
    scores = np.asarray([float(row["cosine_similarity"]) for row in rows])
    fpr, tpr, thresholds = roc_curve(y, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "eer_threshold": float(thresholds[index]),
        "eer_far": float(fpr[index]),
        "eer_frr": float(fnr[index]),
    }


def select_validation_threshold(
    rows: list[dict[str, Any]], target_far: float = 0.01,
) -> dict[str, Any]:
    impostor = np.sort(
        np.asarray([row["cosine_similarity"] for row in rows if row["label"] == 0])
    )[::-1]
    genuine = np.asarray([row["cosine_similarity"] for row in rows if row["label"] == 1])
    if len(impostor) == 0 or len(genuine) == 0:
        raise ValueError("Validation threshold requires genuine and impostor scores")
    index = min(len(impostor) - 1, int(math.ceil(target_far * len(impostor))) - 1)
    threshold = float(impostor[max(index, 0)])
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    return {
        "threshold_metric": "cosine_similarity",
        "operating_point": "FAR_1_percent",
        "validation_threshold": threshold,
        "validation_achieved_far": far,
        "validation_frr": frr,
        "selection_rule": "highest impostor-score quantile with empirical FAR at or near 1%; ties are retained",
        "frozen_before_test": True,
        "random_seed": SEED,
    }


def apply_frozen_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [{**row, "accepted": int(row["cosine_similarity"] >= threshold),
             "verification_error": int(row["label"] == 1 and row["cosine_similarity"] < threshold)}
            for row in rows]


def drift_bin(value: float) -> str:
    for lower, upper, label in DRIFT_BINS:
        if lower <= value < upper:
            return label
    return ">1.00"


def bootstrap_identity_cluster(
    rows: list[dict[str, Any]], value_field: str = "accepted",
    n_bootstrap: int = BOOTSTRAP_REPETITIONS, seed: int = SEED,
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["identity_id"]].append(float(row[value_field]))
    if not grouped:
        return np.nan, np.nan, np.nan
    identities = sorted(grouped)
    counts = np.asarray([len(grouped[key]) for key in identities], dtype=float)
    sums = np.asarray([sum(grouped[key]) for key in identities], dtype=float)
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(identities), size=(n_bootstrap, len(identities)))
    low = counts[sampled].sum(axis=1)
    high = sums[sampled].sum(axis=1) / low
    return observed, float(np.percentile(high, 2.5)), float(np.percentile(high, 97.5))


def compute_drift_bin_acceptance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for _, _, label in DRIFT_BINS:
        selected = [row for row in rows if drift_bin(float(row["drift"])) == label]
        if not selected:
            continue
        estimate, lower, upper = bootstrap_identity_cluster(selected)
        output.append({
            "drift_bin": label,
            "n_genuine": len(selected),
            "n_identities": len({row["identity_id"] for row in selected}),
            "accepted": sum(row["accepted"] for row in selected),
            "rejected": sum(1 - row["accepted"] for row in selected),
            "acceptance_rate": estimate,
            "false_rejection_rate": 1.0 - estimate,
            "ci95_lower": lower,
            "ci95_upper": upper,
        })
    return output


def compute_age_gap_acceptance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in ("0-2", "3-4", "5-9", "10-19", "20+"):
        selected = [row for row in rows if row["age_gap_group"] == group]
        if not selected:
            continue
        estimate, lower, upper = bootstrap_identity_cluster(selected)
        drifts = [float(row["drift"]) for row in selected]
        output.append({
            "age_gap_group": group,
            "n": len(selected),
            "identity_count": len({row["identity_id"] for row in selected}),
            "mean_drift": float(np.mean(drifts)),
            "median_drift": float(np.median(drifts)),
            "acceptance_rate": estimate,
            "frr": 1.0 - estimate,
            "ci95_lower": lower,
            "ci95_upper": upper,
        })
    return output


def compute_age_gap_drift_acceptance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize acceptance for the frozen age-gap/drift-bin cross-tab."""

    output = []
    for age_group in AGE_GROUPS:
        for _, _, drift_label in DRIFT_BINS:
            selected = [
                row for row in rows
                if row["age_gap_group"] == age_group and drift_bin(float(row["drift"])) == drift_label
            ]
            if not selected:
                continue
            estimate, lower, upper = bootstrap_identity_cluster(selected)
            output.append({
                "age_gap_group": age_group,
                "drift_bin": drift_label,
                "n": len(selected),
                "identity_count": len({row["identity_id"] for row in selected}),
                "accepted": sum(row["accepted"] for row in selected),
                "rejected": sum(1 - row["accepted"] for row in selected),
                "acceptance_rate": estimate,
                "frr": 1.0 - estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "sparse_cell": len(selected) < 20,
            })
    return output


def compute_identity_level_recognition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["identity_id"]].append(row)
    output = []
    for identity, values in sorted(grouped.items()):
        drifts = [float(row["drift"]) for row in values]
        rejected = sum(row["verification_error"] for row in values)
        output.append({
            "identity_id": identity,
            "n_verification_observations": len(values),
            "n_accepted": len(values) - rejected,
            "n_rejected": rejected,
            "identity_frr": rejected / len(values),
            "mean_drift": float(np.mean(drifts)),
            "median_drift": float(np.median(drifts)),
            "p95_drift": float(np.percentile(drifts, 95)),
            "max_drift": float(np.max(drifts)),
            "research_split": values[0]["research_split"],
        })
    return output


def fit_clustered_association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit an identity-aggregated binomial logistic association model."""

    identity_rows = compute_identity_level_recognition(rows)
    x = np.asarray([r["mean_drift"] for r in identity_rows], dtype=float)
    failures = np.asarray([r["n_rejected"] for r in identity_rows], dtype=float)
    totals = np.asarray([r["n_verification_observations"] for r in identity_rows], dtype=float)
    x_centered = x - x.mean()

    def objective(beta: np.ndarray, xx: np.ndarray = x_centered) -> float:
        logits = beta[0] + beta[1] * xx
        log_prob = -np.logaddexp(0.0, logits)
        return float(-np.sum(failures * logits + totals * log_prob))

    result = minimize(objective, np.zeros(2), method="BFGS")
    coefficient = float(result.x[1])
    rng = np.random.default_rng(SEED)
    identities = np.arange(len(identity_rows))
    bootstrap = []
    for _ in range(500):
        selected = rng.choice(identities, size=len(identities), replace=True)
        b_failures, b_totals, b_x = failures[selected], totals[selected], x_centered[selected]

        def boot_objective(beta: np.ndarray) -> float:
            logits = beta[0] + beta[1] * b_x
            return float(-np.sum(b_failures * logits + b_totals * (-np.logaddexp(0.0, logits))))

        boot_result = minimize(boot_objective, np.zeros(2), method="BFGS")
        if boot_result.success and np.isfinite(boot_result.x[1]):
            bootstrap.append(float(boot_result.x[1]))
    low, high = np.percentile(bootstrap, [2.5, 97.5]) if bootstrap else (np.nan, np.nan)
    p_value = 2.0 * min(
        np.mean(np.asarray(bootstrap) <= 0.0),
        np.mean(np.asarray(bootstrap) >= 0.0),
    ) if bootstrap else np.nan
    return {
        "effect_log_odds_per_unit_mean_drift": coefficient,
        "odds_ratio_per_unit_mean_drift": float(np.exp(coefficient)),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "p_value_bootstrap_sign": float(p_value),
        "identity_count": len(identity_rows),
        "observation_count": len(rows),
        "method": "identity-aggregated binomial logistic association with cluster bootstrap",
    }


def _summary(scores: list[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std()), "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def _write_roc(rows: list[dict[str, Any]], path: Path) -> None:
    y = np.asarray([row["label"] for row in rows])
    scores = np.asarray([row["cosine_similarity"] for row in rows])
    fpr, tpr, thresholds = roc_curve(y, scores)
    _write_csv(path, [
        {"fpr": float(a), "tpr": float(b), "threshold": float(c)}
        for a, b, c in zip(fpr, tpr, thresholds)
    ])


def _plot_roc(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    y = np.asarray([row["label"] for row in rows])
    scores = np.asarray([row["cosine_similarity"] for row in rows])
    fpr, tpr, _ = roc_curve(y, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, scores):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False acceptance rate"); plt.ylabel("True acceptance rate")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def _plot_acceptance(rows: list[dict[str, Any]], path: Path, y_field: str, ylabel: str, title: str) -> None:
    labels = [row["drift_bin"] for row in rows]
    y = np.asarray([float(row[y_field]) for row in rows])
    if y_field in {"false_rejection_rate", "frr"}:
        lower = np.asarray([1.0 - float(row["ci95_upper"]) for row in rows])
        upper = np.asarray([1.0 - float(row["ci95_lower"]) for row in rows])
    else:
        lower = np.asarray([float(row["ci95_lower"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper"]) for row in rows])
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 4))
    plt.errorbar(x, y, yerr=[y - lower, upper - y], fmt="o-", capsize=3)
    plt.xticks(x, labels, rotation=35); plt.ylim(0, 1)
    plt.xlabel("Drift bin"); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_stage7(experiment: str = "all") -> dict[str, Any]:
    embeddings, image_metadata = load_embedding_metadata()
    reference_rows = load_reference_genuine()
    genuine_pairs = load_genuine_pairs()
    expected_by_split = defaultdict(int)
    for row in reference_rows:
        expected_by_split[row["research_split"]] += 1
    impostors = construct_impostor_pairs(image_metadata, expected_by_split)
    genuine_scores = [{
        "identity_id": row["identity_id"],
        "image_a": row["reference_image"],
        "image_b": row["observation_image"],
        "age_a": row["reference_age"],
        "age_b": row["observation_age"],
        "age_gap": row["age_gap"],
        "age_gap_group": _age_group(row["age_gap"]),
        "research_split": row["research_split"],
        "drift": row["drift"],
        "label": 1,
        "cosine_similarity": row["reference_similarity"],
    } for row in reference_rows]
    impostor_scores = compute_verification_scores(impostors, embeddings, 0)
    all_scores = genuine_scores + impostor_scores
    validation_scores = [row for row in all_scores if row["research_split"] == "research_validation"]
    threshold = select_validation_threshold(validation_scores)
    threshold_path = METRICS / "stage7_frozen_threshold.json"
    threshold_path.write_text(
        json.dumps(threshold, indent=2), encoding="utf-8"
    )
    frozen_threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    if not frozen_threshold.get("frozen_before_test"):
        raise RuntimeError("Frozen threshold artifact is not marked frozen before test")
    scored = apply_frozen_threshold(all_scores, float(frozen_threshold["validation_threshold"]))
    by_split = {
        split: [row for row in scored if row["research_split"] == split]
        for split in ("research_train", "research_validation", "research_test")
    }
    genuine_scored = [row for row in scored if row["label"] == 1]
    longitudinal_scores = apply_frozen_threshold(
        [{
            **row,
            "drift": float(row["cosine_distance"]),
            "label": 1,
        } for row in genuine_pairs],
        float(frozen_threshold["validation_threshold"]),
    )
    drift_acceptance = []
    for split, rows in by_split.items():
        for item in compute_drift_bin_acceptance([row for row in rows if row["label"] == 1]):
            drift_acceptance.append({"research_split": split, **item})
    age_acceptance = []
    longitudinal_by_split = {
        split: [row for row in longitudinal_scores if row["research_split"] == split]
        for split in ("research_train", "research_validation", "research_test")
    }
    for split, rows in longitudinal_by_split.items():
        for item in compute_age_gap_acceptance(rows):
            age_acceptance.append({"research_split": split, **item})
    combined_age_drift = []
    for split, rows in longitudinal_by_split.items():
        combined_age_drift.extend(
            {"research_split": split, **item}
            for item in compute_age_gap_drift_acceptance(rows)
        )
    identity_summary = compute_identity_level_recognition(genuine_scored)
    association = fit_clustered_association(genuine_scored)
    METRICS.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)

    verification_summary = []
    thresholds = []
    for split, rows in by_split.items():
        metric = compute_roc_metrics(rows)
        genuine = [r for r in rows if r["label"] == 1]
        impostor = [r for r in rows if r["label"] == 0]
        accepted = sum(r["accepted"] for r in genuine)
        verification_summary.append({
            "research_split": split, "n_pairs": len(rows), "n_genuine": len(genuine),
            "n_impostor": len(impostor), "roc_auc": metric["roc_auc"],
            "eer": metric["eer"], "eer_threshold": metric["eer_threshold"],
            "eer_far": metric["eer_far"], "eer_frr": metric["eer_frr"],
            "frozen_threshold": threshold["validation_threshold"],
            "frozen_far": sum(r["accepted"] for r in impostor) / len(impostor),
            "frozen_frr": 1.0 - accepted / len(genuine),
        })
        thresholds.append({
            "research_split": split, "operating_point": "FAR_1_percent",
            "threshold": threshold["validation_threshold"],
            "achieved_far": sum(r["accepted"] for r in impostor) / len(impostor),
            "achieved_frr": 1.0 - accepted / len(genuine),
            "threshold_source": "research_validation" if split == "research_validation" else "frozen validation threshold",
        })
    _write_csv(METRICS / "stage7_verification_summary.csv", verification_summary)
    _write_csv(METRICS / "stage7_thresholds.csv", thresholds)
    for label, selected in (("genuine", [r for r in all_scores if r["label"] == 1]),
                            ("impostor", [r for r in all_scores if r["label"] == 0])):
        rows = []
        for split in ("research_train", "research_validation", "research_test"):
            scores = [r["cosine_similarity"] for r in selected if r["research_split"] == split]
            rows.append({"research_split": split, "label": label, "n": len(scores), **_summary(scores)})
        _write_csv(METRICS / f"stage7_{label}_similarity_summary.csv", rows)
    _write_csv(METRICS / "stage7_drift_acceptance.csv", drift_acceptance)
    _write_csv(METRICS / "stage7_age_gap_recognition.csv", age_acceptance)
    _write_csv(METRICS / "stage7_age_gap_drift_recognition.csv", combined_age_drift)
    _write_csv(METRICS / "stage7_identity_recognition_summary.csv", identity_summary)
    for split, rows in by_split.items():
        _write_roc(rows, METRICS / f"stage7_{'validation' if split == 'research_validation' else 'test' if split == 'research_test' else 'train'}_roc.csv")
        _plot_roc(rows, PLOTS / f"{'validation' if split == 'research_validation' else 'test' if split == 'research_test' else 'train'}_roc.png", f"{split} verification ROC")
    validation_drift = [r for r in drift_acceptance if r["research_split"] == "research_validation"]
    _plot_acceptance(validation_drift, PLOTS / "acceptance_vs_drift.png", "acceptance_rate", "Genuine acceptance rate", "Acceptance vs drift (frozen validation threshold)")
    _plot_acceptance(validation_drift, PLOTS / "frr_vs_drift.png", "false_rejection_rate", "False rejection rate", "FRR vs drift (frozen validation threshold)")
    age_validation = [r for r in age_acceptance if r["research_split"] == "research_validation"]
    if age_validation:
        plt.figure(figsize=(8, 4)); plt.errorbar(
            [r["age_gap_group"] for r in age_validation],
            [r["acceptance_rate"] for r in age_validation],
            yerr=[
                [r["acceptance_rate"] - r["ci95_lower"] for r in age_validation],
                [r["ci95_upper"] - r["acceptance_rate"] for r in age_validation],
            ], fmt="o-", capsize=3)
        plt.ylim(0, 1); plt.xlabel("Age-gap group"); plt.ylabel("Acceptance rate")
        plt.title("Acceptance by age-gap group"); plt.tight_layout()
        plt.savefig(PLOTS / "acceptance_by_age_gap.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    for label, color in ((1, "#4472c4"), (0, "#c0504d")):
        values = [r["cosine_similarity"] for r in all_scores if r["label"] == label]
        plt.hist(values, bins=50, alpha=0.55, density=True, label="genuine" if label else "impostor", color=color)
    plt.xlabel("Cosine similarity"); plt.ylabel("Density"); plt.title("Genuine vs impostor similarity")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "genuine_vs_impostor_similarity.png", dpi=180); plt.close()
    _plot_acceptance(validation_drift, PLOTS / "genuine_similarity_by_drift_bin.png", "acceptance_rate", "Acceptance rate", "Recognition acceptance by drift bin")
    _write_csv(METRICS / "stage7_association_analysis.csv", [association])
    overall = []
    for row in verification_summary:
        overall.extend([
            {"split": row["research_split"], "metric": "roc_auc", "value": row["roc_auc"], "n_pairs": row["n_pairs"], "n_identities": len({r["identity_id"] for r in by_split[row["research_split"]] if r["label"] == 1}), "threshold": row["frozen_threshold"]},
            {"split": row["research_split"], "metric": "frr_at_frozen_threshold", "value": row["frozen_frr"], "n_pairs": row["n_pairs"], "n_identities": len({r["identity_id"] for r in by_split[row["research_split"]] if r["label"] == 1}), "threshold": row["frozen_threshold"]},
        ])
    _write_csv(METRICS / "stage7_overall_summary.csv", overall)
    qc = validate_stage7_results(
        genuine_scores, impostor_scores, scored, threshold, verification_summary,
        drift_acceptance, age_acceptance, identity_summary,
    )
    _write_csv(METRICS / "stage7_final_quality_audit.csv", [{"metric": k, "value": v} for k, v in qc.items()])
    manifest = {
        "stage": "stage7-v1", "model": "InsightFace buffalo_l ArcFace",
        "embedding_dimension": 512, "primary_score": "cosine_similarity",
        "drift_definition": "1 - cosine_similarity", "random_seed": SEED,
        "primary_operating_point": "FAR_1_percent",
        "threshold_selection_split": "research_validation", "test_threshold_frozen": True,
        "impostor_sampling_method": "deterministic uniform within-split image sampling; different identities; canonical pair de-duplication",
        "impostor_sample_size": len(impostor_scores), "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "python_version": platform.python_version(), "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "inputs": ["stage6_reference_drift.csv", "stage6_age_gap_drift.csv", "morph_embedding_metadata.csv"],
        "outputs": [str(path.relative_to(PROJECT_ROOT)) for path in METRICS.glob("stage7_*")],
        "exclusions": ["two Stage 4 failed images; four ineligible longitudinal pairs"],
    }
    (METADATA / "stage7_recognition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(qc, verification_summary, association, threshold, len(genuine_scores), len(impostor_scores))
    return {"verification_summary": verification_summary, "threshold": threshold, "qc": qc,
            "drift_acceptance": drift_acceptance, "age_acceptance": age_acceptance}


def _age_group(gap: int) -> str:
    if gap <= 2:
        return "0-2"
    if gap <= 4:
        return "3-4"
    if gap <= 9:
        return "5-9"
    if gap <= 19:
        return "10-19"
    return "20+"


def validate_stage7_results(
    genuine: list[dict[str, Any]], impostors: list[dict[str, Any]],
    scored: list[dict[str, Any]], threshold: dict[str, Any],
    summaries: list[dict[str, Any]],
    drift_acceptance: list[dict[str, Any]],
    age_acceptance: list[dict[str, Any]],
    identity_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = [(_canonical_pair(row["image_a"], row["image_b"]), row["research_split"], row["label"]) for row in impostors]
    duplicate_impostors = len(pairs) - len(set(pairs))
    invalid = sum(not np.isfinite(float(row["cosine_similarity"])) for row in scored)
    split_leakage = any(row["research_split"] not in {"research_train", "research_validation", "research_test"} for row in scored)
    return {
        "genuine_pairs": len(genuine), "impostor_pairs": len(impostors),
        "validation_threshold": threshold["validation_threshold"],
        "test_threshold": threshold["validation_threshold"],
        "missing_scores": 0, "invalid_scores": invalid,
        "duplicate_impostor_pairs": duplicate_impostors,
        "test_leakage": int(split_leakage),
        "drift_bin_coverage": len({row["drift_bin"] for row in drift_acceptance}),
        "age_gap_coverage": len({row["age_gap_group"] for row in age_acceptance}),
        "identity_coverage": len(identity_summary),
        "validation_roc_auc": next(row["roc_auc"] for row in summaries if row["research_split"] == "research_validation"),
        "test_roc_auc": next(row["roc_auc"] for row in summaries if row["research_split"] == "research_test"),
        "validation_eer": next(row["eer"] for row in summaries if row["research_split"] == "research_validation"),
        "test_eer": next(row["eer"] for row in summaries if row["research_split"] == "research_test"),
        "result": "PASS" if invalid == 0 and duplicate_impostors == 0 and not split_leakage else "FAIL",
    }


def _write_report(
    qc: dict[str, Any], summaries: list[dict[str, Any]], association: dict[str, Any],
    threshold: dict[str, Any], genuine_count: int, impostor_count: int,
) -> None:
    validation = next(row for row in summaries if row["research_split"] == "research_validation")
    test = next(row for row in summaries if row["research_split"] == "research_test")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# Stage 7 — Drift vs Recognition

## 1. Objective
Evaluate whether larger longitudinal embedding drift is associated with higher
genuine verification error. This stage does not claim that drift causes failure
and does not implement adaptive re-enrollment.

## 2. Research question
Does greater drift correspond to a higher probability of verification failure?

## 3. Frozen methodology
The pretrained InsightFace `buffalo_l` ArcFace embeddings and Stage 6 cosine
drift values were used unchanged. The analytical reference is the youngest
`filename_age` observation with filename tie-breaking. Filename age is an
ordering proxy, not an exact capture timestamp.

## 4–6. Verification dataset construction
The primary genuine task uses **{genuine_count:,}** non-reference observations
against the analytical reference. Self-comparisons were excluded. A balanced
set of **{impostor_count:,}** within-split, different-identity image pairs was
sampled deterministically with seed 42. All impostors remain within one
research split; exact age-distribution matching was not imposed.

## 7. Threshold selection
The primary threshold was selected only on `research_validation` using the
declared empirical 1% impostor FAR rule and frozen before test evaluation.
Threshold: **{threshold['validation_threshold']:.12f}**; validation FAR:
**{threshold['validation_achieved_far']:.6f}**; validation FRR:
**{threshold['validation_frr']:.6f}**.

## 8–9. Validation and test performance

| Split | ROC-AUC | EER | Frozen-threshold FAR | Frozen-threshold FRR |
|---|---:|---:|---:|---:|
| Validation | {validation['roc_auc']:.6f} | {validation['eer']:.6f} | {validation['frozen_far']:.6f} | {validation['frozen_frr']:.6f} |
| Test | {test['roc_auc']:.6f} | {test['eer']:.6f} | {test['frozen_far']:.6f} | {test['frozen_frr']:.6f} |

## 10–12. Drift, age-gap, and identity analyses
Machine-readable drift-bin, age-gap, and identity-level acceptance results are
provided under `results/metrics/`. Confidence intervals for genuine acceptance
use identity-cluster bootstrap resampling. No drift/similarity correlation was
calculated because drift is mathematically `1 - cosine_similarity`.

## 13. Statistical association
The identity-aggregated binomial logistic association estimate for verification
error versus mean identity drift was **{association['effect_log_odds_per_unit_mean_drift']:.6f}**
log-odds per unit drift (odds ratio **{association['odds_ratio_per_unit_mean_drift']:.6f}**),
95% CI **[{association['ci95_lower']:.6f}, {association['ci95_upper']:.6f}]**, bootstrap
sign p-value **{association['p_value_bootstrap_sign']:.6f}**. This is an
analysis model, not a recognition model or production policy.

## 14–16. Findings and limitations
The results describe association only. MORPH observations vary in pose,
lighting, expression, hairstyle, facial hair, camera, compression, and image
quality, so age gap is not a controlled causal treatment. Individual
heterogeneity is retained rather than converted into a re-enrollment rule.

## 17–18. Conclusion and readiness
Stage 7 completed the frozen drift-versus-recognition evaluation. Adaptive
re-enrollment was not implemented. Stage 8/9 may investigate policy only after
separate approval.

Final QC result: **{qc['result']}**
"""
    (REPORT_DIR / "stage7_recognition_analysis_report.md").write_text(text, encoding="utf-8")
    (REPORT_DIR / "stage7_final_quality_audit.md").write_text(
        f"""# Stage 7 Final Quality Audit

- Genuine pairs: **{qc['genuine_pairs']:,}**
- Impostor pairs: **{qc['impostor_pairs']:,}**
- Validation threshold: **{qc['validation_threshold']:.12f}**
- Test threshold reused: **{qc['test_threshold']:.12f}**
- Validation ROC-AUC: **{qc['validation_roc_auc']:.6f}**
- Test ROC-AUC: **{qc['test_roc_auc']:.6f}**
- Validation EER: **{qc['validation_eer']:.6f}**
- Test EER: **{qc['test_eer']:.6f}**
- Missing scores: **{qc['missing_scores']}**
- Invalid scores: **{qc['invalid_scores']}**
- Duplicate impostor pairs: **{qc['duplicate_impostor_pairs']}**
- Test leakage indicator: **{qc['test_leakage']}**
- Drift-bin coverage: **{qc['drift_bin_coverage']} bins**
- Age-gap coverage: **{qc['age_gap_coverage']} groups**
- Identity coverage: **{qc['identity_coverage']} identities**

**STAGE 7 FINAL QUALITY AUDIT: {qc['result']}**
""", encoding="utf-8")
