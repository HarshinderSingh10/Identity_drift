"""Verification, threshold, drift-bin, statistical, and data-quality analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import auc, roc_auc_score, roc_curve

from src.data import load_dataset_pairs
from src.drift.drift_metrics import cosine_similarity, euclidean_distance
from src.experiments.drift_experiment import (
    DEFAULT_DATASETS,
    RESULTS_ROOT,
    extract_embeddings_for_dataset,
    unique_image_paths,
)

FIGURES_ROOT = RESULTS_ROOT.parent / "figures"
RANDOM_SEED = 20260923
PAIR_FIELDS = [
    "dataset",
    "pair_type",
    "image_a",
    "image_b",
    "cosine_similarity",
    "cosine_distance",
    "euclidean_distance",
]


def threshold_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
    threshold: float,
) -> dict[str, float | int]:
    """Calculate confusion-matrix and rate metrics for ``score >= threshold``."""

    score_array = np.asarray(list(scores), dtype=np.float64)
    label_array = np.asarray(list(labels), dtype=np.int64)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional.")
    if len(score_array) != len(label_array) or len(score_array) == 0:
        raise ValueError("scores and labels must be non-empty and have equal lengths.")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite.")
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1.")

    predicted = score_array >= threshold
    positive = label_array == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    positive_count = tp + fn
    negative_count = tn + fp
    total = len(label_array)
    far = fp / negative_count if negative_count else np.nan
    frr = fn / positive_count if positive_count else np.nan
    return {
        "threshold": float(threshold),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "FAR": float(far),
        "FRR": float(frr),
        "TAR": float(tp / positive_count) if positive_count else np.nan,
        "TNR": float(tn / negative_count) if negative_count else np.nan,
        "accuracy": float((tp + tn) / total),
    }


def calculate_roc_metrics(scores: Iterable[float], labels: Iterable[int]) -> dict[str, Any]:
    """Return ROC-AUC, EER, EER threshold, and ROC arrays."""

    score_array = np.asarray(list(scores), dtype=np.float64)
    label_array = np.asarray(list(labels), dtype=np.int64)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("scores and labels must be non-empty and have equal lengths.")
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("ROC metrics require both positive and negative labels.")

    fpr, tpr, thresholds = roc_curve(label_array, score_array)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    return {
        "roc_auc": float(roc_auc_score(label_array, score_array)),
        "eer": eer,
        "eer_threshold": float(thresholds[eer_index]),
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "thresholds": thresholds,
        "roc_auc_trapezoid": float(auc(fpr, tpr)),
    }


def build_verification_rows(
    dataset_name: str,
    pair_type: str,
    pairs: list[Any],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Calculate verification scores and explicitly report missing embeddings."""

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for pair in pairs:
        image_a = str(pair.image_a)
        image_b = str(pair.image_b)
        if image_a not in embeddings or image_b not in embeddings:
            missing.append(
                {
                    "dataset": dataset_name,
                    "pair_type": pair_type,
                    "image_a": image_a,
                    "image_b": image_b,
                    "error": "Embedding missing for one or both pair images.",
                }
            )
            continue
        similarity = float(cosine_similarity(embeddings[image_a], embeddings[image_b]))
        rows.append(
            {
                "dataset": dataset_name,
                "pair_type": pair_type,
                "image_a": image_a,
                "image_b": image_b,
                "cosine_similarity": similarity,
                "cosine_distance": float(1.0 - similarity),
                "euclidean_distance": float(
                    euclidean_distance(embeddings[image_a], embeddings[image_b])
                ),
            }
        )
    return rows, missing


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, statistic: str, seed: int = RANDOM_SEED) -> tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(2000, values.size))
    samples = values[sample_indices]
    if statistic == "mean":
        estimates = np.mean(samples, axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError("Unsupported bootstrap statistic.")
    return (float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)))


def _bootstrap_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    estimates: list[float] = []
    for _ in range(2000):
        indices = rng.integers(0, len(scores), len(scores))
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        estimates.append(float(roc_auc_score(sample_labels, scores[indices])))
    if not estimates:
        return (np.nan, np.nan)
    return (float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)))


def _drift_bin_label(value: float) -> str:
    lower = np.floor(value * 10.0) / 10.0
    upper = lower + 0.1
    return f"{lower:.2f}-{upper:.2f}"


def _summary_for_dataset(
    dataset_name: str,
    rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positives = np.asarray(
        [r["cosine_similarity"] for r in rows if r["pair_type"] == "positive"],
        dtype=np.float64,
    )
    negatives = np.asarray(
        [r["cosine_similarity"] for r in rows if r["pair_type"] == "negative"],
        dtype=np.float64,
    )
    labels = np.concatenate(
        [np.ones(positives.size, dtype=np.int64), np.zeros(negatives.size, dtype=np.int64)]
    )
    scores = np.concatenate([positives, negatives])
    roc = calculate_roc_metrics(scores, labels)
    eer_metrics = threshold_metrics(scores, labels, roc["eer_threshold"])
    auc_ci = _bootstrap_auc(scores, labels)
    positive_drifts = 1.0 - positives
    mean_drift_ci = _bootstrap_ci(positive_drifts, "mean")
    median_drift_ci = _bootstrap_ci(positive_drifts, "median", seed=RANDOM_SEED + 1)
    return {
        "dataset": dataset_name,
        "positive_pairs": int(positives.size),
        "negative_pairs": int(negatives.size),
        "roc_auc": roc["roc_auc"],
        "roc_auc_ci_95": f"{auc_ci[0]:.8f},{auc_ci[1]:.8f}",
        "eer": roc["eer"],
        "eer_threshold": roc["eer_threshold"],
        "eer_far": eer_metrics["FAR"],
        "eer_frr": eer_metrics["FRR"],
        "max_accuracy": max(r["accuracy"] for r in threshold_rows),
        "max_accuracy_threshold": max(
            threshold_rows, key=lambda row: (row["accuracy"], row["threshold"])
        )["threshold"],
        "mean_positive_similarity": float(np.mean(positives)),
        "median_positive_similarity": float(np.median(positives)),
        "std_positive_similarity": float(np.std(positives)),
        "p05_positive_similarity": float(np.percentile(positives, 5)),
        "p95_positive_similarity": float(np.percentile(positives, 95)),
        "mean_negative_similarity": float(np.mean(negatives)),
        "median_negative_similarity": float(np.median(negatives)),
        "std_negative_similarity": float(np.std(negatives)),
        "p05_negative_similarity": float(np.percentile(negatives, 5)),
        "p95_negative_similarity": float(np.percentile(negatives, 95)),
        "mean_positive_drift": float(np.mean(positive_drifts)),
        "median_positive_drift": float(np.median(positive_drifts)),
        "mean_drift_ci_95": f"{mean_drift_ci[0]:.8f},{mean_drift_ci[1]:.8f}",
        "median_drift_ci_95": f"{median_drift_ci[0]:.8f},{median_drift_ci[1]:.8f}",
    }


def _make_figures(dataset_rows: dict[str, list[dict[str, Any]]], summaries: list[dict[str, Any]]) -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    names = list(dataset_rows)

    plt.figure(figsize=(9, 6))
    for name in names:
        drift = [1.0 - r["cosine_similarity"] for r in dataset_rows[name] if r["pair_type"] == "positive"]
        plt.hist(drift, bins=30, alpha=0.4, density=True, label=name)
    plt.xlabel("Identity drift (1 - cosine similarity)")
    plt.ylabel("Density")
    plt.title("Positive-pair drift distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / "drift_distribution_by_dataset.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.boxplot(
        [[1.0 - r["cosine_similarity"] for r in dataset_rows[name] if r["pair_type"] == "positive"] for name in names],
        tick_labels=names,
        showmeans=True,
    )
    plt.ylabel("Identity drift (1 - cosine similarity)")
    plt.title("Positive-pair drift comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / "drift_boxplot_by_dataset.png", dpi=300)
    plt.close()

    for name in names:
        positive = [r["cosine_similarity"] for r in dataset_rows[name] if r["pair_type"] == "positive"]
        negative = [r["cosine_similarity"] for r in dataset_rows[name] if r["pair_type"] == "negative"]
        plt.figure(figsize=(9, 6))
        plt.hist(positive, bins=30, alpha=0.5, density=True, label="positive/genuine")
        plt.hist(negative, bins=30, alpha=0.5, density=True, label="negative/impostor")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Density")
        plt.title(f"{name}: genuine vs impostor similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / f"{name}_positive_negative_similarity.png", dpi=300)
        plt.close()

        scores = np.asarray(positive + negative)
        labels = np.concatenate(
            [np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
        )
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(labels, scores):.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False accept rate")
        plt.ylabel("True accept rate")
        plt.title(f"{name}: ROC curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_ROOT / f"{name}_roc.png", dpi=300)
        plt.close()

    plt.figure(figsize=(9, 6))
    for name in names:
        summary = next(item for item in summaries if item["dataset"] == name)
        rows = dataset_rows[name]
        scores = np.asarray([r["cosine_similarity"] for r in rows])
        labels = np.asarray([1 if r["pair_type"] == "positive" else 0 for r in rows])
        thresholds = np.linspace(float(scores.min()), float(scores.max()), 200)
        far = [threshold_metrics(scores, labels, threshold)["FAR"] for threshold in thresholds]
        frr = [threshold_metrics(scores, labels, threshold)["FRR"] for threshold in thresholds]
        plt.plot(thresholds, far, label=f"{name} FAR")
        plt.plot(thresholds, frr, "--", label=f"{name} FRR")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Rate")
    plt.title("FAR and FRR across thresholds")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / "far_frr_threshold_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    means = [item["mean_positive_similarity"] for item in summaries]
    medians = [item["median_positive_similarity"] for item in summaries]
    x = np.arange(len(names))
    plt.bar(x - 0.18, means, width=0.36, label="mean similarity")
    plt.bar(x + 0.18, medians, width=0.36, label="median similarity")
    plt.xticks(x, names)
    plt.ylabel("Cosine similarity")
    plt.title("Positive-pair similarity by dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / "dataset_similarity_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    for name in names:
        positive = [
            {
                "drift": 1.0 - r["cosine_similarity"],
                "similarity": r["cosine_similarity"],
            }
            for r in dataset_rows[name]
            if r["pair_type"] == "positive"
        ]
        bins = np.arange(0.0, 1.21, 0.1)
        centers: list[float] = []
        acceptance: list[float] = []
        summary = next(item for item in summaries if item["dataset"] == name)
        for start in bins[:-1]:
            values = [
                item["similarity"]
                for item in positive
                if start <= item["drift"] < start + 0.1
            ]
            if values:
                centers.append(float(start + 0.05))
                acceptance.append(
                    float(np.mean(np.asarray(values) >= summary["eer_threshold"]))
                )
        plt.plot(centers, acceptance, marker="o", label=name)
    plt.xlabel("Positive-pair drift-bin midpoint")
    plt.ylabel("Acceptance rate at dataset EER threshold")
    plt.ylim(-0.02, 1.02)
    plt.title("Drift-bin verification acceptance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / "drift_bin_acceptance.png", dpi=300)
    plt.close()


def run_verification(datasets: list[str] | None = None, *, force: bool = False) -> dict[str, Any]:
    """Run full positive-vs-negative verification analysis."""

    selected = list(datasets or DEFAULT_DATASETS)
    start = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    all_missing: list[dict[str, str]] = []
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    threshold_output: list[dict[str, Any]] = []
    drift_bin_output: list[dict[str, Any]] = []
    quality_output: list[dict[str, Any]] = []
    statistical_output: list[dict[str, Any]] = []
    runtime_output: list[dict[str, Any]] = []

    for dataset_name in selected:
        positive_pairs = load_dataset_pairs(dataset_name, "positive")
        negative_pairs = load_dataset_pairs(dataset_name, "negative")
        combined_pairs = positive_pairs + negative_pairs
        extractor = None
        embeddings, generated, cache_hits, failures = extract_embeddings_for_dataset(
            dataset_name,
            combined_pairs,
            force=force,
            extractor=extractor,
        )
        all_missing.extend(
            {
                "dataset": dataset_name,
                "pair_type": "embedding",
                "image_a": failure["image_path"],
                "image_b": "",
                "error": failure["error"],
            }
            for failure in failures
        )
        positive_rows, positive_missing = build_verification_rows(
            dataset_name, "positive", positive_pairs, embeddings
        )
        negative_rows, negative_missing = build_verification_rows(
            dataset_name, "negative", negative_pairs, embeddings
        )
        rows = positive_rows + negative_rows
        all_missing.extend(positive_missing + negative_missing)
        dataset_rows[dataset_name] = rows
        all_rows.extend(rows)

        scores = np.asarray([r["cosine_similarity"] for r in rows], dtype=np.float64)
        labels = np.asarray([1 if r["pair_type"] == "positive" else 0 for r in rows], dtype=np.int64)
        threshold_values = np.unique(np.concatenate(([np.nextafter(scores.min(), -np.inf)], scores, [np.nextafter(scores.max(), np.inf)])))
        threshold_rows = [
            {"dataset": dataset_name, **threshold_metrics(scores, labels, threshold)}
            for threshold in threshold_values
        ]
        threshold_output.extend(threshold_rows)
        summary = _summary_for_dataset(dataset_name, rows, threshold_rows)
        summaries.append(summary)

        positive_drift = np.asarray(
            [1.0 - r["cosine_similarity"] for r in positive_rows], dtype=np.float64
        )
        for bin_start in np.arange(0.0, max(1.0, np.ceil(float(positive_drift.max()) * 10) / 10) + 0.1, 0.1):
            bin_end = bin_start + 0.1
            mask = (
                (positive_drift >= bin_start)
                & ((positive_drift < bin_end) | ((bin_end > positive_drift.max()) & (positive_drift <= bin_end)))
            )
            if not np.any(mask):
                continue
            bin_scores = 1.0 - positive_drift[mask]
            drift_bin_output.append(
                {
                    "dataset": dataset_name,
                    "drift_bin": f"{bin_start:.2f}-{bin_end:.2f}",
                    "pair_count": int(np.sum(mask)),
                    "mean_drift": float(np.mean(positive_drift[mask])),
                    "median_drift": float(np.median(positive_drift[mask])),
                    "mean_similarity": float(np.mean(bin_scores)),
                    "acceptance_rate_at_eer": float(
                        np.mean(bin_scores >= summary["eer_threshold"])
                    ),
                    "acceptance_rate_at_max_accuracy": float(
                        np.mean(bin_scores >= summary["max_accuracy_threshold"])
                    ),
                }
            )

        drift_values = positive_drift
        quality_output.extend(
            [
                {"dataset": dataset_name, "check": "positive_cosine_below_0", "value": int(np.sum(scores[labels == 1] < 0))},
                {"dataset": dataset_name, "check": "positive_cosine_above_0_9", "value": int(np.sum(scores[labels == 1] > 0.9))},
                {"dataset": dataset_name, "check": "positive_drift_above_1", "value": int(np.sum(drift_values > 1.0))},
                {"dataset": dataset_name, "check": "nan_or_inf_scores", "value": int(np.sum(~np.isfinite(scores)))},
                {"dataset": dataset_name, "check": "cosine_min", "value": float(np.min(scores))},
                {"dataset": dataset_name, "check": "cosine_max", "value": float(np.max(scores))},
                {"dataset": dataset_name, "check": "unique_images", "value": len(unique_image_paths(combined_pairs))},
                {"dataset": dataset_name, "check": "embeddings_generated", "value": generated},
                {"dataset": dataset_name, "check": "cache_hits", "value": cache_hits},
                {"dataset": dataset_name, "check": "embedding_failures", "value": len(failures)},
            ]
        )
        embedding_norms = np.asarray(
            [np.linalg.norm(value) for value in embeddings.values()], dtype=np.float64
        )
        embedding_dimensions = sorted({int(np.asarray(value).size) for value in embeddings.values()})
        identical_embedding_pairs = sum(
            np.array_equal(embeddings[str(pair.image_a)], embeddings[str(pair.image_b)])
            for pair in combined_pairs
            if str(pair.image_a) in embeddings and str(pair.image_b) in embeddings
        )
        duplicate_path_pairs = sum(pair.image_a == pair.image_b for pair in combined_pairs)
        file_hashes: dict[str, int] = {}
        for image_path in unique_image_paths(combined_pairs):
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            file_hashes[digest] = file_hashes.get(digest, 0) + 1
        quality_output.extend(
            [
                {"dataset": dataset_name, "check": "embedding_dimensions", "value": ",".join(map(str, embedding_dimensions))},
                {"dataset": dataset_name, "check": "embedding_norm_min", "value": float(np.min(embedding_norms)) if embedding_norms.size else np.nan},
                {"dataset": dataset_name, "check": "embedding_norm_max", "value": float(np.max(embedding_norms)) if embedding_norms.size else np.nan},
                {"dataset": dataset_name, "check": "embedding_norm_mean", "value": float(np.mean(embedding_norms)) if embedding_norms.size else np.nan},
                {"dataset": dataset_name, "check": "identical_embedding_pairs", "value": int(identical_embedding_pairs)},
                {"dataset": dataset_name, "check": "duplicate_path_pairs", "value": int(duplicate_path_pairs)},
                {"dataset": dataset_name, "check": "duplicate_content_groups", "value": int(sum(count > 1 for count in file_hashes.values()))},
                {"dataset": dataset_name, "check": "duplicate_content_files_beyond_first", "value": int(sum(count - 1 for count in file_hashes.values() if count > 1))},
            ]
        )
        consistency_errors = []
        for row in rows:
            expected = 2.0 * row["cosine_distance"]
            actual = row["euclidean_distance"] ** 2
            consistency_errors.append(abs(actual - expected))
        quality_output.append(
            {
                "dataset": dataset_name,
                "check": "max_euclidean_cosine_consistency_error",
                "value": float(max(consistency_errors, default=np.nan)),
            }
        )
        runtime_output.append(
            {
                "dataset": dataset_name,
                "positive_pairs": len(positive_pairs),
                "negative_pairs": len(negative_pairs),
                "valid_rows": len(rows),
                "unique_images": len(unique_image_paths(combined_pairs)),
                "embeddings_generated": generated,
                "cache_hits": cache_hits,
                "failures": len(failures),
            }
        )

    for index, left in enumerate(selected):
        left_drift = np.asarray(
            [1.0 - r["cosine_similarity"] for r in dataset_rows[left] if r["pair_type"] == "positive"]
        )
        for right in selected[index + 1 :]:
            right_drift = np.asarray(
                [1.0 - r["cosine_similarity"] for r in dataset_rows[right] if r["pair_type"] == "positive"]
            )
            result = mannwhitneyu(left_drift, right_drift, alternative="two-sided")
            u = float(result.statistic)
            effect = (2.0 * u / (len(left_drift) * len(right_drift))) - 1.0
            statistical_output.append(
                {
                    "comparison": f"{left} vs {right}",
                    "test": "Mann-Whitney U",
                    "statistic": u,
                    "p_value": float(result.pvalue),
                    "effect_size_rank_biserial": effect,
                    "confidence_interval_if_available": "",
                    "adjusted_p_value_if_applicable": "",
                }
            )
    if statistical_output:
        order = sorted(range(len(statistical_output)), key=lambda i: statistical_output[i]["p_value"])
        m = len(statistical_output)
        adjusted = [np.nan] * m
        for rank, original_index in enumerate(order, start=1):
            adjusted[original_index] = min(
                1.0, statistical_output[original_index]["p_value"] * m / rank
            )
        for item, value in zip(statistical_output, adjusted):
            item["adjusted_p_value_if_applicable"] = float(value)

    _write_csv(RESULTS_ROOT / "verification_pairs.csv", PAIR_FIELDS, all_rows)
    _write_csv(
        RESULTS_ROOT / "verification_summary.csv",
        list(summaries[0].keys()) if summaries else ["dataset"],
        summaries,
    )
    _write_csv(
        RESULTS_ROOT / "threshold_metrics.csv",
        ["dataset", "threshold", "TP", "TN", "FP", "FN", "FAR", "FRR", "TAR", "TNR", "accuracy"],
        threshold_output,
    )
    _write_csv(
        RESULTS_ROOT / "drift_bins.csv",
        [
            "dataset",
            "drift_bin",
            "pair_count",
            "mean_drift",
            "median_drift",
            "mean_similarity",
            "acceptance_rate_at_eer",
            "acceptance_rate_at_max_accuracy",
        ],
        drift_bin_output,
    )
    _write_csv(
        RESULTS_ROOT / "statistical_tests.csv",
        [
            "comparison",
            "test",
            "statistic",
            "p_value",
            "effect_size_rank_biserial",
            "confidence_interval_if_available",
            "adjusted_p_value_if_applicable",
        ],
        statistical_output,
    )
    _write_csv(RESULTS_ROOT / "data_quality_report.csv", ["dataset", "check", "value"], quality_output)
    _write_csv(
        RESULTS_ROOT / "verification_failures.csv",
        ["dataset", "pair_type", "image_a", "image_b", "error"],
        all_missing,
    )
    _make_figures(dataset_rows, summaries)

    report = _build_report(
        selected,
        summaries,
        runtime_output,
        statistical_output,
        quality_output,
        len(all_rows),
        time.perf_counter() - start,
    )
    (RESULTS_ROOT / "stage2_research_summary.md").write_text(report, encoding="utf-8")
    return {
        "summaries": summaries,
        "runtime": runtime_output,
        "rows": all_rows,
        "failures": all_missing,
        "statistical_tests": statistical_output,
    }


def _build_report(
    datasets: list[str],
    summaries: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    row_count: int,
    elapsed: float,
) -> str:
    lines = [
        "# Stage 2 Verification and Drift Analysis",
        "",
        "## Research objective",
        "This exploratory analysis compares genuine positive-pair similarity with impostor negative-pair similarity and examines how positive-pair drift relates to verification outcomes. It does not estimate causality or define an operational re-enrollment threshold.",
        "",
        "## Experimental setup",
        f"- Datasets: {', '.join(datasets)}",
        f"- Verification rows written: {row_count}",
        "- Embeddings: normalized 512-dimensional ArcFace/InsightFace embeddings",
        "- Similarity score: cosine similarity; higher scores are accepted as more likely genuine",
        f"- Random seed for bootstrap procedures: {RANDOM_SEED}",
        f"- Runtime: {elapsed:.2f} seconds",
        "",
        "## Verification results",
        "",
        "| Dataset | Positive | Negative | ROC-AUC | EER | EER threshold | Max accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['dataset']} | {item['positive_pairs']} | {item['negative_pairs']} | "
            f"{item['roc_auc']:.6f} | {item['eer']:.6f} | {item['eer_threshold']:.6f} | "
            f"{item['max_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "EER and maximum-accuracy thresholds are descriptive candidate operating points, not final or universal thresholds.",
            "",
            "## Positive versus negative similarity",
            "",
        ]
    )
    for item in summaries:
        lines.append(
            f"- **{item['dataset']}**: positive mean/median "
            f"{item['mean_positive_similarity']:.6f}/{item['median_positive_similarity']:.6f}; "
            f"negative mean/median {item['mean_negative_similarity']:.6f}/{item['median_negative_similarity']:.6f}."
        )
    lines.extend(
        [
            "",
            "## Drift interpretation",
            "",
            "For positive pairs, drift is exactly `1 - cosine_similarity`; therefore a correlation between these two columns would be mathematically tautological and is not reported as an independent scientific result. Drift-bin acceptance rates are provided in `drift_bins.csv` at descriptive candidate thresholds.",
            "",
            "## Statistical comparisons",
            "",
        ]
    )
    for item in tests:
        lines.append(
            f"- {item['comparison']}: Mann-Whitney U={item['statistic']:.3f}, "
            f"p={item['p_value']:.6g}, rank-biserial effect={item['effect_size_rank_biserial']:.6f}, "
            f"BH-adjusted p={item['adjusted_p_value_if_applicable']:.6g}."
        )
    lines.extend(
        [
            "",
            "## Data-quality findings",
            "",
            "Counts for negative-similarity positive pairs, high-similarity positive pairs, drift above 1, missing embeddings, and Euclidean/cosine consistency are in `data_quality_report.csv`. Negative cosine values are reported rather than discarded.",
            "",
            "## Limitations",
            "",
            "- The annotation files provide pair labels but no explicit identity IDs.",
            "- Pair-level results do not support identity-disjoint development/test threshold validation.",
            "- The datasets are observational pair collections; these analyses do not establish that age, pose, or any other condition caused a measured change.",
            "- Thresholds are exploratory descriptive operating points, not deployment thresholds.",
            "- Bootstrap intervals and pairwise tests treat annotated pairs as independent observations; shared identities cannot be ruled out from the available metadata.",
            "- Duplicate image content can exist in the supplied image directories; the quality report records detectable duplicate content among referenced files.",
            "",
            "## Conclusions supported",
            "",
            "The outputs quantify genuine/impostor separation and describe how acceptance changes with similarity and positive-pair drift. They provide a verification baseline for later work, but do not by themselves justify a Stability Index or an adaptive re-enrollment policy.",
            "",
            "## Recommended next stage",
            "",
            "Before implementing the Stability Index, obtain or construct an evaluation protocol with explicit identity grouping and identity-disjoint development/validation/test splits, then assess whether drift-derived features add information beyond the verification score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 verification and drift analysis.")
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = list(DEFAULT_DATASETS) if args.all or not args.dataset else list(args.dataset)
    result = run_verification(selected, force=args.force)
    for item in result["summaries"]:
        print(
            f"{item['dataset']}: pairs={item['positive_pairs'] + item['negative_pairs']} "
            f"AUC={item['roc_auc']:.6f} EER={item['eer']:.6f} "
            f"EER_threshold={item['eer_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
