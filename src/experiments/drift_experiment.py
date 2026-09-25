"""Embedding and drift experiment pipeline for validation dataset pairs."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data import load_dataset_pairs
from src.drift.drift_metrics import cosine_similarity, euclidean_distance, identity_drift
from src.embedding.embedding_extractor import FaceEmbeddingExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_ROOT = PROJECT_ROOT / "embeddings" / "validation"
RESULTS_ROOT = PROJECT_ROOT / "results" / "metrics"
DEFAULT_DATASETS = ("agedb_30", "calfw", "cplfw", "lfw")


@dataclass
class ExperimentSummary:
    """Runtime summary for a drift experiment."""

    dataset: str
    positive_pairs: int
    unique_images: int
    embeddings_generated: int
    cache_hits: int
    failures: int
    processing_time_seconds: float
    mean_drift: float | None = None
    median_drift: float | None = None
    p95_drift: float | None = None


def unique_image_paths(pairs: list[Any]) -> list[Path]:
    """Return unique image paths in annotation order for the provided dataset pairs."""

    seen: set[str] = set()
    ordered: list[Path] = []
    for pair in pairs:
        for image_path in (pair.image_a, pair.image_b):
            key = str(image_path)
            if key not in seen:
                seen.add(key)
                ordered.append(image_path)
    return ordered


def build_embedding_cache_path(dataset_name: str) -> Path:
    """Return the on-disk cache path for a dataset's embeddings."""

    EMBEDDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    return EMBEDDINGS_ROOT / f"{dataset_name}_embeddings.npz"


def _ensure_results_dir() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def _dataset_pairs_for_run(dataset_name: str, pilot: bool) -> list[Any]:
    pairs = load_dataset_pairs(dataset_name, pair_type="positive")
    if pilot:
        return pairs[:100]
    return pairs


def extract_embeddings_for_dataset(
    dataset_name: str,
    pairs: list[Any],
    *,
    force: bool = False,
    extractor: FaceEmbeddingExtractor | None = None,
    cache_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], int, int, list[dict[str, str]]]:
    """Extract and cache one embedding per unique image for a dataset."""

    cache_path = cache_path or build_embedding_cache_path(dataset_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = unique_image_paths(pairs)
    cache_data: dict[str, np.ndarray] = {}
    cache_hits = 0
    failures: list[dict[str, str]] = []

    if extractor is None:
        extractor = FaceEmbeddingExtractor()
        extractor.initialize()

    expected_embedding_dimension = (
        512 if hasattr(extractor, "extract_aligned_embedding") else None
    )

    if cache_path.exists() and not force:
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            if "paths" in loaded.files and "embeddings" in loaded.files:
                path_list = [str(path) for path in loaded["paths"].tolist()]
                embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
                cache_is_compatible = (
                    embeddings.ndim == 2
                    and len(path_list) == embeddings.shape[0]
                    and (
                        expected_embedding_dimension is None
                        or embeddings.shape[1] == expected_embedding_dimension
                    )
                )
                if cache_is_compatible:
                    expected_shape = embeddings.shape[1]
                    valid = True
                    for path, embedding in zip(path_list, embeddings, strict=True):
                        candidate = np.asarray(embedding, dtype=np.float32)
                        if candidate.shape != (expected_shape,):
                            valid = False
                            break
                        cache_data[path] = candidate
                    if not valid:
                        cache_data = {}
                else:
                    cache_data = {}
            else:
                for key in loaded.files:
                    if key.startswith("path_"):
                        continue
                    cache_data[str(key)] = np.asarray(loaded[key], dtype=np.float32)
        except Exception:
            cache_data = {}

    to_embed: list[Path] = []
    for image_path in image_paths:
        key = str(image_path)
        if key in cache_data:
            cache_hits += 1
            continue
        to_embed.append(image_path)

    for image_path in to_embed:
        try:
            if hasattr(extractor, "extract_aligned_embedding"):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"OpenCV could not read image file: {image_path}")
                embedding = extractor.extract_aligned_embedding(image)
            else:
                embedding = extractor.extract_embedding(str(image_path))
            cache_data[str(image_path)] = np.asarray(embedding, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive path for runtime robustness
            failures.append({"dataset": dataset_name, "image_path": str(image_path), "error": str(exc)})

    if cache_data and (to_embed or force or not cache_path.exists()):
        valid_shapes: set[tuple[int, ...]] = {tuple(np.asarray(value).shape) for value in cache_data.values()}
        if len(valid_shapes) > 1:
            canonical_shape = None
            filtered: dict[str, np.ndarray] = {}
            for path, value in cache_data.items():
                array = np.asarray(value, dtype=np.float32)
                shape = tuple(array.shape)
                if canonical_shape is None:
                    canonical_shape = shape
                if shape == canonical_shape:
                    filtered[path] = array
            cache_data = filtered
        paths = [str(path) for path in sorted({str(path) for path in cache_data})]
        if not paths:
            return cache_data, len(to_embed), cache_hits, failures
        first_shape = tuple(np.asarray(cache_data[paths[0]], dtype=np.float32).shape)
        embeddings = np.stack([
            np.asarray(cache_data[path], dtype=np.float32).reshape(first_shape)
            for path in paths
        ], axis=0)
        np.savez_compressed(cache_path, paths=np.asarray(paths, dtype=str), embeddings=embeddings)

    return cache_data, len(to_embed), cache_hits, failures


def compute_pairwise_metrics(dataset_name: str, pairs: list[Any], embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Compute per-pair drift and similarity metrics using cached embeddings."""

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        image_a = str(pair.image_a)
        image_b = str(pair.image_b)
        if image_a not in embeddings or image_b not in embeddings:
            continue

        embedding_a = embeddings[image_a]
        embedding_b = embeddings[image_b]
        similarity = float(cosine_similarity(embedding_a, embedding_b))
        drift = float(identity_drift(embedding_a, embedding_b))
        distance = float(euclidean_distance(embedding_a, embedding_b))

        rows.append(
            {
                "dataset": dataset_name,
                "image_a": image_a,
                "image_b": image_b,
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
                "identity_drift": drift,
                "euclidean_distance": distance,
            }
        )

    return rows


def _write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_metrics(dataset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "dataset": dataset_name,
            "number_of_positive_pairs": 0,
            "mean_cosine_similarity": np.nan,
            "median_cosine_similarity": np.nan,
            "std_cosine_similarity": np.nan,
            "min_cosine_similarity": np.nan,
            "max_cosine_similarity": np.nan,
            "mean_identity_drift": np.nan,
            "median_identity_drift": np.nan,
            "std_identity_drift": np.nan,
            "min_identity_drift": np.nan,
            "max_identity_drift": np.nan,
            "p25_identity_drift": np.nan,
            "p75_identity_drift": np.nan,
            "p95_identity_drift": np.nan,
            "mean_euclidean_distance": np.nan,
        }

    similarities = np.asarray([row["cosine_similarity"] for row in rows], dtype=np.float64)
    drifts = np.asarray([row["identity_drift"] for row in rows], dtype=np.float64)
    distances = np.asarray([row["euclidean_distance"] for row in rows], dtype=np.float64)

    return {
        "dataset": dataset_name,
        "number_of_positive_pairs": int(len(rows)),
        "mean_cosine_similarity": float(np.mean(similarities)),
        "median_cosine_similarity": float(np.median(similarities)),
        "std_cosine_similarity": float(np.std(similarities)),
        "min_cosine_similarity": float(np.min(similarities)),
        "max_cosine_similarity": float(np.max(similarities)),
        "mean_identity_drift": float(np.mean(drifts)),
        "median_identity_drift": float(np.median(drifts)),
        "std_identity_drift": float(np.std(drifts)),
        "min_identity_drift": float(np.min(drifts)),
        "max_identity_drift": float(np.max(drifts)),
        "p25_identity_drift": float(np.percentile(drifts, 25)),
        "p75_identity_drift": float(np.percentile(drifts, 75)),
        "p95_identity_drift": float(np.percentile(drifts, 95)),
        "mean_euclidean_distance": float(np.mean(distances)),
    }


def run_experiment(
    datasets: list[str] | None = None,
    *,
    pilot: bool = False,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[ExperimentSummary]]:
    """Run the embedding + drift experiment for selected datasets."""

    if datasets is None:
        selected = list(DEFAULT_DATASETS)
    else:
        selected = list(datasets)

    all_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    summaries: list[ExperimentSummary] = []
    per_dataset_outputs: list[dict[str, Any]] = []

    _ensure_results_dir()
    extractor = FaceEmbeddingExtractor()
    extractor.initialize()

    start_total = time.perf_counter()
    total_cache_hits = 0
    total_embeddings_generated = 0
    total_failure_count = 0

    for dataset_name in selected:
        dataset_start = time.perf_counter()
        pairs = _dataset_pairs_for_run(dataset_name, pilot)
        embeddings, generated_count, cache_hits, failure_list = extract_embeddings_for_dataset(
            dataset_name,
            pairs,
            force=force,
            extractor=extractor,
        )
        total_cache_hits += cache_hits
        total_embeddings_generated += generated_count
        total_failure_count += len(failure_list)

        rows = compute_pairwise_metrics(dataset_name, pairs, embeddings)
        all_rows.extend(rows)
        failure_rows.extend({"dataset": dataset_name, "image_path": failure["image_path"], "error": failure["error"]} for failure in failure_list)

        summary = _summarize_metrics(dataset_name, rows)
        per_dataset_outputs.append(summary)
        summaries.append(
            ExperimentSummary(
                dataset=dataset_name,
                positive_pairs=len(rows),
                unique_images=len(unique_image_paths(pairs)),
                embeddings_generated=generated_count,
                cache_hits=cache_hits,
                failures=len(failure_list),
                processing_time_seconds=time.perf_counter() - dataset_start,
                mean_drift=summary["mean_identity_drift"],
                median_drift=summary["median_identity_drift"],
                p95_drift=summary["p95_identity_drift"],
            )
        )

        _write_rows_csv(
            RESULTS_ROOT / f"{dataset_name}_drift_pairs.csv",
            [
                "dataset",
                "image_a",
                "image_b",
                "cosine_similarity",
                "cosine_distance",
                "identity_drift",
                "euclidean_distance",
            ],
            rows,
        )

    combined_pairs_path = RESULTS_ROOT / "drift_pairs.csv"
    _write_rows_csv(
        combined_pairs_path,
        [
            "dataset",
            "image_a",
            "image_b",
            "cosine_similarity",
            "cosine_distance",
            "identity_drift",
            "euclidean_distance",
        ],
        all_rows,
    )

    summary_path = RESULTS_ROOT / "drift_summary.csv"
    _write_rows_csv(
        summary_path,
        [
            "dataset",
            "number_of_positive_pairs",
            "mean_cosine_similarity",
            "median_cosine_similarity",
            "std_cosine_similarity",
            "min_cosine_similarity",
            "max_cosine_similarity",
            "mean_identity_drift",
            "median_identity_drift",
            "std_identity_drift",
            "min_identity_drift",
            "max_identity_drift",
            "p25_identity_drift",
            "p75_identity_drift",
            "p95_identity_drift",
            "mean_euclidean_distance",
        ],
        per_dataset_outputs,
    )

    failure_path = RESULTS_ROOT / "embedding_failures.csv"
    _write_rows_csv(
        failure_path,
        ["dataset", "image_path", "error"],
        failure_rows,
    )

    total_processing_seconds = time.perf_counter() - start_total
    print(
        "Processed datasets: "
        + ", ".join(selected)
        + f" | pairs={sum(s.positive_pairs for s in summaries)}"
        + f" | unique_images={sum(s.unique_images for s in summaries)}"
        + f" | cache_hits={total_cache_hits}"
        + f" | embeddings_generated={total_embeddings_generated}"
        + f" | failures={total_failure_count}"
        + f" | processing_time={total_processing_seconds:.2f}s"
        + f" | pairs_per_second={sum(s.positive_pairs for s in summaries)/max(total_processing_seconds, 1e-9):.2f}"
    )

    return all_rows, per_dataset_outputs, failure_rows, summaries, summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the validation-pair embedding and drift experiment.")
    parser.add_argument("--pilot", action="store_true", help="Process the first 100 positive pairs for each selected dataset.")
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS, help="Dataset to process, repeatable.")
    parser.add_argument("--all", action="store_true", help="Process all supported datasets.")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even when a cached dataset archive exists.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_names: list[str] = []
    if args.dataset:
        dataset_names.extend(args.dataset)
    if args.all:
        dataset_names = list(DEFAULT_DATASETS)
    if not dataset_names:
        dataset_names = list(DEFAULT_DATASETS)

    run_experiment(dataset_names, pilot=args.pilot, force=args.force)


if __name__ == "__main__":
    main()
