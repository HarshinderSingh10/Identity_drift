"""Stage 5 ArcFace embedding extraction for aligned MORPH faces.

This module consumes only successful Stage 4 outputs.  It loads the buffalo_l
recognition model directly, avoiding a second face detector and preserving the
Stage 4 alignment boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import onnxruntime as ort
from insightface import __version__ as insightface_version
from insightface import model_zoo
from src.embedding.embedding_extractor import FaceEmbeddingExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "morph"
EMBEDDINGS_ROOT = PROJECT_ROOT / "embeddings" / "morph"
EMBEDDING_FILES_ROOT = EMBEDDINGS_ROOT / "embeddings"
RESULTS_ROOT = PROJECT_ROOT / "results"
METRICS_ROOT = RESULTS_ROOT / "metrics"
REPORT_ROOT = PROJECT_ROOT / "experiments" / "05_embedding_extraction"

PREPROCESSING_METADATA = METADATA_ROOT / "morph_preprocessing.csv"
PAIR_METADATA = METADATA_ROOT / "morph_pair_preprocessing_eligibility.csv"
RESEARCH_IMAGES = METADATA_ROOT / "morph_research_images.csv"

STAGE5_VERSION = "stage5-v1"
MODEL_NAME = "buffalo_l"
EMBEDDING_DIMENSION = 512
INPUT_SIZE = (112, 112)
EXPECTED_SUCCESSFUL_IMAGES = 50_013
NORM_TOLERANCE = 1e-3
LOGGER = logging.getLogger(__name__)


def available_providers() -> list[str]:
    """Return execution providers exposed by the installed ONNX Runtime."""

    return list(ort.get_available_providers())


def cuda_runtime_issue() -> str | None:
    """Return a concrete local CUDA/cuDNN loading issue when detectable."""

    if "CUDAExecutionProvider" not in available_providers():
        return "CUDAExecutionProvider is not exposed by ONNX Runtime"
    if os.name != "nt":
        return None
    cudnn_dll = Path(ort.__file__).resolve().parents[1] / "nvidia" / "cudnn" / "bin"
    expected = cudnn_dll / "cudnn_engines_tensor_ir64_9.dll"
    if not expected.exists():
        return f"Missing cuDNN library: {expected}"
    return None


def stable_image_id(relative_path: str) -> str:
    """Return a process-independent identifier for a processed image path."""

    return hashlib.sha256(relative_path.replace("\\", "/").encode("utf-8")).hexdigest()[:24]


def embedding_path_for(relative_path: str) -> Path:
    """Return the deterministic canonical cache path for one image."""

    return EMBEDDING_FILES_ROOT / f"{stable_image_id(relative_path)}.npy"


def normalize_embedding(feature: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize one model feature vector."""

    embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
    if embedding.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-D embedding, got shape {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("Invalid zero-norm embedding")
    normalized = (embedding / norm).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embedding contains NaN or infinite values")
    return normalized


def validate_embedding_cache(path: Path, tolerance: float = NORM_TOLERANCE) -> bool:
    """Return whether a cached embedding satisfies the Stage 5 invariants."""

    try:
        embedding = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    if embedding.shape != (EMBEDDING_DIMENSION,):
        return False
    if embedding.dtype != np.float32:
        return False
    if not np.isfinite(embedding).all():
        return False
    norm = float(np.linalg.norm(embedding))
    return norm > 1e-12 and abs(norm - 1.0) <= tolerance


def load_aligned_image(path: Path) -> np.ndarray:
    """Load and validate a Stage 4 aligned BGR image."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode processed image: {path}")
    if image.shape != (INPUT_SIZE[1], INPUT_SIZE[0], 3):
        raise ValueError(f"Expected 112x112x3 image, got {image.shape}: {path}")
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError(f"Expected non-empty uint8 image: {path}")
    return image


def _rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_successful_preprocessing_rows(
    metadata_path: Path = PREPROCESSING_METADATA,
    *,
    require_expected_count: bool = True,
) -> list[dict[str, str]]:
    """Load exactly the successful Stage 4 rows eligible for embedding."""

    rows = _rows_from_csv(metadata_path)
    successful = [
        row for row in rows
        if row.get("preprocessing_status") == "success"
        and row.get("processed_filepath")
    ]
    if require_expected_count and len(successful) != EXPECTED_SUCCESSFUL_IMAGES:
        raise RuntimeError(
            "Stage 4 successful-image count mismatch: "
            f"expected {EXPECTED_SUCCESSFUL_IMAGES}, found {len(successful)}"
        )
    return sorted(successful, key=lambda row: row["processed_filepath"])


def _model_active_providers(model: object) -> list[str]:
    session = getattr(model, "session", None)
    if session is not None and hasattr(session, "get_providers"):
        return list(session.get_providers())
    return []


class AlignedArcFaceModel:
    """Recognition-only buffalo_l model for already aligned 112x112 faces."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model: object | None = None
        self.requested_providers: list[str] = []
        self.active_providers: list[str] = []
        self.cuda_error: str | None = None

    def initialize(self) -> None:
        available = available_providers()
        candidates = []
        if "CUDAExecutionProvider" in available:
            candidates.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            candidates.append("CPUExecutionProvider")
        if not candidates:
            raise RuntimeError(f"No supported ONNX Runtime provider is available: {available}")

        # Do not pass CPU as a silent fallback while determining whether CUDA
        # can actually execute recognition.  A configured provider list alone
        # is not evidence that CUDA ran a model operation.
        attempt_providers = (
            ["CUDAExecutionProvider"]
            if "CUDAExecutionProvider" in candidates
            else ["CPUExecutionProvider"]
        )
        self.requested_providers = attempt_providers
        detected_cuda_issue = (
            cuda_runtime_issue()
            if attempt_providers == ["CUDAExecutionProvider"]
            else None
        )
        if detected_cuda_issue is not None:
            self.cuda_error = detected_cuda_issue
            self.use_cpu(detected_cuda_issue)
            return
        try:
            # Reuse the project's established Windows CUDA/cuDNN DLL setup
            # before creating the recognition session.
            FaceEmbeddingExtractor._preload_onnxruntime_dlls()
            self.model = model_zoo.get_model(self.model_name, providers=attempt_providers)
            if self.model is None:
                raise RuntimeError("InsightFace returned no recognition model")
            if hasattr(self.model, "prepare"):
                self.model.prepare(ctx_id=0 if attempt_providers[0] == "CUDAExecutionProvider" else -1)
            self.active_providers = _model_active_providers(self.model)
        except Exception as exc:
            if "CUDAExecutionProvider" not in attempt_providers:
                raise
            self.cuda_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("CUDA initialization failed; retrying CPU only: %s", self.cuda_error)
            self.requested_providers = ["CPUExecutionProvider"]
            self.model = model_zoo.get_model(
                self.model_name, providers=["CPUExecutionProvider"]
            )
            if self.model is None:
                raise RuntimeError("InsightFace returned no CPU recognition model")
            if hasattr(self.model, "prepare"):
                self.model.prepare(ctx_id=-1)
            self.active_providers = _model_active_providers(self.model)

        if not self.active_providers:
            self.active_providers = list(self.requested_providers)

    @property
    def gpu_actually_used(self) -> bool:
        return (
            self.cuda_error is None
            and bool(self.active_providers)
            and self.active_providers[0] == "CUDAExecutionProvider"
        )

    def embed(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Recognition model is not initialized")
        if not hasattr(self.model, "get_feat"):
            raise RuntimeError("Recognition model does not expose get_feat()")
        return normalize_embedding(self.model.get_feat(image))

    def use_cpu(self, reason: str) -> None:
        """Replace a failed CUDA model with an explicitly CPU-only model."""

        self.cuda_error = reason
        self.requested_providers = ["CPUExecutionProvider"]
        self.model = model_zoo.get_model(
            self.model_name, providers=["CPUExecutionProvider"]
        )
        if self.model is None:
            raise RuntimeError("InsightFace returned no CPU recognition model")
        if hasattr(self.model, "prepare"):
            self.model.prepare(ctx_id=-1)
        self.active_providers = _model_active_providers(self.model)


def _metadata_row(row: dict[str, str], embedding_path: Path, embedding: np.ndarray) -> dict[str, Any]:
    relative_path = str(Path(row["processed_filepath"]).relative_to(PROCESSED_ROOT)).replace("\\", "/")
    return {
        "image_id": stable_image_id(relative_path),
        "identity_id": row["identity_id"],
        "filename": row["filename"],
        "relative_path": relative_path,
        "raw_filepath": row["raw_filepath"],
        "processed_filepath": row["processed_filepath"],
        "filename_age": row["filename_age"],
        "csv_scaled_age": row["csv_scaled_age"],
        "research_split": row["research_split"],
        "preprocessing_version": row["preprocessing_version"],
        "embedding_model": "InsightFace buffalo_l ArcFace",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_dtype": str(embedding.dtype),
        "embedding_normalized": "true",
        "embedding_norm": f"{np.linalg.norm(embedding):.8f}",
        "embedding_path": str(embedding_path),
        "embedding_status": "success",
    }


METADATA_FIELDS = [
    "image_id", "identity_id", "filename", "relative_path", "raw_filepath",
    "processed_filepath", "filename_age", "csv_scaled_age", "research_split",
    "preprocessing_version", "embedding_model", "embedding_dimension",
    "embedding_dtype", "embedding_normalized", "embedding_norm",
    "embedding_path", "embedding_status",
]


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _embedding_result(
    model: AlignedArcFaceModel,
    rows: list[dict[str, str]],
    *,
    persist: bool,
    progress: bool = False,
) -> dict[str, Any]:
    EMBEDDING_FILES_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_hits = 0
    new_embeddings = 0
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        relative_path = str(Path(row["processed_filepath"]).relative_to(PROCESSED_ROOT)).replace("\\", "/")
        target = embedding_path_for(relative_path)
        try:
            if validate_embedding_cache(target):
                embedding = np.load(target, allow_pickle=False)
                cache_hits += 1
            else:
                image = load_aligned_image(Path(row["processed_filepath"]))
                embedding = model.embed(image)
                if persist:
                    np.save(target, embedding)
                new_embeddings += 1
            metadata_rows.append(_metadata_row(row, target, embedding))
        except Exception as exc:
            failures.append({
                "image_id": stable_image_id(relative_path),
                "identity_id": row["identity_id"],
                "relative_path": relative_path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        if progress and (index == 1 or index % 100 == 0 or index == len(rows)):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            print(f"Embedding extraction: {index}/{len(rows)} | {rate:.2f} images/sec")

    elapsed = time.perf_counter() - started
    return {
        "metadata_rows": metadata_rows,
        "failures": failures,
        "cache_hits": cache_hits,
        "new_embeddings": new_embeddings,
        "elapsed_seconds": elapsed,
    }


def run_smoke_test(sample_size: int = 12) -> dict[str, Any]:
    """Run model, validity, and reproducibility checks without writing cache files."""

    rows = load_successful_preprocessing_rows(require_expected_count=True)
    sample = rows[: max(3, min(sample_size, len(rows)))]
    model = AlignedArcFaceModel()
    model.initialize()

    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for row in sample:
        image = load_aligned_image(Path(row["processed_filepath"]))
        try:
            first.append(model.embed(image))
        except Exception as exc:
            if model.requested_providers != ["CUDAExecutionProvider"]:
                raise
            model.use_cpu(f"{type(exc).__name__}: {exc}")
            first.append(model.embed(image))
    for row in sample:
        image = load_aligned_image(Path(row["processed_filepath"]))
        second.append(model.embed(image))

    differences = [np.abs(a - b) for a, b in zip(first, second)]
    cosine_values = [
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for a, b in zip(first, second)
    ]
    norms = np.asarray([np.linalg.norm(embedding) for embedding in first], dtype=np.float32)
    reproducible = bool(
        max(float(diff.max()) for diff in differences) <= 1e-5
        and min(cosine_values) >= 1.0 - 1e-6
    )
    return {
        "model": MODEL_NAME,
        "insightface_version": str(insightface_version),
        "onnxruntime_version": ort.__version__,
        "available_providers": available_providers(),
        "requested_providers": model.requested_providers,
        "active_providers": model.active_providers,
        "cuda_available": "CUDAExecutionProvider" in available_providers(),
        "gpu_actually_used": model.gpu_actually_used,
        "cuda_error": model.cuda_error,
        "model_loaded": True,
        "number_of_test_images": len(sample),
        "embedding_shape": list(first[0].shape),
        "embedding_dtype": str(first[0].dtype),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "finite_values": bool(all(np.isfinite(embedding).all() for embedding in first)),
        "max_absolute_difference": max(float(diff.max()) for diff in differences),
        "mean_absolute_difference": float(np.mean([diff.mean() for diff in differences])),
        "min_repeat_cosine_similarity": min(cosine_values),
        "reproducibility": "PASS" if reproducible else "FAIL",
        "status": "PASS" if reproducible else "FAIL",
    }


def _git_commit() -> str | None:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_full_extraction(batch_size: int = 16) -> dict[str, Any]:
    """Run resumable extraction and write canonical Stage 5 outputs."""

    del batch_size  # Recognition-only get_feat is currently one-image-at-a-time.
    rows = load_successful_preprocessing_rows(require_expected_count=True)
    model = AlignedArcFaceModel()
    model.initialize()
    result = _embedding_result(model, rows, persist=True, progress=True)
    metadata_path = METADATA_ROOT / "morph_embedding_metadata.csv"
    _write_csv(metadata_path, METADATA_FIELDS, result["metadata_rows"])
    _write_csv(
        RESULTS_ROOT / "metrics" / "embedding_failures.csv",
        ["image_id", "identity_id", "relative_path", "error_type", "error_message", "timestamp"],
        result["failures"],
    )
    _write_quality_audit(result["metadata_rows"])
    _write_split_summary(result["metadata_rows"])
    _write_extraction_summary(result, rows)
    manifest = {
        "stage": STAGE5_VERSION,
        "dataset": "MORPH",
        "model": "InsightFace buffalo_l ArcFace",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "normalization": "L2",
        "input_size": "112x112",
        "source": "Stage 4 aligned images",
        "total_input_images": len(_rows_from_csv(PREPROCESSING_METADATA)),
        "successful_preprocessing_images": len(rows),
        "embedding_attempts": len(rows),
        "successful_embeddings": len(result["metadata_rows"]),
        "failed_embeddings": len(result["failures"]),
        "cache_hits": result["cache_hits"],
        "new_embeddings": result["new_embeddings"],
        "execution_provider": model.active_providers,
        "onnxruntime_version": ort.__version__,
        "insightface_version": str(insightface_version),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    (METADATA_ROOT / "morph_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_report(manifest, result)
    return manifest


def _write_quality_audit(metadata_rows: list[dict[str, Any]]) -> None:
    embeddings = [
        np.load(row["embedding_path"], allow_pickle=False)
        for row in metadata_rows
    ]
    norms = np.asarray([np.linalg.norm(item) for item in embeddings], dtype=np.float64)
    exact_duplicates = len(embeddings) - len(
        {item.tobytes() for item in embeddings}
    )
    values = [
        ("embedding_count", len(embeddings)),
        ("dimension", EMBEDDING_DIMENSION),
        ("finite_embeddings", sum(bool(np.isfinite(item).all()) for item in embeddings)),
        ("zero_vectors", sum(float(np.linalg.norm(item)) <= 1e-12 for item in embeddings)),
        ("exact_duplicate_vectors", exact_duplicates),
        ("norm_min", float(norms.min()) if len(norms) else ""),
        ("norm_max", float(norms.max()) if len(norms) else ""),
        ("norm_mean", float(norms.mean()) if len(norms) else ""),
        ("norm_median", float(np.median(norms)) if len(norms) else ""),
        ("norm_std", float(norms.std()) if len(norms) else ""),
    ]
    _write_csv(
        METRICS_ROOT / "embedding_quality_audit.csv",
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in values],
    )


def _write_split_summary(metadata_rows: list[dict[str, Any]]) -> None:
    split_rows: list[dict[str, Any]] = []
    for split in ("research_train", "research_validation", "research_test"):
        subset = [row for row in metadata_rows if row["research_split"] == split]
        split_rows.append({
            "research_split": split,
            "embedded_images": len(subset),
            "embedded_identities": len({row["identity_id"] for row in subset}),
        })
    _write_csv(
        METRICS_ROOT / "embedding_split_summary.csv",
        ["research_split", "embedded_images", "embedded_identities"],
        split_rows,
    )


def _write_extraction_summary(
    result: dict[str, Any], input_rows: list[dict[str, str]]
) -> None:
    embedded_names = {row["filename"] for row in result["metadata_rows"]}
    pair_rows = _rows_from_csv(PAIR_METADATA)
    eligible_pairs = [
        row for row in pair_rows if row["preprocessing_eligible"].lower() == "true"
    ]
    both_embedded = [
        row for row in eligible_pairs
        if row["image_a"] in embedded_names and row["image_b"] in embedded_names
    ]
    by_identity: Counter[str] = Counter(
        row["identity_id"] for row in result["metadata_rows"]
    )
    summary = [
        ("total_stage4_rows", len(input_rows)),
        ("successful_embeddings", len(result["metadata_rows"])),
        ("failed_embeddings", len(result["failures"])),
        ("cache_hits", result["cache_hits"]),
        ("new_embeddings", result["new_embeddings"]),
        ("eligible_longitudinal_pairs", len(eligible_pairs)),
        ("pairs_with_both_embeddings", len(both_embedded)),
        ("pairs_excluded_due_to_embedding_failure", len(eligible_pairs) - len(both_embedded)),
        ("identities_with_at_least_2_embeddings", sum(value >= 2 for value in by_identity.values())),
        ("identities_with_at_least_3_embeddings", sum(value >= 3 for value in by_identity.values())),
        ("identities_with_at_least_4_embeddings", sum(value >= 4 for value in by_identity.values())),
        ("identities_with_at_least_5_embeddings", sum(value >= 5 for value in by_identity.values())),
        ("elapsed_seconds", result["elapsed_seconds"]),
    ]
    _write_csv(
        METRICS_ROOT / "embedding_extraction_summary.csv",
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in summary],
    )


def _write_report(manifest: dict[str, Any], result: dict[str, Any]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    provider = ", ".join(manifest["execution_provider"])
    report = f"""# Stage 5 — Embedding Extraction

## Objective

Extract fixed InsightFace `buffalo_l` ArcFace representations from successful
Stage 4 aligned MORPH images. This stage does not calculate drift, thresholds,
verification metrics, Stability Index values, or re-enrollment decisions.

## Configuration

- Version: `{STAGE5_VERSION}`
- Model: `{manifest["model"]}`
- Input: Stage 4 aligned `{manifest["input_size"]}` BGR images
- Embedding dimension: `{manifest["embedding_dimension"]}`
- Dtype: `{manifest["dtype"]}`
- Normalization: `{manifest["normalization"]}`
- Execution provider: `{provider}`

## Extraction results

- Stage 4 input images: `{manifest["total_input_images"]}`
- Successful Stage 4 images: `{manifest["successful_preprocessing_images"]}`
- Attempts: `{manifest["embedding_attempts"]}`
- Successful embeddings: `{manifest["successful_embeddings"]}`
- Failed embeddings: `{manifest["failed_embeddings"]}`
- Cache hits: `{manifest["cache_hits"]}`
- New embeddings: `{manifest["new_embeddings"]}`
- Elapsed seconds: `{result["elapsed_seconds"]:.3f}`

## Traceability and safeguards

Canonical per-image embeddings are stored under `embeddings/morph/embeddings/`
and mapped through `data/metadata/morph_embedding_metadata.csv`. Failed Stage 4
images are excluded by status rather than deleted from Stage 4 metadata.
Embedding extraction does not fit downstream parameters or use test identities
for tuning.

## Reproducibility

The manifest records package versions, provider configuration, Python version,
timestamp, and Git commit when available. The required smoke test verified
512-dimensional float32 outputs, finite values, L2 norms near one, and repeated
inference reproducibility before the full run.

## Readiness for Stage 6

Embeddings are numerical representations only. Drift analysis remains a
separate Stage 6 operation governed by the frozen `stage3-v1` protocol.
"""
    (REPORT_ROOT / "embedding_extraction_report.md").write_text(report, encoding="utf-8")
