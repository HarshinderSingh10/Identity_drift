"""Deterministic MORPH face detection and ArcFace-aligned image preprocessing."""

from __future__ import annotations

import csv
import hashlib
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "morph"
PROCESSED_IMAGES_ROOT = PROCESSED_ROOT / "images"
RESULTS_ROOT = PROJECT_ROOT / "results"
METRICS_ROOT = RESULTS_ROOT / "metrics"
PLOTS_ROOT = RESULTS_ROOT / "plots" / "preprocessing"
REPORT_ROOT = PROJECT_ROOT / "experiments" / "04_preprocessing"
PAIR_METADATA = METADATA_ROOT / "morph_longitudinal_pairs.csv"
IMAGE_METADATA = METADATA_ROOT / "morph_research_images.csv"
PREPROCESSING_VERSION = "stage4-v1"
MODEL_NAME = "buffalo_l"
OUTPUT_SIZE = 112
# MORPH images are tightly cropped 112x112 faces. 160x160 is the smallest
# detector configuration that preserves the SCRFD output shape and detects
# the sample images reliably.
DETECTION_SIZE = (160, 160)
VALID_STATUSES = {
    "success",
    "no_face",
    "multiple_faces",
    "image_unreadable",
    "alignment_failed",
    "invalid_image",
}
LOGGER = logging.getLogger(__name__)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_image(path: Path) -> np.ndarray:
    """Load a BGR uint8 image or raise a clear validation error."""

    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 3-channel image: {path}")
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Image is empty: {path}")
    return image


def deterministic_processed_path(raw_path: Path) -> Path:
    """Map a raw MORPH path to a stable processed relative path."""

    relative = raw_path.relative_to(PROJECT_ROOT / "data" / "raw" / "Morph" / "Images")
    return PROCESSED_IMAGES_ROOT / relative.with_suffix(".png")


def detect_and_align(
    detector: FaceAnalysis,
    image: np.ndarray,
) -> tuple[np.ndarray, int, int | None, float | None, str, str]:
    """Detect and align exactly one face without invoking recognition."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return image, 0, None, None, "invalid_image", "Expected uint8 BGR image"
    bboxes, landmarks = detector.det_model.detect(image, max_num=0, metric="default")
    face_count = int(bboxes.shape[0])
    if face_count == 0:
        return image, 0, None, None, "no_face", "No face detected"
    if face_count > 1:
        return image, face_count, None, None, "multiple_faces", "Multiple faces detected; no face selected"
    if landmarks is None or len(landmarks) != 1:
        return image, face_count, None, float(bboxes[0, 4]), "alignment_failed", "Landmarks unavailable"
    try:
        aligned = norm_crop(image, landmarks[0], image_size=OUTPUT_SIZE, mode="arcface")
    except Exception as exc:
        return image, face_count, 0, float(bboxes[0, 4]), "alignment_failed", str(exc)
    if aligned.shape != (OUTPUT_SIZE, OUTPUT_SIZE, 3) or aligned.dtype != np.uint8:
        return image, face_count, 0, float(bboxes[0, 4]), "alignment_failed", "Unexpected aligned output"
    return aligned, face_count, 0, float(bboxes[0, 4]), "success", ""


def _provider_list() -> list[str]:
    # The installed CUDA/cuDNN runtime is unavailable on this host. Use the
    # deterministic CPU detector rather than silently retrying failed CUDA.
    return ["CPUExecutionProvider"]


def _detector() -> FaceAnalysis:
    providers = _provider_list()
    detector = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection"],
        providers=providers,
    )
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    detector.prepare(ctx_id=ctx_id, det_thresh=0.5, det_size=DETECTION_SIZE)
    return detector


def _metadata_rows() -> list[dict[str, str]]:
    with IMAGE_METADATA.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _process_row(row: dict[str, str], detector: FaceAnalysis, force: bool) -> dict[str, Any]:
    raw_path = Path(row["filepath"])
    processed_path = deterministic_processed_path(raw_path)
    base = {
        "dataset": row["dataset"],
        "identity_id": row["identity_id"],
        "filename": row["filename"],
        "raw_filepath": str(raw_path),
        "processed_filepath": str(processed_path),
        "research_split": row["research_split"],
        "filename_age": row["filename_age"],
        "csv_scaled_age": row["csv_scaled_age"],
        "image_width": "",
        "image_height": "",
        "image_channels": "",
        "face_count": "",
        "selected_face_index": "",
        "detection_confidence": "",
        "alignment_status": "not_attempted",
        "preprocessing_status": "",
        "failure_reason": "",
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_name": MODEL_NAME,
        "input_color_order": "BGR",
        "output_color_order": "BGR",
        "output_width": OUTPUT_SIZE,
        "output_height": OUTPUT_SIZE,
        "output_dtype": "uint8",
        "model_normalization": "ArcFace model-internal pixel normalization is deferred to Stage 5; no embedding normalization performed",
    }
    if processed_path.exists() and not force:
        try:
            processed = load_image(processed_path)
            base.update({
                "image_width": processed.shape[1],
                "image_height": processed.shape[0],
                "image_channels": processed.shape[2],
                "face_count": 1,
                "selected_face_index": 0,
                "alignment_status": "cached",
                "preprocessing_status": "success",
            })
            return base
        except ValueError:
            processed_path.unlink()
    try:
        image = load_image(raw_path)
    except FileNotFoundError as exc:
        base.update({"preprocessing_status": "image_unreadable", "failure_reason": str(exc)})
        return base
    except ValueError as exc:
        base.update({"preprocessing_status": "invalid_image", "failure_reason": str(exc)})
        return base
    base.update({
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "image_channels": image.shape[2],
    })
    aligned, count, selected, confidence, alignment_status, failure = detect_and_align(detector, image)
    base.update({
        "face_count": count,
        "selected_face_index": "" if selected is None else selected,
        "detection_confidence": "" if confidence is None else f"{confidence:.8f}",
        "alignment_status": alignment_status,
        "preprocessing_status": alignment_status,
        "failure_reason": failure,
    })
    if alignment_status == "success":
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(processed_path), aligned):
            base["preprocessing_status"] = "alignment_failed"
            base["failure_reason"] = "OpenCV could not write processed image"
    return base


def _build_pair_eligibility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["filename"]: row for row in rows}
    with PAIR_METADATA.open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    output = []
    for pair in pairs:
        left, right = by_name.get(pair["image_a"]), by_name.get(pair["image_b"])
        left_status = left["preprocessing_status"] if left else "missing_metadata"
        right_status = right["preprocessing_status"] if right else "missing_metadata"
        eligible = left_status == "success" and right_status == "success"
        reasons = []
        if left_status != "success":
            reasons.append(f"image_a:{left_status}")
        if right_status != "success":
            reasons.append(f"image_b:{right_status}")
        output.append({
            "pair_id": pair["pair_id"],
            "image_a": pair["image_a"],
            "image_b": pair["image_b"],
            "preprocessing_eligible": str(eligible).lower(),
            "image_a_status": left_status,
            "image_b_status": right_status,
            "exclusion_reason": ";".join(reasons),
            "research_split": pair["research_split"],
        })
    return output


def _write_plots(rows: list[dict[str, Any]]) -> None:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["preprocessing_status"] for row in rows)
    plt.figure()
    plt.bar(status_counts.keys(), status_counts.values())
    plt.ylabel("Images")
    plt.title("MORPH preprocessing status")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "preprocessing_success_failure.png", dpi=150)
    plt.close()

    face_counts = Counter(row["face_count"] for row in rows)
    plt.figure()
    plt.bar(face_counts.keys(), face_counts.values())
    plt.xlabel("Detected face count")
    plt.ylabel("Images")
    plt.title("MORPH face-count distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "face_count_distribution.png", dpi=150)
    plt.close()

    ages = [int(row["filename_age"]) for row in rows if row["preprocessing_status"] == "success"]
    failures = [int(row["filename_age"]) for row in rows if row["preprocessing_status"] != "success"]
    plt.figure()
    plt.hist([ages, failures], bins=range(16, 79), label=["success", "failure"])
    plt.xlabel("Filename age")
    plt.ylabel("Images")
    plt.title("Preprocessing outcomes by filename age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "failures_by_age.png", dpi=150)
    plt.close()

    split_status: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split_status[row["research_split"]][row["preprocessing_status"]] += 1
    splits = sorted(split_status)
    statuses = sorted(status_counts)
    plt.figure()
    bottom = np.zeros(len(splits))
    for status in statuses:
        values = np.array([split_status[split][status] for split in splits])
        plt.bar(splits, values, bottom=bottom, label=status)
        bottom += values
    plt.ylabel("Images")
    plt.title("Preprocessing outcomes by research split")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "failures_by_split.png", dpi=150)
    plt.close()


def _write_examples(rows: list[dict[str, Any]]) -> None:
    examples_root = PLOTS_ROOT / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)
    selected = []
    for status in ("success", "no_face", "multiple_faces", "invalid_image", "alignment_failed"):
        selected.extend(row for row in rows if row["preprocessing_status"] == status)
    selected = sorted(selected, key=lambda row: (row["preprocessing_status"], row["filename"]))[:20]
    for row in selected:
        if row["preprocessing_status"] == "success":
            image = cv2.imread(row["processed_filepath"], cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(row["raw_filepath"], cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imwrite(str(examples_root / f"{row['preprocessing_status']}_{row['filename']}.png"), image)


def run_preprocessing(force: bool = False) -> dict[str, Any]:
    """Run deterministic MORPH preprocessing and write all Stage 4 outputs."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = _metadata_rows()
    detector = _detector()
    output_rows = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        output_rows.append(_process_row(row, detector, force))
        if index % 250 == 0 or index == total:
            LOGGER.info("Processed %d/%d MORPH images", index, total)

    fields = list(output_rows[0])
    _write_csv(METADATA_ROOT / "morph_preprocessing.csv", fields, output_rows)
    eligibility = _build_pair_eligibility(output_rows)
    _write_csv(
        METADATA_ROOT / "morph_pair_preprocessing_eligibility.csv",
        list(eligibility[0]),
        eligibility,
    )

    status_counts = Counter(row["preprocessing_status"] for row in output_rows)
    face_counts = Counter(str(row["face_count"]) for row in output_rows)
    identity_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        identity_rows[row["identity_id"]].append(row)
    identity_coverage = []
    for identity, identity_images in sorted(identity_rows.items()):
        valid = sum(row["preprocessing_status"] == "success" for row in identity_images)
        identity_coverage.append({
            "identity_id": identity,
            "observation_count": len(identity_images),
            "valid_observations": valid,
            "all_observations_successful": str(valid == len(identity_images)).lower(),
            "at_least_2_valid": str(valid >= 2).lower(),
            "at_least_3_valid": str(valid >= 3).lower(),
            "at_least_4_valid": str(valid >= 4).lower(),
            "at_least_5_valid": str(valid >= 5).lower(),
            "lost_observations": len(identity_images) - valid,
            "longitudinally_usable": str(valid >= 2).lower(),
        })
    eligible_pairs = sum(row["preprocessing_eligible"] == "true" for row in eligibility)
    split_rows = []
    for split in ("research_train", "research_validation", "research_test"):
        split_images = [row for row in output_rows if row["research_split"] == split]
        split_rows.append({
            "split": split,
            "total_images": len(split_images),
            "successful_images": sum(row["preprocessing_status"] == "success" for row in split_images),
            "failed_images": sum(row["preprocessing_status"] != "success" for row in split_images),
        })
    summary_rows = [
        {"metric": "total_images", "value": total},
        {"metric": "successful_images", "value": status_counts["success"]},
        {"metric": "failed_images", "value": total - status_counts["success"]},
        {"metric": "failure_rate", "value": f"{(total - status_counts['success']) / total:.8f}"},
        {"metric": "single_face_images", "value": face_counts["1"]},
        {"metric": "multi_face_images", "value": sum(count for key, count in face_counts.items() if key.isdigit() and int(key) > 1)},
        {"metric": "zero_face_images", "value": face_counts["0"]},
        {"metric": "eligible_longitudinal_pairs", "value": eligible_pairs},
        {"metric": "ineligible_longitudinal_pairs", "value": len(eligibility) - eligible_pairs},
        {"metric": "identities_all_observations_successful", "value": sum(row["all_observations_successful"] == "true" for row in identity_coverage)},
        {"metric": "identities_at_least_2_valid", "value": sum(row["at_least_2_valid"] == "true" for row in identity_coverage)},
        {"metric": "identities_at_least_3_valid", "value": sum(row["at_least_3_valid"] == "true" for row in identity_coverage)},
        {"metric": "identities_at_least_4_valid", "value": sum(row["at_least_4_valid"] == "true" for row in identity_coverage)},
        {"metric": "identities_at_least_5_valid", "value": sum(row["at_least_5_valid"] == "true" for row in identity_coverage)},
    ]
    _write_csv(METRICS_ROOT / "preprocessing_summary.csv", ["metric", "value"], summary_rows)
    _write_csv(
        METRICS_ROOT / "preprocessing_failures.csv",
        fields,
        [row for row in output_rows if row["preprocessing_status"] != "success"],
    )
    _write_csv(METRICS_ROOT / "preprocessing_split_summary.csv", list(split_rows[0]), split_rows)
    _write_csv(METRICS_ROOT / "preprocessing_identity_coverage.csv", list(identity_coverage[0]), identity_coverage)
    _write_plots(output_rows)
    _write_examples(output_rows)

    report = f"""# Stage 4 - MORPH Face Preprocessing

## Scope

This stage performed deterministic image loading, face detection, landmark alignment, and processed-image storage. It did not call the ArcFace recognition inference API, calculate embeddings, calculate cosine similarity, calculate drift, run verification, choose thresholds, build a Stability Index, or implement re-enrollment.

## Configuration

- Version: `{PREPROCESSING_VERSION}`
- Detector/model package: `{MODEL_NAME}`, detection module only
- Execution provider: CPUExecutionProvider (CUDA/cuDNN was unavailable on this host)
- Detector input size: `{DETECTION_SIZE[0]}x{DETECTION_SIZE[1]}`
- Alignment: InsightFace `face_align.norm_crop`, `mode="arcface"`
- Output size: `{OUTPUT_SIZE}x{OUTPUT_SIZE}`
- Input/output array convention: OpenCV BGR, `uint8`, pixel range 0-255
- ArcFace recognition normalization: not applied here; model-internal preprocessing remains for Stage 5
- Multiple-face policy: no face is selected; status is `multiple_faces`

## Results

- Total images: {total}
- Successful: {status_counts["success"]}
- Failed: {total - status_counts["success"]}
- Failure rate: {(total - status_counts["success"]) / total:.8f}
- Zero-face images: {face_counts["0"]}
- Single-face images: {face_counts["1"]}
- Multiple-face images: {sum(count for key, count in face_counts.items() if key.isdigit() and int(key) > 1)}
- Eligible longitudinal pairs: {eligible_pairs}
- Ineligible longitudinal pairs: {len(eligibility) - eligible_pairs}

## Traceability and eligibility

Every MORPH image has a preprocessing metadata row, including failures. Processed paths preserve the original `Train`, `Validation`, or `Test` relative path under `data/processed/morph/images/`. The original Stage 2 pair file was not modified; pair eligibility is written separately.

## Split integrity

The preprocessing transformation does not reassign identities or images. The Stage 2 research split assignment is copied unchanged, and no preprocessing rule was tuned using test-set statistics.

## Raw-data and embedding safeguards

Raw MORPH, AgeDB images, and AgeDB protocol files were treated as read-only. No new `.npy` or `.npz` embedding files were created, and no embedding or similarity computation was performed.
"""
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "preprocessing_report.md").write_text(report, encoding="utf-8")
    return {
        "total_images": total,
        "successful_images": status_counts["success"],
        "failed_images": total - status_counts["success"],
        "eligible_pairs": eligible_pairs,
    }
