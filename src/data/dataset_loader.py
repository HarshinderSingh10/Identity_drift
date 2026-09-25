"""Utilities for loading and validating the validation annotation pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "validation"

DATASET_CONFIG: dict[str, dict[str, str]] = {
    "agedb_30": {
        "annotation": "agedb_30_ann.txt",
        "image_dir": "agedb_30_112x112",
    },
    "calfw": {
        "annotation": "calfw_ann.txt",
        "image_dir": "calfw_112x112",
    },
    "cplfw": {
        "annotation": "cplfw_ann.txt",
        "image_dir": "cplfw_112x112",
    },
    "lfw": {
        "annotation": "lfw_ann.txt",
        "image_dir": "lfw_112x112",
    },
}
SUPPORTED_DATASETS = tuple(DATASET_CONFIG)


@dataclass(frozen=True)
class DatasetPair:
    """One parsed annotation row mapped to absolute image paths."""

    dataset: str
    image_a: Path
    image_b: Path
    identity_a: str | None = None
    identity_b: str | None = None
    pair_type: str = "positive"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the pair."""

        return {
            "dataset": self.dataset,
            "image_a": str(self.image_a),
            "image_b": str(self.image_b),
            "identity_a": self.identity_a,
            "identity_b": self.identity_b,
            "pair_type": self.pair_type,
            "metadata": self.metadata,
        }


def list_supported_datasets() -> list[str]:
    """Return the dataset names supported by this loader."""

    return list(SUPPORTED_DATASETS)


def load_dataset_pairs(dataset_name: str, pair_type: str = "positive") -> list[DatasetPair]:
    """Return annotated image pairs for a dataset.

    Parameters
    ----------
    dataset_name:
        One of the supported validation datasets: agedb_30, calfw, cplfw, lfw.
    pair_type:
        ``positive``, ``negative``, or ``all``. The default is ``positive``.

    The annotation files are plain-text, whitespace-delimited records of the form:
    ``label image_a image_b`` where label is 1 for same identity and 0 for different
    identity. The loader preserves annotation order and resolves both image paths to
    absolute files under ``data/raw/validation``.
    """

    config = _require_dataset_config(dataset_name)
    annotation_path = DATA_ROOT / config["annotation"]
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")

    pair_filter = _normalize_pair_type(pair_type)
    records: list[DatasetPair] = []
    missing_paths: list[str] = []

    for line_number, raw_line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue

        fields = raw_line.strip().split()
        if len(fields) != 3:
            raise ValueError(
                f"Malformed annotation record in {annotation_path.name} at line {line_number}: "
                f"expected 3 fields, got {len(fields)}."
            )

        try:
            label = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"Malformed annotation record in {annotation_path.name} at line {line_number}: "
                f"invalid label '{fields[0]}'."
            ) from exc

        if label not in (0, 1):
            raise ValueError(
                f"Malformed annotation record in {annotation_path.name} at line {line_number}: "
                f"label must be 0 or 1, got '{fields[0]}'."
            )

        pair_kind = "positive" if label == 1 else "negative"
        if pair_filter not in ("all", pair_kind):
            continue

        image_a = _resolve_image_reference(dataset_name, fields[1])
        image_b = _resolve_image_reference(dataset_name, fields[2])

        if not image_a.exists() or not image_b.exists():
            missing = []
            if not image_a.exists():
                missing.append(str(image_a))
            if not image_b.exists():
                missing.append(str(image_b))
            missing_paths.append(
                f"{dataset_name}:{line_number}:{'; '.join(missing)}"
            )
            continue

        records.append(
            DatasetPair(
                dataset=dataset_name,
                image_a=image_a,
                image_b=image_b,
                identity_a=None,
                identity_b=None,
                pair_type=pair_kind,
                metadata={
                    "annotation_file": annotation_path.name,
                    "line_number": line_number,
                    "label": label,
                    "image_a_reference": fields[1],
                    "image_b_reference": fields[2],
                },
            )
        )

    if missing_paths:
        raise FileNotFoundError(
            "Missing image references in annotation file for "
            f"{dataset_name}: {'; '.join(missing_paths)}"
        )

    return records


def get_dataset_summary(dataset_name: str) -> dict[str, Any]:
    """Return a summary of the dataset annotations and file references."""

    config = _require_dataset_config(dataset_name)
    annotation_path = DATA_ROOT / config["annotation"]
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")

    annotation_records = 0
    positive_pairs = 0
    negative_pairs = 0
    malformed_records = 0
    missing_references: list[str] = []
    unique_images: set[str] = set()

    for line_number, raw_line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue

        fields = raw_line.strip().split()
        if len(fields) != 3:
            malformed_records += 1
            continue

        try:
            label = int(fields[0])
        except ValueError:
            malformed_records += 1
            continue

        if label not in (0, 1):
            malformed_records += 1
            continue

        annotation_records += 1
        unique_images.add(fields[1])
        unique_images.add(fields[2])

        if label == 1:
            positive_pairs += 1
        else:
            negative_pairs += 1

        image_a = _resolve_image_reference(dataset_name, fields[1])
        image_b = _resolve_image_reference(dataset_name, fields[2])

        if not image_a.exists() or not image_b.exists():
            missing_references.append(
                f"line {line_number}: {fields[1]} / {fields[2]}"
            )

    returned_positive_pairs = 0
    for pair in load_dataset_pairs(dataset_name, pair_type="positive"):
        returned_positive_pairs += 1

    return {
        "dataset": dataset_name,
        "annotation_records": annotation_records,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "returned_positive_pairs": returned_positive_pairs,
        "unique_identities": None,
        "unique_images": len(unique_images),
        "missing_image_references": len(missing_references),
        "malformed_annotation_records": malformed_records,
    }


def _require_dataset_config(dataset_name: str) -> dict[str, str]:
    """Return the configured annotation and image directory mapping."""

    try:
        return DATASET_CONFIG[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {', '.join(SUPPORTED_DATASETS)}."
        ) from exc


def _normalize_pair_type(pair_type: str) -> str:
    """Normalize pair_type to positive/negative/all."""

    normalized = (pair_type or "").strip().lower()
    if normalized not in {"positive", "negative", "all"}:
        raise ValueError("pair_type must be one of: positive, negative, all")
    return normalized


def _resolve_image_reference(dataset_name: str, image_reference: str) -> Path:
    """Resolve an annotation image reference to an absolute file path."""

    relative_path = Path(image_reference)
    candidate = (DATA_ROOT / relative_path).resolve()
    if candidate.exists():
        return candidate

    dataset_dir = DATA_ROOT / DATASET_CONFIG[dataset_name]["image_dir"]
    if relative_path.parts and relative_path.parts[0] == dataset_dir.name:
        candidate = (dataset_dir / Path(*relative_path.parts[1:])).resolve()
        if candidate.exists():
            return candidate

    candidate = (dataset_dir / relative_path.name).resolve()
    return candidate
