from pathlib import Path

import cv2
import numpy as np
import pytest

from src.experiments.preprocess_morph import (
    PREPROCESSING_VERSION,
    deterministic_processed_path,
    detect_and_align,
    load_image,
)


def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")


def test_load_image_rejects_invalid_image(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(path)


def test_processed_path_is_deterministic() -> None:
    raw = Path(r"E:\PythonProject\Identity_drift\data\raw\Morph\Images\Train\00013_00M19.JPG")
    first = deterministic_processed_path(raw)
    second = deterministic_processed_path(raw)
    assert first == second
    assert first.suffix == ".png"


def test_detector_marks_invalid_arrays_without_selecting_face() -> None:
    invalid = np.zeros((112, 112), dtype=np.uint8)
    aligned, count, selected, confidence, status, reason = detect_and_align(
        object(), invalid
    )
    assert aligned is invalid
    assert count == 0
    assert selected is None
    assert confidence is None
    assert status == "invalid_image"
    assert reason


def test_preprocessing_version_is_stage4() -> None:
    assert PREPROCESSING_VERSION == "stage4-v1"
