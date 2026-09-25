from __future__ import annotations

from pathlib import Path

import pytest

from src.data import get_dataset_summary, list_supported_datasets, load_dataset_pairs


SUPPORTED_DATASETS = ["agedb_30", "calfw", "cplfw", "lfw"]


def test_all_four_datasets_can_be_discovered() -> None:
    assert list_supported_datasets() == SUPPORTED_DATASETS


def test_annotation_files_parse_for_all_datasets() -> None:
    for dataset_name in SUPPORTED_DATASETS:
        positive_pairs = load_dataset_pairs(dataset_name, pair_type="positive")
        negative_pairs = load_dataset_pairs(dataset_name, pair_type="negative")

        assert len(positive_pairs) == 3000
        assert len(negative_pairs) == 3000
        assert all(pair.pair_type == "positive" for pair in positive_pairs)
        assert all(pair.pair_type == "negative" for pair in negative_pairs)
        assert all(pair.image_a.exists() for pair in positive_pairs)
        assert all(pair.image_b.exists() for pair in positive_pairs)


def test_positive_and_negative_pairs_are_separated() -> None:
    for dataset_name in SUPPORTED_DATASETS:
        all_pairs = load_dataset_pairs(dataset_name, pair_type="all")
        positives = load_dataset_pairs(dataset_name, pair_type="positive")
        negatives = load_dataset_pairs(dataset_name, pair_type="negative")

        assert len(all_pairs) == len(positives) + len(negatives)
        assert len(positives) == 3000
        assert len(negatives) == 3000
        assert {pair.pair_type for pair in all_pairs} == {"positive", "negative"}


def test_referenced_paths_resolve_to_existing_files() -> None:
    first_pair = load_dataset_pairs("lfw", pair_type="positive")[0]

    assert first_pair.image_a.exists()
    assert first_pair.image_b.exists()
    assert first_pair.image_a.name == "0.bmp"
    assert first_pair.image_b.name == "1.bmp"
    assert first_pair.dataset == "lfw"


def test_missing_images_are_detected(monkeypatch, tmp_path) -> None:
    import src.data.dataset_loader as dataset_loader

    dataset_dir = tmp_path / "demo_112x112"
    dataset_dir.mkdir()
    (dataset_dir / "0.bmp").write_bytes(b"fake")
    annotation_path = tmp_path / "demo_ann.txt"
    annotation_path.write_text("1 demo_112x112/0.bmp demo_112x112/999.bmp\n", encoding="utf-8")

    monkeypatch.setattr(dataset_loader, "DATA_ROOT", tmp_path)
    monkeypatch.setitem(
        dataset_loader.DATASET_CONFIG,
        "demo",
        {"annotation": "demo_ann.txt", "image_dir": "demo_112x112"},
    )

    with pytest.raises(FileNotFoundError, match="Missing image references"):
        dataset_loader.load_dataset_pairs("demo", pair_type="positive")


def test_dataset_names_are_correct() -> None:
    for dataset_name in SUPPORTED_DATASETS:
        summary = get_dataset_summary(dataset_name)
        assert summary["dataset"] == dataset_name


def test_original_raw_files_are_not_modified() -> None:
    annotation_file = Path(__file__).resolve().parents[1] / "data" / "raw" / "validation" / "lfw_ann.txt"
    original_bytes = annotation_file.read_bytes()

    load_dataset_pairs("lfw", pair_type="positive")

    assert annotation_file.read_bytes() == original_bytes


def test_returned_pairs_preserve_annotation_order() -> None:
    pairs = load_dataset_pairs("calfw", pair_type="positive")

    assert pairs[0].image_a.name == "0.bmp"
    assert pairs[0].image_b.name == "1.bmp"
    assert pairs[1].image_a.name == "2.bmp"
    assert pairs[1].image_b.name == "3.bmp"
    assert pairs[-1].image_a.name == "11398.bmp"
    assert pairs[-1].image_b.name == "11399.bmp"


def test_summary_statistics_are_consistent() -> None:
    summary = get_dataset_summary("agedb_30")

    assert summary["annotation_records"] == summary["positive_pairs"] + summary["negative_pairs"]
    assert summary["returned_positive_pairs"] == summary["positive_pairs"]
    assert summary["missing_image_references"] == 0
    assert summary["malformed_annotation_records"] == 0
    assert summary["unique_images"] == 12000
    assert summary["unique_identities"] is None
