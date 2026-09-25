from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import load_dataset_pairs
from src.drift.drift_metrics import cosine_similarity, identity_drift
from src.experiments.drift_experiment import (
    compute_pairwise_metrics,
    extract_embeddings_for_dataset,
    unique_image_paths,
)


def test_positive_pairs_are_loaded_correctly() -> None:
    pairs = load_dataset_pairs("lfw", pair_type="positive")
    assert len(pairs) == 3000
    assert all(pair.pair_type == "positive" for pair in pairs)
    assert pairs[0].image_a.name == "0.bmp"
    assert pairs[0].image_b.name == "1.bmp"


def test_unique_image_extraction_works() -> None:
    pairs = load_dataset_pairs("agedb_30", pair_type="positive")[:10]
    unique = unique_image_paths(pairs)
    assert len(unique) == 20
    assert [item.name for item in unique[:4]] == ["0.bmp", "1.bmp", "2.bmp", "3.bmp"]


def test_duplicate_image_paths_are_deduplicated() -> None:
    pairs = load_dataset_pairs("calfw", pair_type="positive")[:6]
    unique = unique_image_paths(pairs)
    assert len(unique) == 12
    assert len({str(path) for path in unique}) == len(unique)


def test_embeddings_are_cached(tmp_path) -> None:
    pairs = load_dataset_pairs("lfw", pair_type="positive")[:5]

    class DummyExtractor:
        def __init__(self):
            self.calls = 0

        def extract_embedding(self, path: str):
            self.calls += 1
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    extractor = DummyExtractor()
    embeddings, generated, hits, failures = extract_embeddings_for_dataset(
        "lfw",
        pairs,
        force=True,
        extractor=extractor,
        cache_path=tmp_path / "lfw_embeddings.npz",
    )

    assert generated == 10
    assert hits == 0
    assert len(failures) == 0
    assert len(embeddings) == 10


def test_cached_embeddings_are_reused(tmp_path, monkeypatch) -> None:
    pairs = load_dataset_pairs("lfw", pair_type="positive")[:5]

    class DummyExtractor:
        def __init__(self):
            self.calls = 0

        def extract_embedding(self, path: str):
            self.calls += 1
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    extractor = DummyExtractor()
    cache_path = tmp_path / "lfw_embeddings.npz"
    first = extract_embeddings_for_dataset(
        "lfw", pairs, force=True, extractor=extractor, cache_path=cache_path
    )
    second = extract_embeddings_for_dataset(
        "lfw", pairs, force=False, extractor=extractor, cache_path=cache_path
    )

    assert first[1] == 10
    assert second[2] == 10
    assert extractor.calls == 10


def test_incompatible_aligned_embedding_cache_is_ignored(tmp_path) -> None:
    pairs = load_dataset_pairs("lfw", pair_type="positive")[:1]
    cache_path = tmp_path / "lfw_embeddings.npz"
    paths = [str(pairs[0].image_a), str(pairs[0].image_b)]
    np.savez_compressed(
        cache_path,
        paths=np.asarray(paths, dtype=str),
        embeddings=np.zeros((2, 3), dtype=np.float32),
    )

    class DummyAlignedExtractor:
        def __init__(self):
            self.calls = 0

        def extract_aligned_embedding(self, image):
            self.calls += 1
            return np.ones(512, dtype=np.float32)

    extractor = DummyAlignedExtractor()
    embeddings, generated, hits, failures = extract_embeddings_for_dataset(
        "lfw",
        pairs,
        extractor=extractor,
        cache_path=cache_path,
    )

    assert generated == 2
    assert hits == 0
    assert failures == []
    assert extractor.calls == 2
    assert all(value.shape == (512,) for value in embeddings.values())


def test_pairwise_similarity_matches_drift_metrics() -> None:
    embeddings = {
        "img_a": np.array([1.0, 0.0], dtype=np.float32),
        "img_b": np.array([0.0, 1.0], dtype=np.float32),
    }
    pairs = [type("Pair", (), {"image_a": Path("img_a"), "image_b": Path("img_b")})()]

    rows = compute_pairwise_metrics("lfw", pairs, embeddings)
    assert len(rows) == 1
    assert rows[0]["cosine_similarity"] == pytest.approx(cosine_similarity(embeddings["img_a"], embeddings["img_b"]))
    assert rows[0]["identity_drift"] == pytest.approx(identity_drift(embeddings["img_a"], embeddings["img_b"]))
    assert rows[0]["cosine_distance"] == pytest.approx(1.0 - rows[0]["cosine_similarity"])


def test_result_rows_contain_required_fields() -> None:
    embeddings = {
        "img_a": np.array([1.0, 0.0], dtype=np.float32),
        "img_b": np.array([0.0, 1.0], dtype=np.float32),
    }
    pairs = [type("Pair", (), {"image_a": Path("img_a"), "image_b": Path("img_b")})()]

    rows = compute_pairwise_metrics("lfw", pairs, embeddings)
    required = {"dataset", "image_a", "image_b", "cosine_similarity", "cosine_distance", "identity_drift", "euclidean_distance"}
    assert set(rows[0].keys()) == required


def test_failed_image_extraction_is_recorded_without_crashing(tmp_path) -> None:
    class DummyExtractor:
        def extract_embedding(self, path: str):
            raise RuntimeError("simulated failure")

    pairs = load_dataset_pairs("lfw", pair_type="positive")[:2]
    _, generated, hits, failures = extract_embeddings_for_dataset(
        "lfw",
        pairs,
        force=True,
        extractor=DummyExtractor(),
        cache_path=tmp_path / "lfw_embeddings.npz",
    )

    assert generated == 4
    assert hits == 0
    assert len(failures) == 4
    assert all(record["dataset"] == "lfw" for record in failures)
    assert all("simulated failure" in record["error"] for record in failures)


def test_pilot_mode_processes_exactly_100_pairs_per_dataset() -> None:
    from src.experiments.drift_experiment import DEFAULT_DATASETS, _dataset_pairs_for_run

    for dataset_name in DEFAULT_DATASETS:
        pairs = _dataset_pairs_for_run(dataset_name, pilot=True)
        assert len(pairs) == 100
