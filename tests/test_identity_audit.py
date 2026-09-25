from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.identity_audit import (
    build_positive_graph,
    connected_components,
    file_sha256,
    find_graph_contradictions,
    hash_groups_for_paths,
    _duplicate_restricted_rows,
    _verification_metrics,
)


class Pair:
    def __init__(self, left: str, right: str) -> None:
        self.image_a = Path(left)
        self.image_b = Path(right)


def test_positive_graph_connected_components() -> None:
    graph = build_positive_graph(
        [Pair("a", "b"), Pair("b", "c"), Pair("x", "y")]
    )
    components = connected_components(graph)
    assert {frozenset(component) for component in components} == {
        frozenset({"a", "b", "c"}),
        frozenset({"x", "y"}),
    }


def test_graph_flags_negative_edge_inside_positive_component() -> None:
    contradictions = find_graph_contradictions(
        [Pair("a", "b"), Pair("b", "c")],
        [Pair("a", "c"), Pair("x", "y")],
    )
    assert len(contradictions) == 1
    assert contradictions[0]["image_a"] == "a"
    assert contradictions[0]["image_b"] == "c"


def test_graph_does_not_invent_singletons() -> None:
    graph = build_positive_graph([])
    assert connected_components(graph) == []


def test_exact_duplicate_detection(tmp_path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    third = tmp_path / "third.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    third.write_bytes(b"different")

    assert file_sha256(first) == file_sha256(second)
    groups = hash_groups_for_paths([first, second, third])
    assert sorted(len(paths) for paths in groups.values()) == [1, 2]


def test_duplicate_restriction_excludes_reused_content() -> None:
    rows = [
        {"pair_type": "positive", "image_a": "a", "image_b": "b", "cosine_similarity": "0.9"},
        {"pair_type": "negative", "image_a": "c", "image_b": "d", "cosine_similarity": "0.1"},
    ]
    restricted, excluded = _duplicate_restricted_rows(
        "demo",
        rows,
        {"a": "hash1", "b": "hash2", "c": "hash1", "d": "hash3"},
    )
    assert excluded == 2
    assert restricted == []


def test_empty_verification_metrics_are_explicit() -> None:
    result = _verification_metrics([])
    assert result["positive_pairs"] == 0
    assert result["negative_pairs"] == 0
    assert np.isnan(result["roc_auc"])
