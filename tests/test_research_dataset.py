from src.experiments.research_dataset import (
    age_gap_group,
    assign_identity_splits,
    build_longitudinal_pairs,
    parse_morph_filename,
    validate_identity_splits,
)


def test_parse_morph_filename_uses_filename_age() -> None:
    assert parse_morph_filename("00013_00M19.JPG") == ("00013", 19)


def test_age_gap_groups() -> None:
    assert [age_gap_group(value) for value in (0, 2, 3, 4, 5, 9, 10, 19, 20, 40)] == [
        "0-2", "0-2", "3-4", "3-4", "5-9", "5-9", "10-19", "10-19", "20+", "20+"
    ]


def test_identity_split_is_deterministic_and_disjoint() -> None:
    first = assign_identity_splits(["b", "a", "c", "d", "e", "f"], seed=42)
    second = assign_identity_splits(["f", "e", "d", "c", "b", "a"], seed=42)
    assert first == second
    validate_identity_splits(first)


def test_longitudinal_pairs_are_unique_and_deterministic() -> None:
    rows = [
        {"identity_id": "1", "filename": "1_0M20.JPG", "filename_age": 20, "research_split": "research_train"},
        {"identity_id": "1", "filename": "1_1M30.JPG", "filename_age": 30, "research_split": "research_train"},
        {"identity_id": "1", "filename": "1_2M35.JPG", "filename_age": 35, "research_split": "research_train"},
    ]
    pairs = build_longitudinal_pairs(rows)
    assert len(pairs) == 3
    assert len({(row["image_a"], row["image_b"]) for row in pairs}) == 3
    assert pairs[0]["age_gap_group"] == "10-19"
