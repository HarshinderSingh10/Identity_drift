import json

import pytest

from src.experiments.protocol_definition import (
    age_gap_group,
    select_analytical_reference,
    sort_observations,
    validate_experiment_registry,
    validate_protocol_config,
)


def test_protocol_json_has_valid_frozen_schema() -> None:
    with open("results/metrics/protocol_definition.json", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_protocol_config(config)
    assert config["embedding_configuration"]["status"].startswith("planned")


def test_age_gap_assignment_matches_stage2_metadata() -> None:
    assert [age_gap_group(value) for value in (0, 2, 3, 4, 5, 9, 10, 19, 20)] == [
        "0-2", "0-2", "3-4", "3-4", "5-9", "5-9", "10-19", "10-19", "20+"
    ]
    with pytest.raises(ValueError):
        age_gap_group(-1)


def test_reference_and_observation_order_are_deterministic() -> None:
    observations = [
        {"filename": "id_1M20.JPG", "filename_age": 20},
        {"filename": "id_0M20.JPG", "filename_age": 20},
        {"filename": "id_2M18.JPG", "filename_age": 18},
    ]
    assert select_analytical_reference(observations)["filename"] == "id_2M18.JPG"
    assert [row["filename"] for row in sort_observations(observations)] == [
        "id_2M18.JPG", "id_0M20.JPG", "id_1M20.JPG"
    ]


def test_registry_rejects_test_as_threshold_selection_split() -> None:
    with open("results/metrics/protocol_definition.json", encoding="utf-8") as stream:
        config = json.load(stream)
    row = dict(config["experiments"][0])
    row["validation_split"] = "research_test"
    with pytest.raises(ValueError):
        validate_experiment_registry([row] * 10)
