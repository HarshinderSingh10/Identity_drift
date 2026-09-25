"""Validation helpers for the frozen Stage 3 experimental protocol."""

from __future__ import annotations

from typing import Any, Iterable

REQUIRED_EXPERIMENT_IDS = (
    "EXP-A",
    "EXP-B",
    "EXP-C",
    "EXP-D",
    "EXP-E",
    "EXP-F",
    "EXP-G",
    "EXP-H",
    "EXP-I",
    "EXP-J",
)
VALID_SPLITS = {"research_train", "research_validation", "research_test"}
VALID_STATUSES = {"planned_after_embedding_extraction", "blocked", "future"}


def age_gap_group(age_gap: int) -> str:
    """Return the frozen Stage 3 age-gap category."""

    if age_gap < 0:
        raise ValueError("age_gap must be non-negative")
    if age_gap <= 2:
        return "0-2"
    if age_gap <= 4:
        return "3-4"
    if age_gap <= 9:
        return "5-9"
    if age_gap <= 19:
        return "10-19"
    return "20+"


def select_analytical_reference(observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select the youngest filename-age observation with deterministic tie-breaking."""

    candidates = list(observations)
    if not candidates:
        raise ValueError("At least one observation is required")
    required = {"filename", "filename_age"}
    if any(not required.issubset(observation) for observation in candidates):
        raise ValueError("Observations must contain filename and filename_age")
    return min(candidates, key=lambda row: (int(row["filename_age"]), str(row["filename"])))


def sort_observations(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic filename-age then filename ordering."""

    return sorted(
        observations,
        key=lambda row: (int(row["filename_age"]), str(row["filename"])),
    )


def validate_experiment_registry(rows: Iterable[dict[str, Any]]) -> None:
    """Validate the required experiment registry shape and split roles."""

    rows = list(rows)
    observed = {row.get("experiment_id") for row in rows}
    missing = set(REQUIRED_EXPERIMENT_IDS) - observed
    if missing:
        raise ValueError(f"Missing experiment IDs: {sorted(missing)}")
    required_fields = {
        "experiment_id",
        "research_question",
        "dataset",
        "population",
        "unit_of_analysis",
        "input_metadata",
        "primary_metric",
        "secondary_metrics",
        "development_split",
        "validation_split",
        "test_split",
        "statistical_method",
        "threshold_source",
        "status",
    }
    for row in rows:
        missing_fields = required_fields - set(row)
        if missing_fields:
            raise ValueError(
                f"{row.get('experiment_id', '<unknown>')} missing fields: "
                f"{sorted(missing_fields)}"
            )
        if row["status"] not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {row['status']}")
        for field in ("development_split", "validation_split", "test_split"):
            if row[field] not in VALID_SPLITS:
                raise ValueError(f"Invalid split in {field}: {row[field]}")
        if not (
            row["development_split"] == "research_train"
            and row["validation_split"] == "research_validation"
            and row["test_split"] == "research_test"
        ):
            raise ValueError(f"Invalid split roles for {row['experiment_id']}")


def validate_protocol_config(config: dict[str, Any]) -> None:
    """Validate the core versioned protocol configuration."""

    required = {
        "protocol_version",
        "random_seed",
        "primary_dataset",
        "secondary_dataset",
        "age_variable",
        "primary_drift_metric",
        "embedding_configuration",
        "split_roles",
        "age_gap_groups",
        "experiments",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Protocol missing fields: {sorted(missing)}")
    if config["primary_dataset"] != "MORPH":
        raise ValueError("MORPH must be the primary dataset")
    if config["secondary_dataset"] != "AgeDB":
        raise ValueError("AgeDB must be the secondary dataset")
    if config["primary_drift_metric"] != "cosine_distance":
        raise ValueError("Cosine distance must remain the primary drift metric")
    embedding_configuration = config["embedding_configuration"]
    if embedding_configuration.get("embedding_dimension") != 512:
        raise ValueError("The protocol requires 512-dimensional embeddings")
    if embedding_configuration.get("normalization") != "L2 normalization before cosine comparison":
        raise ValueError("The protocol requires L2-normalized embeddings")
    validate_experiment_registry(config["experiments"])
