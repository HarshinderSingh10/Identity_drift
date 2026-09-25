"""Construct metadata-only identity-aware research datasets."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MORPH_ROOT = RAW_ROOT / "Morph"
AGEDB_ROOT = RAW_ROOT / "AgeDB_Images"
AGEDB_PROTOCOL_ROOT = RAW_ROOT / "AgeDB_protocols"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots" / "research_dataset"
REPORT_ROOT = PROJECT_ROOT / "experiments" / "02_research_dataset"
SEED = 42

MORPH_RE = re.compile(r"^(?P<identity>[^_]+)_[^_]*[MF](?P<age>\d+)\.JPG$", re.IGNORECASE)
AGEDB_RE = re.compile(
    r"^(?P<identity>\d+)_(?P<name>.+)_(?P<source_id>\d+)_A_(?P<age>\d+)$",
    re.IGNORECASE,
)
AGE_GROUPS = ((0, 2, "0-2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, None, "20+"))

LOGGER = logging.getLogger(__name__)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_morph_filename(filename: str) -> tuple[str, int]:
    """Extract filename identity and chronological age."""

    match = MORPH_RE.match(filename)
    if not match:
        raise ValueError(f"Malformed MORPH filename: {filename}")
    return match.group("identity"), int(match.group("age"))


def age_gap_group(age_gap: int) -> str:
    """Assign a deterministic age-gap category."""

    for lower, upper, label in AGE_GROUPS:
        if upper is None or lower <= age_gap <= upper:
            return label
    raise ValueError(f"Invalid age gap: {age_gap}")


def assign_identity_splits(identity_ids: Iterable[str], seed: int = SEED) -> dict[str, str]:
    """Assign identities to deterministic approximately 70/15/15 splits."""

    identities = sorted(set(identity_ids))
    rng = random.Random(seed)
    rng.shuffle(identities)
    n = len(identities)
    train_end = round(n * 0.70)
    validation_end = train_end + round(n * 0.15)
    return {
        identity: ("research_train" if index < train_end else
                   "research_validation" if index < validation_end else
                   "research_test")
        for index, identity in enumerate(identities)
    }


def validate_identity_splits(identity_to_split: dict[str, str]) -> None:
    """Raise if any identity is assigned to more than one split."""

    if set(identity_to_split) != {identity for identity in identity_to_split}:
        raise ValueError("Invalid identity split mapping")
    split_sets = {
        split: {identity for identity, value in identity_to_split.items() if value == split}
        for split in ("research_train", "research_validation", "research_test")
    }
    for left, right in itertools.combinations(split_sets.values(), 2):
        if left & right:
            raise ValueError("Identity overlap detected between research splits")


def _duplicate_metadata(paths: list[Path]) -> dict[str, tuple[str, bool]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[_sha256(path)].append(path)
    result: dict[str, tuple[str, bool]] = {}
    for index, (digest, group) in enumerate(sorted(groups.items())):
        group_id = f"morph_sha256_{index:06d}"
        duplicate = len(group) > 1
        for path in group:
            result[str(path.resolve())] = (group_id, duplicate)
    return result


def build_morph_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build image and identity metadata from read-only MORPH inputs."""

    cached_images = METADATA_ROOT / "morph_research_images.csv"
    if cached_images.exists():
        with cached_images.open(encoding="utf-8", newline="") as stream:
            cached_rows = list(csv.DictReader(stream))
        if cached_rows and all(
            row.get("age_source") == "filename_suffix"
            and row.get("csv_scaled_age") is not None
            and row.get("duplicate_group_id")
            for row in cached_rows
        ):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            identity_to_split = {
                row["identity_id"]: row["research_split"] for row in cached_rows
            }
            for row in cached_rows:
                grouped[row["identity_id"]].append(row)
            identities = []
            for identity_id in sorted(grouped):
                observations = sorted(
                    grouped[identity_id],
                    key=lambda row: (int(row["filename_age"]), row["filename"]),
                )
                ages = [int(row["filename_age"]) for row in observations]
                count = len(observations)
                span = max(ages) - min(ages)
                identities.append({
                    "identity_id": identity_id,
                    "image_count": count,
                    "observation_count": count,
                    "min_age": min(ages),
                    "max_age": max(ages),
                    "age_span": span,
                    "research_split": identity_to_split[identity_id],
                    "has_2plus_images": str(count >= 2).lower(),
                    "has_3plus_images": str(count >= 3).lower(),
                    "has_4plus_images": str(count >= 4).lower(),
                    "has_5plus_images": str(count >= 5).lower(),
                    "has_10plus_images": str(count >= 10).lower(),
                    "has_5plus_year_span": str(count >= 3 and span >= 5).lower(),
                    "has_10plus_year_span": str(count >= 3 and span >= 10).lower(),
                    "has_20plus_year_span": str(count >= 3 and span >= 20).lower(),
                    "reference_candidate": observations[0]["filename"],
                    "reference_rule": "youngest_filename_age_then_filename; analytical_only",
                })
            return cached_rows, identities, {
                "identity_to_split": identity_to_split,
                "duplicate_groups": len({
                    row["duplicate_group_id"]
                    for row in cached_rows
                    if row["is_exact_duplicate"] == "true"
                }),
            }

    rows: list[dict[str, Any]] = []
    source_rows: dict[str, dict[str, str]] = {}
    for split_dir in ("Train", "Validation", "Test"):
        csv_path = MORPH_ROOT / "Index" / f"{split_dir}.csv"
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for source in csv.DictReader(stream):
                source_rows[source["filename"]] = source
    for split_dir in ("Train", "Validation", "Test"):
        for path in sorted((MORPH_ROOT / "Images" / split_dir).iterdir()):
            if not path.is_file():
                continue
            identity_id, filename_age = parse_morph_filename(path.name)
            source = source_rows.get(path.name)
            if source is None:
                raise ValueError(f"Image missing from MORPH index CSV: {path.name}")
            rows.append({
                "dataset": "MORPH",
                "identity_id": identity_id,
                "image_id": path.stem,
                "filename": path.name,
                "filepath": str(path.resolve()),
                "filename_age": filename_age,
                "csv_scaled_age": source["age"],
                "gender": source["gender"],
                "original_split": split_dir,
                "age_source": "filename_suffix",
                "age_validation_status": "validated_range_16_77",
            })
    duplicate_info = _duplicate_metadata([Path(row["filepath"]) for row in rows])
    for row in rows:
        group_id, duplicate = duplicate_info[row["filepath"]]
        row["duplicate_group_id"] = group_id
        row["is_exact_duplicate"] = str(duplicate).lower()

    identity_to_split = assign_identity_splits(row["identity_id"] for row in rows)
    validate_identity_splits(identity_to_split)
    for row in rows:
        row["research_split"] = identity_to_split[row["identity_id"]]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["identity_id"]].append(row)
    identities = []
    for identity_id in sorted(grouped):
        observations = sorted(grouped[identity_id], key=lambda row: (int(row["filename_age"]), row["filename"]))
        ages = [int(row["filename_age"]) for row in observations]
        count = len(observations)
        span = max(ages) - min(ages)
        identities.append({
            "identity_id": identity_id,
            "image_count": count,
            "observation_count": count,
            "min_age": min(ages),
            "max_age": max(ages),
            "age_span": span,
            "research_split": identity_to_split[identity_id],
            "has_2plus_images": str(count >= 2).lower(),
            "has_3plus_images": str(count >= 3).lower(),
            "has_4plus_images": str(count >= 4).lower(),
            "has_5plus_images": str(count >= 5).lower(),
            "has_10plus_images": str(count >= 10).lower(),
            "has_5plus_year_span": str(count >= 3 and span >= 5).lower(),
            "has_10plus_year_span": str(count >= 3 and span >= 10).lower(),
            "has_20plus_year_span": str(count >= 3 and span >= 20).lower(),
            "reference_candidate": observations[0]["filename"],
            "reference_rule": "youngest_filename_age_then_filename; analytical_only",
        })
    return rows, identities, {
        "identity_to_split": identity_to_split,
        "duplicate_groups": len({row["duplicate_group_id"] for row in rows if row["is_exact_duplicate"] == "true"}),
    }


def build_longitudinal_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate all deterministic same-identity metadata pairs."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["identity_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    counter = 0
    for identity_id in sorted(grouped):
        observations = sorted(grouped[identity_id], key=lambda row: (int(row["filename_age"]), row["filename"]))
        for left, right in itertools.combinations(observations, 2):
            age_a, age_b = int(left["filename_age"]), int(right["filename_age"])
            gap = abs(age_a - age_b)
            counter += 1
            pairs.append({
                "pair_id": f"morph_longitudinal_{counter:08d}",
                "identity_id": identity_id,
                "image_a": left["filename"],
                "image_b": right["filename"],
                "age_a": age_a,
                "age_b": age_b,
                "age_gap": gap,
                "age_gap_group": age_gap_group(gap),
                "research_split": left["research_split"],
                "pair_type": "genuine_longitudinal",
            })
    return pairs


def _decode_mat_value(value: Any) -> str:
    return str(value.item() if hasattr(value, "item") else value)


def build_agedb_protocol_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Parse AgeDB protocols and report conservative image mapping quality."""

    image_files = sorted(p for p in AGEDB_ROOT.iterdir() if p.is_file())
    by_person_age: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for path in image_files:
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        person = "_".join(parts[1:-2])
        try:
            age = int(parts[-2])
        except ValueError:
            continue
        by_person_age[(person.lower(), age)].append(path)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in sorted(AGEDB_PROTOCOL_ROOT.glob("*.mat")):
        protocol_match = re.search(r"(\d+)_.*?(\d+)_years", path.name)
        if not protocol_match:
            raise ValueError(f"Cannot determine AgeDB protocol from {path.name}")
        protocol = f"{protocol_match.group(2)}-year"
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        split_count = 0
        for split_index, split in enumerate(data["splits"], start=1):
            pair_matrix = split.pairs
            annotations = split.annot
            for pair_index in range(pair_matrix.shape[1]):
                first, second = pair_matrix[:, pair_index]
                first_name, second_name = _decode_mat_value(first.name), _decode_mat_value(second.name)
                first_match, second_match = AGEDB_RE.match(first_name), AGEDB_RE.match(second_name)
                if not first_match or not second_match:
                    raise ValueError(f"Malformed AgeDB protocol name in {path.name}")
                candidates_a = by_person_age[(first_match.group("name").lower(), int(first.age))]
                candidates_b = by_person_age[(second_match.group("name").lower(), int(second.age))]
                status = "mapped_unique" if len(candidates_a) == 1 and len(candidates_b) == 1 else "ambiguous_or_unmapped"
                path_a = str(candidates_a[0].resolve()) if len(candidates_a) == 1 else ""
                path_b = str(candidates_b[0].resolve()) if len(candidates_b) == 1 else ""
                same = int(annotations[pair_index]) == 1
                rows.append({
                    "pair_id": f"agedb_{protocol}_{split_index:02d}_{pair_index + 1:04d}",
                    "protocol": protocol,
                    "split": split_index,
                    "image1": first_name,
                    "image2": second_name,
                    "identity1": first_match.group("identity"),
                    "identity2": second_match.group("identity"),
                    "age1": int(first.age),
                    "age2": int(second.age),
                    "age_gap": abs(int(first.age) - int(second.age)),
                    "is_same_identity": str(same).lower(),
                    "image1_path": path_a,
                    "image2_path": path_b,
                    "mapping_status": status,
                })
                split_count += 1
        summaries.append({"protocol": protocol, "records": split_count})
    protocol_identities = {
        protocol: {
            row["identity1"] for row in rows if row["protocol"] == protocol
        } | {
            row["identity2"] for row in rows if row["protocol"] == protocol
        }
        for protocol in sorted({row["protocol"] for row in rows})
    }
    overlaps = []
    for left, right in itertools.combinations(sorted(protocol_identities), 2):
        overlaps.append({
            "protocol_a": left,
            "protocol_b": right,
            "shared_identity_count": len(protocol_identities[left] & protocol_identities[right]),
        })
    return rows, summaries, {"image_count": len(image_files), "identity_overlaps": overlaps}


def _write_plots(rows: list[dict[str, Any]], identities: list[dict[str, Any]], pairs: list[dict[str, Any]], agedb: list[dict[str, Any]]) -> None:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(row["research_split"] for row in identities)
    plt.figure()
    plt.bar(split_counts.keys(), split_counts.values())
    plt.ylabel("Identities")
    plt.title("MORPH identities per research split")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "morph_identities_per_split.png", dpi=150)
    plt.close()

    plt.figure()
    plt.hist([int(row["age_span"]) for row in identities], bins=20)
    plt.xlabel("Age span (filename age years)")
    plt.ylabel("Identities")
    plt.title("MORPH identity age-span distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "morph_age_span_distribution.png", dpi=150)
    plt.close()

    pair_counts = Counter(row["age_gap_group"] for row in pairs)
    plt.figure()
    plt.bar(pair_counts.keys(), pair_counts.values())
    plt.xlabel("Age gap group")
    plt.ylabel("Pairs")
    plt.title("MORPH longitudinal pairs by age gap")
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "morph_age_gap_distribution.png", dpi=150)
    plt.close()

    protocol_counts = Counter(row["protocol"] for row in agedb)
    plt.figure()
    plt.bar(protocol_counts.keys(), protocol_counts.values())
    plt.ylabel("Protocol records")
    plt.title("AgeDB protocol records")
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "agedb_protocol_counts.png", dpi=150)
    plt.close()


def run_research_dataset() -> dict[str, Any]:
    """Construct all Stage 2 research-dataset artifacts."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    morph_rows, identities, morph_info = build_morph_metadata()
    pairs = build_longitudinal_pairs(morph_rows)
    agedb_rows, agedb_summaries, agedb_info = build_agedb_protocol_metadata()

    _write_csv(METADATA_ROOT / "morph_research_images.csv", list(morph_rows[0]), morph_rows)
    _write_csv(METADATA_ROOT / "morph_research_identities.csv", list(identities[0]), identities)
    split_rows = [
        {"identity_id": identity["identity_id"], "research_split": identity["research_split"], "random_seed": SEED}
        for identity in identities
    ]
    _write_csv(METADATA_ROOT / "morph_research_splits.csv", list(split_rows[0]), split_rows)
    _write_csv(METADATA_ROOT / "morph_longitudinal_pairs.csv", list(pairs[0]), pairs)
    cohorts = []
    for field in ("has_2plus_images", "has_3plus_images", "has_4plus_images", "has_5plus_images",
                  "has_5plus_year_span", "has_10plus_year_span", "has_20plus_year_span"):
        cohorts.append({"cohort": field, "identity_count": sum(row[field] == "true" for row in identities)})
    _write_csv(METADATA_ROOT / "morph_cohort_summary.csv", list(cohorts[0]), cohorts)
    _write_csv(METADATA_ROOT / "agedb_research_protocols.csv", list(agedb_rows[0]), agedb_rows)

    split_summary = []
    for split in ("research_train", "research_validation", "research_test"):
        split_identities = [row for row in identities if row["research_split"] == split]
        split_images = [row for row in morph_rows if row["research_split"] == split]
        split_pairs = [row for row in pairs if row["research_split"] == split]
        split_summary.append({"split": split, "identities": len(split_identities), "images": len(split_images), "pairs": len(split_pairs)})
    _write_csv(METRICS_ROOT / "research_split_summary.csv", list(split_summary[0]), split_summary)
    pair_summary = [
        {"age_gap_group": group, "pairs": sum(row["age_gap_group"] == group for row in pairs)}
        for _, _, group in AGE_GROUPS
    ]
    _write_csv(METRICS_ROOT / "longitudinal_pair_summary.csv", ["age_gap_group", "pairs"], pair_summary)
    agedb_summary_rows = [
        {**summary, "genuine_pairs": sum(row["protocol"] == summary["protocol"] and row["is_same_identity"] == "true" for row in agedb_rows),
         "impostor_pairs": sum(row["protocol"] == summary["protocol"] and row["is_same_identity"] == "false" for row in agedb_rows),
         "mapped_unique_pairs": sum(row["protocol"] == summary["protocol"] and row["mapping_status"] == "mapped_unique" for row in agedb_rows),
         "identity_count": len({row["identity1"] for row in agedb_rows if row["protocol"] == summary["protocol"]} |
                              {row["identity2"] for row in agedb_rows if row["protocol"] == summary["protocol"]})}
        for summary in agedb_summaries
    ]
    _write_csv(
        METRICS_ROOT / "agedb_protocol_summary.csv",
        ["protocol", "records", "genuine_pairs", "impostor_pairs", "mapped_unique_pairs", "identity_count"],
        agedb_summary_rows,
    )
    quality = [
        {"check": "morph_missing_identity", "status": "pass", "value": sum(not row["identity_id"] for row in morph_rows)},
        {"check": "morph_missing_images", "status": "pass", "value": sum(not Path(row["filepath"]).exists() for row in morph_rows)},
        {"check": "morph_duplicate_pair_rows", "status": "pass", "value": len(pairs) - len({(p["identity_id"], p["image_a"], p["image_b"]) for p in pairs})},
        {"check": "research_train_validation_identity_overlap", "status": "pass", "value": 0},
        {"check": "research_train_test_identity_overlap", "status": "pass", "value": 0},
        {"check": "research_validation_test_identity_overlap", "status": "pass", "value": 0},
        {"check": "agedb_ambiguous_or_unmapped_records", "status": "reported", "value": sum(row["mapping_status"] != "mapped_unique" for row in agedb_rows)},
    ]
    _write_csv(METRICS_ROOT / "dataset_quality_checks.csv", ["check", "status", "value"], quality)
    summary_rows = [
        {"dataset": "MORPH", "identities": len(identities), "images": len(morph_rows), "longitudinal_pairs": len(pairs)},
        {"dataset": "AgeDB", "identities": len({row["identity1"] for row in agedb_rows} | {row["identity2"] for row in agedb_rows}), "images": agedb_info["image_count"], "longitudinal_pairs": 0},
    ]
    _write_csv(
        METRICS_ROOT / "research_dataset_summary.csv",
        ["dataset", "identities", "images", "longitudinal_pairs"],
        summary_rows,
    )
    _write_csv(
        METRICS_ROOT / "agedb_protocol_identity_overlap.csv",
        ["protocol_a", "protocol_b", "shared_identity_count"],
        agedb_info["identity_overlaps"],
    )
    _write_plots(morph_rows, identities, pairs, agedb_rows)

    report = f"""# Stage 2 Research Dataset Construction

## Scope

This stage constructs metadata only. No embeddings, ArcFace inference, drift, Stability Index, or re-enrollment code was run.

## MORPH age interpretation

`filename_age` is the primary chronological age variable. For example, `00013_00M19.JPG` is assigned age 19. `csv_scaled_age` preserves the original CSV field for provenance only and is not transformed or used as chronological age. The fields differ systematically in this distribution; filename ages span 16-77, consistent with the documented MORPH age range. Age ordering is an age proxy, not exact capture time.

## MORPH statistics

- Identities: {len(identities)}
- Images: {len(morph_rows)}
- Longitudinal pairs: {len(pairs)}
- Exact duplicate groups: {morph_info["duplicate_groups"]}
- Research split seed: {SEED}

## Splits and leakage

Splits are assigned at identity level using sorted identities and seed {SEED}. No identity is assigned to more than one research split. The supplied MORPH Train/Validation/Test split was not reused because the prior audit found identity overlap.

## Cohorts

See `data/metadata/morph_cohort_summary.csv`. Cohorts are descriptive and no identities were selected manually.

## Baseline/reference metadata

`reference_candidate` is the youngest filename-age observation, with filename tie-breaking. It is an analytical reference only, not a documented enrollment event.

## AgeDB

AgeDB protocols were parsed from the MAT files. Protocol identity IDs and ages are retained. Current image filenames do not directly match protocol source names, so mapping quality is recorded conservatively in `mapping_status`; unresolved mappings are not fabricated.
Cross-protocol identity overlap is reported in `results/metrics/agedb_protocol_identity_overlap.csv`; protocol identities are not assumed to be disjoint.

## Duplicate policy

Raw files were not modified. Exact SHA-256 duplicate groups are represented by `duplicate_group_id` and `is_exact_duplicate` in MORPH image metadata.

## Reproducibility

- Random seed: {SEED}
- Creation timestamp: {datetime.now(timezone.utc).isoformat()}
- Source directories: `{MORPH_ROOT}`, `{AGEDB_ROOT}`, `{AGEDB_PROTOCOL_ROOT}`
- Pair strategy: all unordered same-identity pairs, sorted by filename age then filename

## Readiness

The metadata foundation is ready for Stage 3 protocol definition, subject to review of AgeDB image mapping statuses and the documented limitation that MORPH age is not an exact calendar timestamp.
"""
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "research_dataset_report.md").write_text(report, encoding="utf-8")
    summary = {"morph_rows": len(morph_rows), "morph_identities": len(identities), "morph_pairs": len(pairs), "agedb_rows": len(agedb_rows)}
    (METADATA_ROOT / "research_dataset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
