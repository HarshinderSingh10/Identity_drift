"""Stage 2.5 methodological audit for identity metadata and duplicate content."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.dataset_loader import SUPPORTED_DATASETS, load_dataset_pairs
from src.experiments.drift_experiment import DEFAULT_DATASETS, RESULTS_ROOT


AUDIT_FIELDS = [
    "dataset",
    "annotation_records",
    "positive_pairs",
    "negative_pairs",
    "unique_images",
    "positive_unique_images",
    "negative_unique_images",
    "positive_negative_overlap",
    "duplicate_groups",
    "duplicate_files_beyond_first",
    "largest_duplicate_group",
]


def file_sha256(path: Path) -> str:
    """Return the exact SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_positive_graph(pairs: Iterable[Any]) -> dict[str, set[str]]:
    """Build an undirected graph from positive pair image paths."""

    graph: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        left = str(pair.image_a)
        right = str(pair.image_b)
        graph[left].add(right)
        graph[right].add(left)
    return dict(graph)


def connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    """Return connected components using iterative depth-first traversal."""

    components: list[set[str]] = []
    visited: set[str] = set()
    for node in graph:
        if node in visited:
            continue
        component: set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(graph.get(current, set()) - visited)
        components.append(component)
    return components


def find_graph_contradictions(
    positive_pairs: Iterable[Any],
    negative_pairs: Iterable[Any],
) -> list[dict[str, str]]:
    """Find negative edges whose endpoints share a positive graph component."""

    graph = build_positive_graph(positive_pairs)
    component_by_node: dict[str, int] = {}
    for index, component in enumerate(connected_components(graph)):
        for node in component:
            component_by_node[node] = index

    contradictions: list[dict[str, str]] = []
    for pair in negative_pairs:
        left = str(pair.image_a)
        right = str(pair.image_b)
        if left in component_by_node and component_by_node.get(left) == component_by_node.get(right):
            contradictions.append(
                {
                    "image_a": left,
                    "image_b": right,
                    "component": str(component_by_node[left]),
                }
            )
    return contradictions


def hash_groups_for_paths(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group paths by exact file-content hash."""

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[file_sha256(path)].append(path)
    return dict(groups)


def _pair_hashes(pair: Any, hashes: dict[str, str]) -> tuple[str, str]:
    return hashes[str(pair.image_a)], hashes[str(pair.image_b)]


def _unique_pair_keys(pairs: Iterable[Any]) -> set[tuple[str, str]]:
    return {(str(pair.image_a), str(pair.image_b)) for pair in pairs}


def _verification_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    positives = np.asarray(
        [float(row["cosine_similarity"]) for row in rows if row["pair_type"] == "positive"],
        dtype=np.float64,
    )
    negatives = np.asarray(
        [float(row["cosine_similarity"]) for row in rows if row["pair_type"] == "negative"],
        dtype=np.float64,
    )
    if positives.size == 0 or negatives.size == 0:
        return {
            "positive_pairs": int(positives.size),
            "negative_pairs": int(negatives.size),
            "roc_auc": np.nan,
            "eer": np.nan,
            "eer_threshold": np.nan,
            "mean_positive_similarity": np.nan,
            "mean_negative_similarity": np.nan,
        }
    scores = np.concatenate([positives, negatives])
    labels = np.concatenate(
        [np.ones(positives.size, dtype=np.int64), np.zeros(negatives.size, dtype=np.int64)]
    )
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "positive_pairs": int(positives.size),
        "negative_pairs": int(negatives.size),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "eer_threshold": float(thresholds[index]),
        "mean_positive_similarity": float(np.mean(positives)),
        "mean_negative_similarity": float(np.mean(negatives)),
    }


def _read_verification_rows() -> list[dict[str, str]]:
    path = RESULTS_ROOT / "verification_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Verification results are required: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_audit(dataset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    positive_pairs = load_dataset_pairs(dataset_name, "positive")
    negative_pairs = load_dataset_pairs(dataset_name, "negative")
    all_pairs = positive_pairs + negative_pairs
    referenced_paths = sorted(
        {pair.image_a for pair in all_pairs} | {pair.image_b for pair in all_pairs}
    )
    hashes = hash_groups_for_paths(referenced_paths)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    positive_images = {str(path) for pair in positive_pairs for path in (pair.image_a, pair.image_b)}
    negative_images = {str(path) for pair in negative_pairs for path in (pair.image_a, pair.image_b)}

    audit = {
        "dataset": dataset_name,
        "annotation_records": len(all_pairs),
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "unique_images": len(referenced_paths),
        "positive_unique_images": len(positive_images),
        "negative_unique_images": len(negative_images),
        "positive_negative_overlap": len(positive_images & negative_images),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files_beyond_first": sum(len(paths) - 1 for paths in duplicate_groups),
        "largest_duplicate_group": max((len(paths) for paths in duplicate_groups), default=1),
    }

    positive_hashes = Counter(_pair_hashes(pair, {str(p): digest for digest, paths in hashes.items() for p in paths}) for pair in positive_pairs)
    negative_hashes = Counter(_pair_hashes(pair, {str(p): digest for digest, paths in hashes.items() for p in paths}) for pair in negative_pairs)
    path_to_hash = {str(path): digest for digest, paths in hashes.items() for path in paths}
    positive_pair_hash_groups = Counter(
        digest
        for pair in positive_pairs
        for digest in _pair_hashes(pair, path_to_hash)
        if len(hashes[digest]) > 1
    )
    negative_pair_hash_groups = Counter(
        digest
        for pair in negative_pairs
        for digest in _pair_hashes(pair, path_to_hash)
        if len(hashes[digest]) > 1
    )
    pair_group_use: dict[str, set[str]] = defaultdict(set)
    for pair_type, pairs in (("positive", positive_pairs), ("negative", negative_pairs)):
        for pair in pairs:
            for digest in _pair_hashes(pair, path_to_hash):
                if len(hashes[digest]) > 1:
                    pair_group_use[digest].add(pair_type)

    duplicate_rows = [
        {"dataset": dataset_name, "metric": "unique_hashes", "value": len(hashes)},
        {"dataset": dataset_name, "metric": "duplicate_groups", "value": len(duplicate_groups)},
        {
            "dataset": dataset_name,
            "metric": "duplicate_files_beyond_first",
            "value": sum(len(paths) - 1 for paths in duplicate_groups),
        },
        {
            "dataset": dataset_name,
            "metric": "largest_duplicate_group",
            "value": max((len(paths) for paths in duplicate_groups), default=1),
        },
        {
            "dataset": dataset_name,
            "metric": "positive_pairs_with_duplicate_content",
            "value": sum(
                1
                for pair in positive_pairs
                if any(len(hashes[digest]) > 1 for digest in _pair_hashes(pair, path_to_hash))
            ),
        },
        {
            "dataset": dataset_name,
            "metric": "negative_pairs_with_duplicate_content",
            "value": sum(
                1
                for pair in negative_pairs
                if any(len(hashes[digest]) > 1 for digest in _pair_hashes(pair, path_to_hash))
            ),
        },
        {
            "dataset": dataset_name,
            "metric": "duplicate_hash_groups_used_in_multiple_positive_pairs",
            "value": sum(count > 1 for count in positive_pair_hash_groups.values()),
        },
        {
            "dataset": dataset_name,
            "metric": "duplicate_hash_groups_shared_positive_negative",
            "value": sum(
                1
                for digest, pair_types in pair_group_use.items()
                if pair_types == {"positive", "negative"}
            ),
        },
        {
            "dataset": dataset_name,
            "metric": "duplicate_hash_groups_used_in_multiple_negative_pairs",
            "value": sum(count > 1 for count in negative_pair_hash_groups.values()),
        },
    ]
    largest_groups = sorted(duplicate_groups, key=len, reverse=True)[:10]
    for index, paths in enumerate(largest_groups, start=1):
        duplicate_rows.append(
            {
                "dataset": dataset_name,
                "metric": f"largest_group_{index}_size",
                "value": len(paths),
            }
        )
    size_distribution = Counter(len(paths) for paths in duplicate_groups)
    for group_size, group_count in sorted(size_distribution.items()):
        duplicate_rows.append(
            {
                "dataset": dataset_name,
                "metric": f"duplicate_group_size_{group_size}_count",
                "value": group_count,
            }
        )

    graph = build_positive_graph(positive_pairs)
    components = connected_components(graph)
    contradictions = find_graph_contradictions(positive_pairs, negative_pairs)
    graph_rows = [
        {
            "dataset": dataset_name,
            "grouping_source": "positive_pair_graph_inferred",
            "group_id": index,
            "group_size": len(component),
            "component_member_count": len(component),
            "is_ground_truth_identity": False,
            "negative_contradiction_count": sum(
                item["component"] == str(index) for item in contradictions
            ),
        }
        for index, component in enumerate(components)
    ]
    graph_rows.append(
        {
            "dataset": dataset_name,
            "grouping_source": "positive_pair_graph_inferred_summary",
            "group_id": "SUMMARY",
            "group_size": len(components),
            "component_member_count": max((len(component) for component in components), default=0),
            "is_ground_truth_identity": False,
            "negative_contradiction_count": len(contradictions),
        }
    )
    return audit, {"duplicate_rows": duplicate_rows, "graph_rows": graph_rows, "hashes": hashes}


def _cross_dataset_duplicate_rows(dataset_hashes: dict[str, dict[str, list[Path]]]) -> list[dict[str, Any]]:
    by_hash: dict[str, set[str]] = defaultdict(set)
    for dataset, groups in dataset_hashes.items():
        for digest in groups:
            by_hash[digest].add(dataset)
    shared = [datasets for datasets in by_hash.values() if len(datasets) > 1]
    return [
        {
            "dataset": "cross_dataset",
            "metric": "cross_dataset_shared_hash_groups",
            "value": len(shared),
        },
        {
            "dataset": "cross_dataset",
            "metric": "cross_dataset_duplicate_files_beyond_first_dataset",
            "value": sum(len(datasets) - 1 for datasets in shared),
        },
    ]


def _duplicate_restricted_rows(
    dataset: str,
    rows: list[dict[str, str]],
    hashes: dict[str, str],
) -> tuple[list[dict[str, str]], int]:
    """Keep pairs only when neither endpoint content hash is reused.

    This conservative rule is declared before comparison: any verification pair
    touching exact duplicate content is excluded from the sensitivity view.
    """

    counts = Counter(hashes.values())
    restricted = [
        row
        for row in rows
        if counts[hashes[row["image_a"]]] == 1 and counts[hashes[row["image_b"]]] == 1
    ]
    return restricted, len(rows) - len(restricted)


def run_audit(datasets: list[str] | None = None) -> dict[str, Any]:
    """Run the complete Stage 2.5 audit and write its artifacts."""

    selected = list(datasets or DEFAULT_DATASETS)
    audit_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    dataset_hashes: dict[str, dict[str, list[Path]]] = {}
    for dataset in selected:
        audit, details = _dataset_audit(dataset)
        audit_rows.append(audit)
        duplicate_rows.extend(details["duplicate_rows"])
        graph_rows.extend(details["graph_rows"])
        dataset_hashes[dataset] = details["hashes"]
    duplicate_rows.extend(_cross_dataset_duplicate_rows(dataset_hashes))

    verification_rows = _read_verification_rows()
    impact_rows: list[dict[str, Any]] = []
    for dataset in selected:
        original = [row for row in verification_rows if row["dataset"] == dataset]
        path_hashes = {
            str(path): digest
            for digest, paths in dataset_hashes[dataset].items()
            for path in paths
        }
        restricted, excluded = _duplicate_restricted_rows(dataset, original, path_hashes)
        for view_name, view_rows, excluded_count in (
            ("original", original, 0),
            ("duplicate_restricted", restricted, excluded),
        ):
            metrics = _verification_metrics(view_rows)
            impact_rows.append(
                {
                    "dataset": dataset,
                    "view": view_name,
                    "exclusion_rule": (
                        "None"
                        if view_name == "original"
                        else "Exclude any pair touching exact duplicate image content across referenced images."
                    ),
                    "excluded_pairs": excluded_count,
                    **metrics,
                }
            )

    _write_csv(RESULTS_ROOT / "identity_audit.csv", AUDIT_FIELDS, audit_rows)
    _write_csv(RESULTS_ROOT / "duplicate_analysis.csv", ["dataset", "metric", "value"], duplicate_rows)
    _write_csv(
        RESULTS_ROOT / "identity_group_analysis.csv",
        [
            "dataset",
            "grouping_source",
            "group_id",
            "group_size",
            "component_member_count",
            "is_ground_truth_identity",
            "negative_contradiction_count",
        ],
        graph_rows,
    )
    _write_csv(
        RESULTS_ROOT / "duplicate_pair_impact.csv",
        [
            "dataset",
            "view",
            "exclusion_rule",
            "excluded_pairs",
            "positive_pairs",
            "negative_pairs",
            "roc_auc",
            "eer",
            "eer_threshold",
            "mean_positive_similarity",
            "mean_negative_similarity",
        ],
        impact_rows,
    )
    report = _build_report(audit_rows, duplicate_rows, graph_rows, impact_rows)
    (RESULTS_ROOT / "stage2_5_report.md").write_text(report, encoding="utf-8")
    return {
        "audit": audit_rows,
        "duplicates": duplicate_rows,
        "groups": graph_rows,
        "impact": impact_rows,
    }


def _build_report(
    audit_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
) -> str:
    summary_rows = [row for row in graph_rows if row["group_id"] == "SUMMARY"]
    lines = [
        "# Stage 2.5 Methodological Audit",
        "",
        "## Objective",
        "This audit evaluates whether the current pair annotations and image files support identity-aware, duplicate-aware, individualized, and temporal analyses. It does not implement a Stability Index or re-enrollment policy.",
        "",
        "## Data audit",
        "",
        "| Dataset | Records | Positive | Negative | Unique images | Pos/neg overlap | Duplicate groups | Duplicate files beyond first | Largest group |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['dataset']} | {row['annotation_records']} | {row['positive_pairs']} | "
            f"{row['negative_pairs']} | {row['unique_images']} | {row['positive_negative_overlap']} | "
            f"{row['duplicate_groups']} | {row['duplicate_files_beyond_first']} | {row['largest_duplicate_group']} |"
        )
    lines.extend(
        [
            "",
            "## Identity metadata findings",
            "",
            "**Identity IDs are not available in the current project data.** The annotation rows contain only a binary pair label and two numeric image-path references. No subject names, person IDs, original identity folders, dates, ages, timestamps, or alternate metadata files were found in the project data used by this audit.",
            "",
            "Numeric filenames and annotation ordering were not treated as identity information.",
            "",
            "## Pair graph findings",
            "",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"- **{row['dataset']}**: {row['group_size']} inferred positive-pair connected components; "
            f"largest component size {row['component_member_count']}; "
            f"negative-edge contradictions {row['negative_contradiction_count']}. "
            "These components are not ground-truth identities."
        )
    lines.extend(
        [
            "",
            "Because each current positive pair uses a separate pair of image paths, the graph is expected to contain mostly size-two components. This provides no validated identity grouping beyond the annotated pair itself.",
            "",
            "## Duplicate-content findings",
            "",
        ]
    )
    for row in duplicate_rows:
        if row["metric"] in {
            "positive_pairs_with_duplicate_content",
            "negative_pairs_with_duplicate_content",
            "duplicate_hash_groups_used_in_multiple_positive_pairs",
            "duplicate_hash_groups_shared_positive_negative",
        }:
            lines.append(f"- {row['dataset']} {row['metric']}: {row['value']}")
    lines.extend(
        [
            "",
            "Exact duplicate content can create dependence between pair observations. It does not prove label leakage by itself, but any pair touching reused content should not be treated as fully independent evidence.",
            "",
            "## Duplicate sensitivity analysis",
            "",
            "The duplicate-restricted view was defined before calculation as: exclude every verification pair in which either endpoint's exact content hash is reused by another referenced image. This is a conservative sensitivity analysis, not a claim that the restricted view is the correct primary dataset.",
            "",
            "| Dataset | View | Pairs retained | Excluded | ROC-AUC | EER |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in impact_rows:
        lines.append(
            f"| {row['dataset']} | {row['view']} | {row['positive_pairs'] + row['negative_pairs']} | "
            f"{row['excluded_pairs']} | {row['roc_auc']:.6f} | {row['eer']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Identity-disjoint split feasibility",
            "",
            "A ground-truth identity-disjoint development/validation/test split cannot currently be guaranteed because explicit identity IDs are absent. Positive-pair connected components may be used only as an exploratory inferred grouping; they must not be presented as verified identities.",
            "",
            "## Individual stability feasibility",
            "",
            "Individual mean drift, variance, and baseline stability cannot be legitimately estimated from the current data. The available annotations identify pair relationships, not people with multiple independently identified observations. The inferred graph components are not validated identity groups.",
            "",
            "## Temporal drift feasibility",
            "",
            "Temporal drift is not supported by the current annotation format. No capture dates, age values, timestamps, years, or ordered longitudinal observations are available. Filenames and pair ordering do not provide legitimate temporal information. AgeDB-30, CALFW, and CPLFW must not be treated as longitudinal sequences from this data alone.",
            "",
            "## Threats to validity",
            "",
            "- No explicit identity IDs or identity-disjoint evaluation protocol.",
            "- Exact duplicate image content is common and reduces effective independence.",
            "- Pair labels support verification, but not individualized trajectories.",
            "- Dataset differences are descriptive and cannot establish causal age or pose effects.",
            "- Duplicate restriction changes the estimand and is reported only as sensitivity analysis.",
            "",
            "## What the current data supports",
            "",
            "- Pair-level genuine/impostor verification analysis.",
            "- Dataset-level positive-pair drift distributions.",
            "- Exploratory duplicate-content and graph-consistency audits.",
            "- Descriptive sensitivity analysis under an explicit duplicate exclusion rule.",
            "",
            "## What the current data does not support",
            "",
            "- Ground-truth identity-level stability baselines.",
            "- Identity-disjoint train/development/test validation.",
            "- Temporal or longitudinal drift claims.",
            "- Temporary-versus-persistent drift analysis.",
            "- Adaptive re-enrollment decisions for individuals.",
            "",
            "## Minimum additional data required",
            "",
            "The minimum next protocol should provide: a stable subject ID, multiple observations per subject, capture date or ordered visit/session metadata, condition metadata where relevant, raw/aligned image linkage, and a predefined identity-disjoint development/validation/test split. A small controlled consenting-volunteer longitudinal study can satisfy these requirements without downloading a massive training dataset.",
            "",
            "## Recommended Stage 3 protocol",
            "",
            "Do not implement the Stability Index yet. First acquire or construct a small identity-aware longitudinal/appearance-variation evaluation protocol, validate duplicate and subject grouping rules, and reserve identity-disjoint evaluation data. Then test whether a stability feature adds information beyond the verification score before considering re-enrollment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2.5 methodological audit.")
    parser.add_argument("--dataset", action="append", choices=SUPPORTED_DATASETS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    selected = list(DEFAULT_DATASETS) if args.all or not args.dataset else list(args.dataset)
    result = run_audit(selected)
    for row in result["audit"]:
        print(
            f"{row['dataset']}: records={row['annotation_records']} "
            f"duplicate_groups={row['duplicate_groups']} "
            f"duplicate_files_beyond_first={row['duplicate_files_beyond_first']}"
        )


if __name__ == "__main__":
    main()
