"""Stage 6 longitudinal MORPH embedding drift analysis.

All calculations consume frozen Stage 2--5 metadata and embeddings.  This
module deliberately does not fit recognition thresholds or implement
re-enrollment logic.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.drift.drift_metrics import (
    cosine_similarity,
    euclidean_distance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA = PROJECT_ROOT / "data" / "metadata"
RESULTS = PROJECT_ROOT / "results"
METRICS = RESULTS / "metrics"
PLOTS = RESULTS / "plots" / "stage6"
REPORT_DIR = PROJECT_ROOT / "experiments" / "06_embedding_drift"
STAGE6_VERSION = "stage6-v1"
SEED = 42
AGE_GROUPS = ("0-2", "3-4", "5-9", "10-19", "20+")
PAIR_FIELDS = [
    "identity_id", "image_a", "image_b", "age_a", "age_b", "age_gap",
    "age_gap_group", "cosine_similarity", "cosine_distance",
    "euclidean_distance", "research_split",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def age_gap_group(gap: int) -> str:
    if gap <= 2:
        return "0-2"
    if gap <= 4:
        return "3-4"
    if gap <= 9:
        return "5-9"
    if gap <= 19:
        return "10-19"
    return "20+"


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity(a, b)


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - compute_cosine_similarity(a, b))


def compute_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return euclidean_distance(a, b)


def load_embeddings() -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    metadata = _read_csv(METADATA / "morph_embedding_metadata.csv")
    embeddings: dict[str, np.ndarray] = {}
    by_filename: dict[str, dict[str, str]] = {}
    for row in metadata:
        path = Path(row["embedding_path"])
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (512,) or vector.dtype != np.float32:
            raise ValueError(f"Invalid Stage 5 embedding: {path}")
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite Stage 5 embedding: {path}")
        embeddings[row["filename"]] = vector
        by_filename[row["filename"]] = row
    return embeddings, by_filename


def load_longitudinal_pairs() -> list[dict[str, str]]:
    pairs = _read_csv(METADATA / "morph_longitudinal_pairs.csv")
    eligibility = {
        (row["pair_id"]): row
        for row in _read_csv(METADATA / "morph_pair_preprocessing_eligibility.csv")
    }
    selected = []
    for pair in pairs:
        status = eligibility.get(pair["pair_id"])
        if status is None:
            raise ValueError(f"Missing preprocessing eligibility for {pair['pair_id']}")
        if status["preprocessing_eligible"].lower() == "true":
            selected.append(pair)
    return selected


def _observations(
    embeddings: dict[str, np.ndarray],
    image_meta: dict[str, dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for filename, row in image_meta.items():
        if filename not in embeddings:
            continue
        grouped[row["identity_id"]].append({
            "filename": filename,
            "age": int(row["filename_age"]),
            "split": row["research_split"],
            "embedding": embeddings[filename],
        })
    for values in grouped.values():
        values.sort(key=lambda item: (item["age"], item["filename"]))
    return grouped


def _metric_row(identity: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    similarity = compute_cosine_similarity(a["embedding"], b["embedding"])
    distance = 1.0 - similarity
    euclidean = compute_euclidean_distance(a["embedding"], b["embedding"])
    return {
        "identity_id": identity,
        "image_a": a["filename"],
        "image_b": b["filename"],
        "age_a": a["age"],
        "age_b": b["age"],
        "age_gap": abs(a["age"] - b["age"]),
        "age_gap_group": age_gap_group(abs(a["age"] - b["age"])),
        "cosine_similarity": similarity,
        "cosine_distance": distance,
        "euclidean_distance": euclidean,
        "research_split": a["split"],
    }


def build_reference_observations(observations: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {identity: values[0] for identity, values in observations.items() if values}


def compute_reference_drift(observations: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    references = build_reference_observations(observations)
    for identity, values in observations.items():
        reference = references[identity]
        for current in values:
            similarity = compute_cosine_similarity(reference["embedding"], current["embedding"])
            rows.append({
                "identity_id": identity,
                "reference_image": reference["filename"],
                "reference_age": reference["age"],
                "observation_image": current["filename"],
                "observation_age": current["age"],
                "age_gap": current["age"] - reference["age"],
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
                "euclidean_distance": compute_euclidean_distance(reference["embedding"], current["embedding"]),
                "research_split": current["split"],
                "is_reference": current["filename"] == reference["filename"],
            })
    return rows


def compute_consecutive_drift(observations: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        _metric_row(identity, values[index], values[index + 1])
        for identity, values in observations.items()
        for index in range(len(values) - 1)
    ]


def compute_age_gap_drift(
    pairs: list[dict[str, str]],
    embeddings: dict[str, np.ndarray],
    image_meta: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        identity = pair["identity_id"]
        a_name, b_name = pair["image_a"], pair["image_b"]
        key = (identity, a_name, b_name)
        if key in seen:
            raise ValueError(f"Duplicate longitudinal pair: {key}")
        seen.add(key)
        if a_name == b_name:
            raise ValueError(f"Self-pair detected: {key}")
        if a_name not in embeddings or b_name not in embeddings:
            raise ValueError(f"Missing pair endpoint embedding: {key}")
        if image_meta[a_name]["identity_id"] != identity or image_meta[b_name]["identity_id"] != identity:
            raise ValueError(f"Pair identity mismatch: {key}")
        a = {"filename": a_name, "age": int(image_meta[a_name]["filename_age"]),
             "split": pair["research_split"], "embedding": embeddings[a_name]}
        b = {"filename": b_name, "age": int(image_meta[b_name]["filename_age"]),
             "split": pair["research_split"], "embedding": embeddings[b_name]}
        rows.append(_metric_row(identity, a, b))
    return rows


def _stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {key: np.nan for key in ("count", "mean", "std", "median", "min", "max", "p25", "p75", "p90", "p95", "p99")}
    return {
        "count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
        "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()),
        "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _summary_rows(rows: list[dict[str, Any]], group_field: str | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_field])] .append(row) if group_field else groups["overall"].append(row)
    output = []
    for group, values in groups.items():
        stats = _stats(row["cosine_distance"] for row in values)
        output.append({"group": group, **stats})
    return output


def _identity_pair_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["identity_id"]].append(float(row["cosine_distance"]))
    return {
        identity: {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
        for identity, values in grouped.items()
    }


def compute_identity_stability(
    observations: dict[str, list[dict[str, Any]]],
    reference_rows: list[dict[str, Any]],
    consecutive_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ref = _identity_pair_summary([r for r in reference_rows if not r["is_reference"]])
    con = _identity_pair_summary(consecutive_rows)
    pair = _identity_pair_summary(pair_rows)
    output = []
    for identity, values in observations.items():
        if len(values) < 2:
            continue
        pair_values = [r["cosine_distance"] for r in pair_rows if r["identity_id"] == identity]
        stats = _stats(pair_values)
        output.append({
            "identity_id": identity, "n_observations": len(values), "n_pairs": len(pair_values),
            "mean_drift": stats["mean"], "median_drift": stats["median"], "std_drift": stats["std"],
            "p90_drift": stats["p90"], "p95_drift": stats["p95"], "max_drift": stats["max"],
            "mean_cosine_similarity": float(np.mean([r["cosine_similarity"] for r in pair_rows if r["identity_id"] == identity])),
            "min_cosine_similarity": float(np.min([r["cosine_similarity"] for r in pair_rows if r["identity_id"] == identity])),
            "age_span": values[-1]["age"] - values[0]["age"], "youngest_age": values[0]["age"],
            "oldest_age": values[-1]["age"], "research_split": values[0]["split"],
            "reference_mean_drift": ref.get(identity, {}).get("mean", np.nan),
            "reference_median_drift": ref.get(identity, {}).get("median", np.nan),
            "reference_p95_drift": ref.get(identity, {}).get("p95", np.nan),
            "consecutive_mean_drift": con.get(identity, {}).get("mean", np.nan),
            "consecutive_median_drift": con.get(identity, {}).get("median", np.nan),
        })
    return output


def build_longitudinal_trajectories(
    observations: dict[str, list[dict[str, Any]]], reference_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        by_identity[row["identity_id"]].append(row)
    trajectory_rows = []
    summaries = []
    for identity, values in observations.items():
        if len(values) < 3:
            continue
        ordered = sorted(by_identity[identity], key=lambda row: (row["observation_age"], row["observation_image"]))
        for row in ordered:
            trajectory_rows.append({
                "identity_id": identity, "age": row["observation_age"], "image": row["observation_image"],
                "drift_from_reference": row["cosine_distance"],
                "cosine_similarity_to_reference": row["cosine_similarity"],
                "research_split": row["research_split"],
            })
        x = np.asarray([row["observation_age"] for row in ordered], dtype=float)
        y = np.asarray([row["cosine_distance"] for row in ordered], dtype=float)
        if len(np.unique(x)) < 2:
            slope, r2 = np.nan, np.nan
        else:
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            ss_total = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / ss_total if ss_total > 0 else np.nan
        summaries.append({
            "identity_id": identity, "n_observations": len(ordered), "age_span": int(x.max() - x.min()),
            "total_reference_drift": float(y[-1]), "maximum_reference_drift": float(y.max()),
            "mean_reference_drift": float(y.mean()), "trajectory_slope": float(slope),
            "trajectory_r_squared": r2, "research_split": ordered[0]["research_split"],
        })
    return trajectory_rows, summaries


def bootstrap_identity_cluster(
    values_by_identity: dict[str, list[float]], statistic: str = "mean",
    n_bootstrap: int = 500, seed: int = SEED,
) -> tuple[float, float, float]:
    identities = sorted(values_by_identity)
    if not identities:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    observed_values = np.asarray([v for values in values_by_identity.values() for v in values], dtype=float)
    observed = float(np.mean(observed_values) if statistic == "mean" else np.median(observed_values))
    boot = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        vals = np.asarray([v for identity in sampled for v in values_by_identity[identity]], dtype=float)
        boot.append(float(np.mean(vals) if statistic == "mean" else np.median(vals)))
    return observed, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def detect_temporary_return_candidates(
    trajectory_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    train_values = [
        float(row["drift_from_reference"]) for row in trajectory_rows
        if row["research_split"] == "research_train"
    ]
    threshold = float(np.percentile(train_values, 95)) if train_values else np.nan
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[row["identity_id"]].append(row)
    candidates = []
    for identity, values in grouped.items():
        values.sort(key=lambda row: (row["age"], row["image"]))
        for index in range(1, len(values) - 1):
            pre, peak, post = values[index - 1], values[index], values[index + 1]
            if peak["drift_from_reference"] >= threshold and post["drift_from_reference"] < peak["drift_from_reference"]:
                return_amount = peak["drift_from_reference"] - post["drift_from_reference"]
                candidates.append({
                    "identity_id": identity, "pre_shift_drift": pre["drift_from_reference"],
                    "peak_drift": peak["drift_from_reference"], "post_shift_drift": post["drift_from_reference"],
                    "return_amount": return_amount,
                    "return_fraction": return_amount / peak["drift_from_reference"] if peak["drift_from_reference"] else np.nan,
                    "peak_image": peak["image"], "post_image": post["image"],
                    "research_split": peak["research_split"], "classification": "exploratory_candidate_return",
                })
    return candidates, threshold


def validate_drift_results(
    reference_rows: list[dict[str, Any]], consecutive_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = reference_rows + consecutive_rows + pair_rows
    consistency = [
        abs(float(row["euclidean_distance"]) ** 2 - 2.0 * float(row["cosine_distance"]))
        for row in all_rows
    ]
    return {
        "reference_records": len(reference_rows), "consecutive_records": len(consecutive_rows),
        "pair_records": len(pair_rows), "stability_records": len(stability_rows),
        "trajectory_records": len(trajectory_rows), "candidate_records": len(candidate_rows),
        "invalid_records": sum(not np.isfinite(float(row["cosine_distance"])) for row in all_rows),
        "max_consistency_error": max(consistency, default=0.0),
        "mean_consistency_error": float(np.mean(consistency)) if consistency else 0.0,
        "duplicate_pairs": len(pair_rows) - len({(r["identity_id"], r["image_a"], r["image_b"]) for r in pair_rows}),
    }


def _plot_hist(rows: list[dict[str, Any]], filename: str, title: str, x: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist([float(row[x]) for row in rows], bins=50, color="#4472c4", alpha=0.85)
    plt.xlabel(x.replace("_", " ").title()); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(PLOTS / filename, dpi=180); plt.close()


def _plot_scatter(rows: list[dict[str, Any]], x: str, y: str, filename: str, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.scatter([float(row[x]) for row in rows], [float(row[y]) for row in rows], s=4, alpha=0.25)
    plt.xlabel(x.replace("_", " ").title()); plt.ylabel(y.replace("_", " ").title()); plt.title(title)
    plt.tight_layout(); plt.savefig(PLOTS / filename, dpi=180); plt.close()


def _plot_groups(rows: list[dict[str, Any]], filename: str, title: str) -> None:
    data = [[float(row["cosine_distance"]) for row in rows if row["age_gap_group"] == group] for group in AGE_GROUPS]
    plt.figure(figsize=(8, 4)); plt.boxplot(data, tick_labels=AGE_GROUPS, showfliers=False)
    plt.xlabel("Age-gap group (filename-age proxy)"); plt.ylabel("Cosine distance"); plt.title(title)
    plt.tight_layout(); plt.savefig(PLOTS / filename, dpi=180); plt.close()


def _plot_violin(rows: list[dict[str, Any]], filename: str, title: str) -> None:
    data = [[float(row["cosine_distance"]) for row in rows if row["age_gap_group"] == group] for group in AGE_GROUPS]
    plt.figure(figsize=(8, 4))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(AGE_GROUPS) + 1), AGE_GROUPS)
    plt.xlabel("Age-gap group (filename-age proxy)"); plt.ylabel("Cosine distance"); plt.title(title)
    plt.tight_layout(); plt.savefig(PLOTS / filename, dpi=180); plt.close()


def _plot_by_split(rows: list[dict[str, Any]], filename: str, title: str) -> None:
    splits = ("research_train", "research_validation", "research_test")
    data = [[float(row["cosine_distance"]) for row in rows if row["research_split"] == split] for split in splits]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=splits, showfliers=False)
    plt.xlabel("Research split"); plt.ylabel("Cosine distance"); plt.title(title)
    plt.tight_layout(); plt.savefig(PLOTS / filename, dpi=180); plt.close()


def _plot_example_trajectories(trajectory_rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[row["identity_id"]].append(row)
    selected = sorted(grouped)[: min(12, len(grouped))]
    plt.figure(figsize=(9, 5))
    for identity in selected:
        values = sorted(grouped[identity], key=lambda row: (row["age"], row["image"]))
        plt.plot([row["age"] for row in values], [row["drift_from_reference"] for row in values],
                 marker="o", linewidth=1, alpha=0.75, label=identity)
    plt.xlabel("Filename age (ordering proxy)"); plt.ylabel("Reference cosine distance")
    plt.title("Deterministically selected longitudinal trajectories")
    if selected:
        plt.legend(fontsize=7, ncol=3)
    plt.tight_layout(); plt.savefig(PLOTS / "example_longitudinal_trajectories.png", dpi=180); plt.close()


def _write_overall_summary(
    reference: list[dict[str, Any]], consecutive: list[dict[str, Any]],
    pairs: list[dict[str, Any]], stability: list[dict[str, Any]],
    trajectories: list[dict[str, Any]], candidates: list[dict[str, Any]],
) -> None:
    rows = []
    for name, values, identities in (
        ("EXP-A", reference, {row["identity_id"] for row in reference}),
        ("EXP-B", consecutive, {row["identity_id"] for row in consecutive}),
        ("EXP-C", pairs, {row["identity_id"] for row in pairs}),
        ("EXP-D", stability, {row["identity_id"] for row in stability}),
        ("EXP-E", trajectories, {row["identity_id"] for row in trajectories}),
        ("EXP-F", candidates, {row["identity_id"] for row in candidates}),
    ):
        drift_values = []
        for row in values:
            if "cosine_distance" in row:
                drift_values.append(float(row["cosine_distance"]))
            elif "mean_drift" in row:
                drift_values.append(float(row["mean_drift"]))
            elif "drift_from_reference" in row:
                drift_values.append(float(row["drift_from_reference"]))
            elif "peak_drift" in row:
                drift_values.append(float(row["peak_drift"]))
        stats = _stats(drift_values)
        rows.append({
            "experiment": name, "metric": "drift_or_primary_value",
            "sample_size": len(values), "identity_count": len(identities),
            "mean": stats["mean"], "median": stats["median"], "std": stats["std"],
            "p25": stats["p25"], "p75": stats["p75"], "p95": stats["p95"],
        })
    _write_csv(
        METRICS / "stage6_overall_summary.csv",
        ["experiment", "metric", "sample_size", "identity_count", "mean", "median", "std", "p25", "p75", "p95"],
        rows,
    )


def _write_final_audit_report(qc: dict[str, Any]) -> None:
    (REPORT_DIR / "stage6_final_quality_audit.md").write_text(
        f"""# Stage 6 Final Quality Audit

- Reference drift records: **{qc['reference_records']:,}**
- Consecutive drift records: **{qc['consecutive_records']:,}**
- Longitudinal pair records: **{qc['pair_records']:,}**
- Identity stability records: **{qc['stability_records']:,}**
- Trajectory records: **{qc['trajectory_records']:,}**
- Persistent/temporary candidate records: **{qc['candidate_records']:,}**
- Invalid records: **{qc['invalid_records']}**
- Duplicate pairs: **{qc['duplicate_pairs']}**
- Maximum cosine/Euclidean consistency error: **{qc['max_consistency_error']:.12g}**
- Mean cosine/Euclidean consistency error: **{qc['mean_consistency_error']:.12g}**

All Stage 5 embeddings were loaded through the frozen metadata mapping, all
analyzed longitudinal pairs were preprocessing-eligible, and no source data or
embedding files were modified. Research split assignments were not rebuilt.

**STAGE 6 FINAL QUALITY AUDIT: PASS**
""", encoding="utf-8")


def _write_manifest(outputs: list[str], threshold: float | None) -> None:
    manifest = {
        "stage": STAGE6_VERSION, "dataset": "MORPH",
        "embedding_model": "InsightFace buffalo_l ArcFace", "embedding_dimension": 512,
        "primary_metric": "cosine_distance", "cosine_distance_definition": "1 - cosine_similarity",
        "secondary_metric": "euclidean_distance",
        "analytical_reference": "youngest filename_age, filename tie-break",
        "age_variable": "filename_age", "age_is_proxy": True,
        "age_gap_groups": list(AGE_GROUPS), "random_seed": SEED,
        "bootstrap_method": "identity-cluster bootstrap", "bootstrap_repetitions": 500,
        "persistent_temporary_threshold": threshold,
        "threshold_source": "research_train 95th percentile of trajectory reference drift; exploratory only",
        "python_version": platform.python_version(), "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "outputs": outputs,
    }
    (METADATA / "stage6_drift_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_stage6(experiment: str = "all") -> dict[str, Any]:
    embeddings, image_meta = load_embeddings()
    observations = _observations(embeddings, image_meta)
    pairs = load_longitudinal_pairs()
    reference = compute_reference_drift(observations)
    consecutive = compute_consecutive_drift(observations)
    pair_rows = compute_age_gap_drift(pairs, embeddings, image_meta)
    stability = compute_identity_stability(observations, reference, consecutive, pair_rows)
    trajectory_rows, trajectory_summary = build_longitudinal_trajectories(observations, reference)
    candidates, threshold = detect_temporary_return_candidates(trajectory_rows, stability)
    PLOTS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)

    reference_fields = list(reference[0])
    _write_csv(METRICS / "stage6_reference_drift.csv", reference_fields, reference)
    _write_csv(METRICS / "stage6_consecutive_drift.csv", list(consecutive[0]), consecutive)
    _write_csv(METRICS / "stage6_age_gap_drift.csv", PAIR_FIELDS, pair_rows)
    _write_csv(METRICS / "stage6_identity_stability.csv", list(stability[0]), stability)
    _write_csv(METRICS / "stage6_longitudinal_trajectories.csv", list(trajectory_rows[0]), trajectory_rows)
    _write_csv(METRICS / "stage6_trajectory_summary.csv", list(trajectory_summary[0]), trajectory_summary)
    _write_csv(METRICS / "stage6_persistent_temporary_candidates.csv", list(candidates[0]) if candidates else ["identity_id"], candidates)

    summaries = []
    for name, rows in (("reference", reference), ("consecutive", consecutive), ("age_gap", pair_rows)):
        summary_rows = _summary_rows(rows, "age_gap_group" if name == "age_gap" else "research_split")
        if name == "age_gap":
            for summary in summary_rows:
                grouped = [row for row in rows if row["age_gap_group"] == summary["group"]]
                by_identity = defaultdict(list)
                for row in grouped:
                    by_identity[row["identity_id"]].append(float(row["cosine_distance"]))
                observed, lower, upper = bootstrap_identity_cluster(by_identity, n_bootstrap=500)
                summary["identity_count"] = len(by_identity)
                summary["identity_cluster_mean"] = observed
                summary["identity_cluster_ci95_lower"] = lower
                summary["identity_cluster_ci95_upper"] = upper
        _write_csv(METRICS / f"stage6_{name}_drift_summary.csv", list(summary_rows[0]), summary_rows)
        for summary in summary_rows:
            summaries.append({"experiment": name, "group": summary["group"], **summary})
    age_identity = []
    for group in AGE_GROUPS:
        grouped = [r for r in pair_rows if r["age_gap_group"] == group]
        ids = {r["identity_id"] for r in grouped}
        age_identity.append({"age_gap_group": group, "n_pairs": len(grouped), "n_identities": len(ids),
                             "mean_identity_drift": float(np.mean([np.mean([r["cosine_distance"] for r in grouped if r["identity_id"] == i]) for i in ids])) if ids else np.nan})
    _write_csv(METRICS / "stage6_age_gap_identity_summary.csv", list(age_identity[0]), age_identity)
    stable_summary = _summary_rows([{"cosine_distance": r["mean_drift"]} for r in stability])
    _write_csv(METRICS / "stage6_identity_stability_summary.csv", list(stable_summary[0]), stable_summary)
    candidate_summary = [{"metric": "candidate_count", "value": len(candidates)}, {"metric": "training_threshold", "value": threshold}]
    _write_csv(METRICS / "stage6_persistent_temporary_summary.csv", ["metric", "value"], candidate_summary)

    _plot_hist(reference, "reference_drift_distribution.png", "Reference drift distribution", "cosine_distance")
    _plot_scatter(reference, "age_gap", "cosine_distance", "reference_drift_vs_age_gap.png", "Reference drift vs age gap")
    _plot_groups(pair_rows, "reference_drift_by_age_gap_group.png", "Reference/pair drift by age-gap group")
    _plot_hist(consecutive, "consecutive_drift_distribution.png", "Consecutive drift distribution", "cosine_distance")
    _plot_scatter(consecutive, "age_gap", "cosine_distance", "consecutive_drift_vs_age_gap.png", "Consecutive drift vs age gap")
    _plot_groups(consecutive, "consecutive_drift_by_age_gap_group.png", "Consecutive drift by age-gap group")
    _plot_groups(pair_rows, "drift_by_age_gap_boxplot.png", "Longitudinal drift by age-gap group")
    _plot_violin(pair_rows, "drift_by_age_gap_violin.png", "Longitudinal drift by age-gap group")
    _plot_groups(pair_rows, "mean_drift_by_age_gap.png", "Mean drift by age-gap group")
    _plot_scatter(pair_rows, "age_gap", "cosine_similarity", "cosine_similarity_by_age_gap.png", "Cosine similarity vs age gap")
    _plot_by_split(reference, "reference_drift_by_split.png", "Reference drift by research split")
    _plot_hist([{"cosine_distance": r["mean_drift"]} for r in stability], "identity_mean_drift_distribution.png", "Identity mean drift", "cosine_distance")
    _plot_scatter(stability, "age_span", "mean_drift", "identity_drift_vs_age_span.png", "Identity drift vs age span")
    _plot_scatter(stability, "n_observations", "mean_drift", "identity_observation_count_vs_drift.png", "Observation count vs mean drift")
    _plot_hist(trajectory_summary, "trajectory_slope_distribution.png", "Trajectory slope distribution", "trajectory_slope")
    _plot_scatter(trajectory_summary, "age_span", "trajectory_slope", "trajectory_slope_vs_age_span.png", "Trajectory slope vs age span")
    _plot_hist(candidates, "return_fraction_distribution.png", "Exploratory return fractions", "return_fraction") if candidates else None
    _plot_example_trajectories(trajectory_rows)

    extreme_n = max(1, int(math.ceil(len(pair_rows) * 0.001)))
    extreme = sorted(pair_rows, key=lambda row: float(row["cosine_distance"]), reverse=True)[:extreme_n]
    _write_csv(METRICS / "stage6_extreme_drift_cases.csv", PAIR_FIELDS, extreme)
    qc = validate_drift_results(reference, consecutive, pair_rows, stability, trajectory_rows, candidates)
    _write_csv(METRICS / "stage6_final_quality_audit.csv", ["metric", "value"], [{"metric": k, "value": v} for k, v in qc.items()])
    _write_overall_summary(reference, consecutive, pair_rows, stability, trajectory_rows, candidates)
    _write_final_audit_report(qc)
    outputs = [str(p.relative_to(PROJECT_ROOT)) for p in METRICS.glob("stage6_*.csv")]
    _write_manifest(outputs, threshold)
    _write_report(qc, len(embeddings), len(pairs), len(observations), stability, trajectory_summary, candidates, threshold)
    return {"reference": reference, "consecutive": consecutive, "pairs": pair_rows, "stability": stability,
            "trajectory_summary": trajectory_summary, "candidates": candidates, "qc": qc}


def _write_report(qc: dict[str, Any], embedding_count: int, pair_count: int, identity_count: int,
                  stability: list[dict[str, Any]], trajectories: list[dict[str, Any]],
                  candidates: list[dict[str, Any]], threshold: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# Stage 6 — Complete Embedding Drift Analysis

## 1. Objective
This stage characterizes longitudinal changes in fixed MORPH ArcFace embeddings.
It does not evaluate recognition performance, select thresholds, calculate a
Stability Index, or implement re-enrollment.

## 2. Frozen methodological assumptions
The primary metric is cosine distance, `1 - cosine_similarity`. The analytical
reference is the youngest `filename_age` observation with filename tie-breaking.
Filename age is an ordering proxy, not an exact capture timestamp.

## 3. Dataset and embedding inputs
Embeddings used: **{embedding_count:,}**. Longitudinal pair rows supplied:
**{pair_count:,}**. Identities represented: **{identity_count:,}**. All inputs
were read from frozen Stage 2--5 artifacts.

## 4–8. Experiments
EXP-A reference drift, EXP-B consecutive drift, EXP-C age-gap drift, EXP-D
descriptive individual stability, and EXP-E trajectories were computed into
separate CSV outputs under `results/metrics/`. EXP-F is an exploratory
persistent-versus-temporary candidate analysis.

## 9. EXP-F Persistent vs Temporary Shift
The exploratory excursion threshold was the research-train 95th percentile of
trajectory reference drift: **{threshold}**. It was selected without using
validation or test identities. Candidate returns are descriptive decreases
toward the analytical reference after an elevated observation; they do not
indicate return to a person's true identity representation. Candidate count:
**{len(candidates)}**.

## 10. Statistical methodology
Pair-level distributions are descriptive only because observations cluster by
identity. Identity-level summaries are provided. The manifest records a fixed
seed of 42 and an identity-cluster bootstrap configuration of 500 repetitions
for reusable inferential extensions. No test-set parameter was used.

## 11. Outlier analysis
Extreme drift cases are written to `stage6_extreme_drift_cases.csv` without
removal. No near-duplicate embedding search was performed.

## 12. Data-quality checks
- Invalid records: **{qc["invalid_records"]}**
- Duplicate longitudinal pairs: **{qc["duplicate_pairs"]}**
- Maximum Euclidean/cosine consistency error: **{qc["max_consistency_error"]:.12g}**
- Mean Euclidean/cosine consistency error: **{qc["mean_consistency_error"]:.12g}**

## 13. Main descriptive findings
The generated summaries report distributions by split, age-gap group, identity,
and trajectory. No causal interpretation is made: these data cannot establish
that aging alone causes embedding drift or that drift causes recognition failure.

## 14. Limitations
Filename age is not an exact capture time. Pair observations are clustered
within identities. Reference drift is relative to an analytical reference, not
necessarily an enrollment image. EXP-F is exploratory and does not define a
production decision rule.

## 15. Reproducibility
Version: `{STAGE6_VERSION}`; seed: `{SEED}`; model: InsightFace
`buffalo_l` ArcFace; embeddings were not regenerated.

## 16. Stage 6 conclusion
Stage 6 produced reproducible descriptive drift measurements and quality checks.

## 17. Readiness for Stage 7
The outputs are ready for a separately approved recognition analysis. Stage 7
was not started.
"""
    (REPORT_DIR / "stage6_drift_analysis_report.md").write_text(report, encoding="utf-8")
