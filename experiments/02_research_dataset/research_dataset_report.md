# Stage 2 Research Dataset Construction

## Scope

This stage constructs metadata only. No embeddings, ArcFace inference, drift, Stability Index, or re-enrollment code was run.

## MORPH age interpretation

`filename_age` is the primary chronological age variable. For example, `00013_00M19.JPG` is assigned age 19. `csv_scaled_age` preserves the original CSV field for provenance only and is not transformed or used as chronological age. The fields differ systematically in this distribution; filename ages span 16-77, consistent with the documented MORPH age range. Age ordering is an age proxy, not exact capture time.

## MORPH statistics

- Identities: 19033
- Images: 50015
- Longitudinal pairs: 84137
- Exact duplicate groups: 0
- Research split seed: 42

## Splits and leakage

Splits are assigned at identity level using sorted identities and seed 42. No identity is assigned to more than one research split. The supplied MORPH Train/Validation/Test split was not reused because the prior audit found identity overlap.

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

- Random seed: 42
- Creation timestamp: 2026-09-23T10:39:04.940177+00:00
- Source directories: `E:\PythonProject\Identity_drift\data\raw\Morph`, `E:\PythonProject\Identity_drift\data\raw\AgeDB_Images`, `E:\PythonProject\Identity_drift\data\raw\AgeDB_protocols`
- Pair strategy: all unordered same-identity pairs, sorted by filename age then filename

## Readiness

The metadata foundation is ready for Stage 3 protocol definition, subject to review of AgeDB image mapping statuses and the documented limitation that MORPH age is not an exact calendar timestamp.
