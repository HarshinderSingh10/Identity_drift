# Stage 3 - Experimental Protocol Definition

## Purpose and freeze status

This document freezes the methodology before embedding extraction. Stage 3 defines research units, questions, hypotheses, split roles, metrics, planned analyses, and blocking conditions. It does not extract embeddings, run ArcFace, calculate drift, select thresholds, build a Stability Index, or implement re-enrollment.

Protocol version: `stage3-v1`.

## Research questions

1. **RQ1 - Representation drift:** How much does an identity's ArcFace embedding change across observations?
2. **RQ2 - Age-associated drift:** How does drift vary across predefined age gaps?
3. **RQ3 - Individual stability:** Do identities differ in stability?
4. **RQ4 - Longitudinal trajectory:** How does representation evolve across ordered observations?
5. **RQ5 - Drift and recognition:** How does drift relate to genuine verification acceptance under a separately selected operating point?
6. **RQ6 - Temporary vs persistent drift:** Do later observations move toward the analytical reference or remain shifted?
7. **RQ7 - Re-enrollment:** Can a future drift-aware rule identify potentially useful representation updates?

No question asserts causality. H1-H6 are recorded in `results/metrics/protocol_definition.json`; negative or weak results remain valid.

## Experimental units

- **Identity:** unique MORPH filename-derived identity ID.
- **Observation:** one image belonging to one identity.
- **Analytical reference:** the youngest `filename_age`, then filename tie-break. This is not a documented enrollment event.
- **Longitudinal pair:** two different observations of the same identity.
- **Age gap:** `abs(age_a - age_b)`.
- **Genuine pair:** same-identity pair.
- **Impostor pair:** different-identity pair. It is never called identity drift.

## Age and ordering

`filename_age` is the primary chronological research variable. `csv_scaled_age` is retained only for provenance and is not used for ordering. Filename age is an age proxy, not an exact capture date or timestamp. No calendar-time claims will be made.

The frozen age-gap groups are `0-2`, `3-4`, `5-9`, `10-19`, and `20+` years. No linear age-gap effect is assumed.

## Embedding and drift definition

The planned representation is the 512-dimensional L2-normalized ArcFace recognition embedding from the InsightFace `buffalo_l` model. The primary drift measure is:

```text
drift(i,j) = 1 - cosine_similarity(e_i, e_j)
```

Cosine distance is primary. Euclidean distance is secondary for normalized-vector consistency. No alternative primary metric, Stability Index formula, or re-enrollment threshold is frozen here.

## Planned experiments

### EXP-A - Reference drift

For identities with at least two observations, compare the analytical reference with every later ordered observation. Preserve identity, image names, ages, age gap, and split.

### EXP-B - Consecutive drift

For identities with at least three observations, sort by filename age then filename and compare adjacent observations. Consecutive drift is distinct from cumulative reference drift.

### EXP-C - Age-gap drift

Compare reference and consecutive drift distributions across the five predefined age-gap groups. Analyze identity clustering and report effect sizes, confidence intervals, identity counts, and sample counts.

### EXP-D - Individual stability

Summarize identity-level mean, median, standard deviation, minimum, maximum, valid-pair count, observation count, and age span. Do not call this a Stability Index.

### EXP-E - Trajectories

Preserve both `E1 -> E2 -> ...` consecutive drift and `E1 -> E2`, `E1 -> E3`, ... reference drift for identities with at least three observations.

### EXP-F - Return-toward-reference and persistent-shift candidates

For `E1`, `E2`, `E3`, compare `drift(E1,E2)` with `drift(E1,E3)`. A smaller later value is a descriptive return-toward-reference pattern; a larger or sustained value is a persistent-shift candidate. These labels do not assert biological recovery or permanence.

### EXP-G - Drift and recognition

Use genuine and separately defined impostor pairs. Analyze acceptance/error across drift bins at a validation-derived operating point. Drift and cosine similarity for the same pair are mathematically coupled, so no independent correlation claim will be made.

### EXP-H - AgeDB external validation

AgeDB is blocked for image-level experiments until protocol-to-local-image mapping is resolved. Its protocols overlap in identity and are not assumed to be identity-disjoint.

### EXP-I and EXP-J

The Stability Index and adaptive re-enrollment experiments remain future work. Their formulas, weights, and thresholds will be developed only after empirical drift inspection on development/validation identities.

## Threshold and split protocol

- `research_train`: method development and exploratory statistical development.
- `research_validation`: operating-point and methodology parameter selection.
- `research_test`: untouched final evaluation.

No test identity may influence threshold selection, model/feature selection, Stability Index parameters, or re-enrollment thresholds. Candidate threshold approaches are descriptive EER, fixed FAR, and validation-optimized threshold. No final threshold is selected in Stage 3.

## Statistical analysis plan

Longitudinal pair rows are clustered within identity and must not be treated as independent. Primary analysis will aggregate at identity level where appropriate and use identity-cluster bootstrap intervals. Mixed-effects models such as `drift ~ age_gap + (1 | identity)` may be considered only after inspecting the data structure. Benjamini-Hochberg correction will be used for multiple planned comparisons when applicable. Report effect sizes, confidence intervals, sample sizes, and identity counts; p-values are secondary.

## Reproducibility

- Protocol version: `stage3-v1`
- Research split seed: `42`
- MORPH metadata: `data/metadata/morph_research_images.csv`, `morph_research_identities.csv`, `morph_longitudinal_pairs.csv`
- Planned model: InsightFace `buffalo_l` ArcFace recognition model
- Planned embedding dimension: 512
- Planned normalization: L2
- Stage 3 embedding count: 0

The complete experiment registry is in `results/metrics/protocol_definition.json`.

## Methodological concerns and stop conditions

- Age is not capture time.
- The analytical reference is not a real enrollment event.
- MORPH supplied splits were replaced because of identity overlap.
- AgeDB mapping is incomplete and blocks image-level validation.
- Exact duplicate flags are available, but any future independence claim must account for clustering and duplicate policy.
- No causal claims will be made from observational age associations.

Stage 3 stops here. The next approved stage is preprocessing, not embedding extraction.
