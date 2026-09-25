# Stage 6 — Complete Embedding Drift Analysis

## 1. Objective
This stage characterizes longitudinal changes in fixed MORPH ArcFace embeddings.
It does not evaluate recognition performance, select thresholds, calculate a
Stability Index, or implement re-enrollment.

## 2. Frozen methodological assumptions
The primary metric is cosine distance, `1 - cosine_similarity`. The analytical
reference is the youngest `filename_age` observation with filename tie-breaking.
Filename age is an ordering proxy, not an exact capture timestamp.

## 3. Dataset and embedding inputs
Embeddings used: **50,013**. Longitudinal pair rows supplied:
**84,133**. Identities represented: **19,032**. All inputs
were read from frozen Stage 2--5 artifacts.

## 4–8. Experiments
EXP-A reference drift, EXP-B consecutive drift, EXP-C age-gap drift, EXP-D
descriptive individual stability, and EXP-E trajectories were computed into
separate CSV outputs under `results/metrics/`. EXP-F is an exploratory
persistent-versus-temporary candidate analysis.

## 9. EXP-F Persistent vs Temporary Shift
The exploratory excursion threshold was the research-train 95th percentile of
trajectory reference drift: **0.36145462863255484**. It was selected without using
validation or test identities. Candidate returns are descriptive decreases
toward the analytical reference after an elevated observation; they do not
indicate return to a person's true identity representation. Candidate count:
**830**.

## 10. Statistical methodology
Pair-level distributions are descriptive only because observations cluster by
identity. Identity-level summaries are provided. The manifest records a fixed
seed of 42 and an identity-cluster bootstrap configuration of 500 repetitions
for reusable inferential extensions. No test-set parameter was used.

## 11. Outlier analysis
Extreme drift cases are written to `stage6_extreme_drift_cases.csv` without
removal. No near-duplicate embedding search was performed.

## 12. Data-quality checks
- Invalid records: **0**
- Duplicate longitudinal pairs: **0**
- Maximum Euclidean/cosine consistency error: **2.39935832091e-07**
- Mean Euclidean/cosine consistency error: **1.52635080662e-08**

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
Version: `stage6-v1`; seed: `42`; model: InsightFace
`buffalo_l` ArcFace; embeddings were not regenerated.

## 16. Stage 6 conclusion
Stage 6 produced reproducible descriptive drift measurements and quality checks.

## 17. Readiness for Stage 7
The outputs are ready for a separately approved recognition analysis. Stage 7
was not started.
