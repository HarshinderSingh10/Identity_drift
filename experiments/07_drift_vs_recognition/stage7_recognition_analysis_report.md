# Stage 7 — Drift vs Recognition

## 1. Objective
Evaluate whether larger longitudinal embedding drift is associated with higher
genuine verification error. This stage does not claim that drift causes failure
and does not implement adaptive re-enrollment.

## 2. Research question
Does greater drift correspond to a higher probability of verification failure?

## 3. Frozen methodology
The pretrained InsightFace `buffalo_l` ArcFace embeddings and Stage 6 cosine
drift values were used unchanged. The analytical reference is the youngest
`filename_age` observation with filename tie-breaking. Filename age is an
ordering proxy, not an exact capture timestamp.

## 4–6. Verification dataset construction
The primary genuine task uses **30,981** non-reference observations
against the analytical reference. Self-comparisons were excluded. A balanced
set of **30,981** within-split, different-identity image pairs was
sampled deterministically with seed 42. All impostors remain within one
research split; exact age-distribution matching was not imposed.

## 7. Threshold selection
The primary threshold was selected only on `research_validation` using the
declared empirical 1% impostor FAR rule and frozen before test evaluation.
Threshold: **0.206770330667**; validation FAR:
**0.010157**; validation FRR:
**0.005501**.

## 8–9. Validation and test performance

| Split | ROC-AUC | EER | Frozen-threshold FAR | Frozen-threshold FRR |
|---|---:|---:|---:|---:|
| Validation | 0.996745 | 0.002751 | 0.010157 | 0.005501 |
| Test | 0.996201 | 0.003745 | 0.008592 | 0.007270 |

## 10–12. Drift, age-gap, and identity analyses
Machine-readable drift-bin, age-gap, and identity-level acceptance results are
provided under `results/metrics/`. Confidence intervals for genuine acceptance
use identity-cluster bootstrap resampling. No drift/similarity correlation was
calculated because drift is mathematically `1 - cosine_similarity`.

## 13. Statistical association
The identity-aggregated binomial logistic association estimate for verification
error versus mean identity drift was **23.449044**
log-odds per unit drift (odds ratio **15268289091.142790**),
95% CI **[21.094738, 26.334882]**, bootstrap
sign p-value **0.000000**. This is an
analysis model, not a recognition model or production policy.

## 14–16. Findings and limitations
The results describe association only. MORPH observations vary in pose,
lighting, expression, hairstyle, facial hair, camera, compression, and image
quality, so age gap is not a controlled causal treatment. Individual
heterogeneity is retained rather than converted into a re-enrollment rule.

## 17–18. Conclusion and readiness
Stage 7 completed the frozen drift-versus-recognition evaluation. Adaptive
re-enrollment was not implemented. Stage 8/9 may investigate policy only after
separate approval.

Final QC result: **PASS**
