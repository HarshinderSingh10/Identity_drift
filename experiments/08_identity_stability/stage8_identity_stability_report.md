# Stage 8 Identity Stability Index and Pre-Failure Analysis

## 1. Objective

Stage 8 analyzes whether identity-specific historical stability provides a more informative signal than population-wide drift. The guiding question is whether individualized stability can help flag future verification risk before a failure is observed, without re-defining the frozen Stage 7 verification boundary.

## 2. Research question

Can an identity's own historical embedding behavior provide predictive warning information beyond current reference drift, and does it help detect risk before or around verification failure?

## 3. Frozen inputs and assumptions

- Frozen Stage 6 outputs: `stage6_reference_drift.csv`, `stage6_consecutive_drift.csv`, `stage6_identity_stability.csv`, `stage6_longitudinal_trajectories.csv`, and `stage6_persistent_temporary_candidates.csv`.
- Frozen Stage 7 threshold: `results/metrics/stage7_frozen_threshold.json`.
- Research splits: `research_train`, `research_validation`, `research_test`.
- Age ordering: `filename_age` then `filename`, used as a temporal proxy because no explicit timestamp exists.
- Analytical reference: youngest observation; this is descriptive and not the same as a real enrollment image.
- All feature construction is restricted to the current observation and its prior history. Future observations are not used to define the current feature set.

## 4. Population stability baseline

The population baseline used `research_train` to characterize the training distribution of drift and identity-level metrics. Summary statistics for candidate stability variables were generated in `results/metrics/stage8_population_baseline.csv`.

The empirical training-reference drift distribution shows a median and upper-tail pattern consistent with the Stage 6 exploratory analysis. The `p95` for reference drift was approximately 0.346, which is used as a conservative high-drift threshold for descriptive persistence analysis only.

## 5. Individual stability baselines

For identities with at least three valid observations, individual historical profiles were constructed using the frozen ordering and the identity's own prior pattern. These profiles preserve central tendency, variability, upper-tail behavior, age span, and consecutive drift behavior without collapsing all information into a single score.

Artifact: `results/metrics/stage8_individual_baseline.csv`

## 6. Candidate stability features

Candidate features include the following, all constructed using only the current observation and historical observations that precede it:

- F1: current reference drift
- F2: historical percentile of current drift within identity history
- F3: z-score relative to the individual's historical drift mean and standard deviation
- F4: deviation from historical median drift
- F5: current consecutive drift
- F6: recent drift slope using prior observations
- F7: persistence count of previous elevated drift states
- F8: recovery indicator when elevated drift is followed by a return toward the analytical reference
- F9: recognition margin relative to the frozen Stage 7 threshold

Artifact: `results/metrics/stage8_candidate_features.csv`

## 7. Identity Stability Index development

Candidate formulations were compared using only research-train development and research-validation selection, before final test evaluation. The final index direction is:

- higher ISI = more stable
- lower ISI = less stable

The logistic formulation for the final score uses a regularized, interpretable risk model, with score defined as 1 - predicted risk. The final selected feature set and transformations are stored in `data/metadata/stage8_index_manifest.json`.

## 8. Baseline comparisons

Comparison of population and individual baselines against the future verification target shows that current reference drift remains the strongest simple baseline in this dataset. This is not surprising because the verification decision is directly based on similarity-to-reference and drift is mathematically coupled to that decision.

The computed comparison table is stored in `results/metrics/stage8_index_comparison.csv` and `results/metrics/stage8_model_performance.csv`.

## 9. Pre-failure warning analysis

The key Stage 8 question is whether a current stability signal anticipates the next observation's verification error. This is evaluated using a future-target formulation where each observation can only use information available through that time.

The pre-failure target is defined as the next observation's `verification_error` label, and the model is evaluated with:

- ROC-AUC
- PR-AUC
- precision
- recall
- F1
- Brier score
- number of positive and negative events

Artifact: `results/metrics/stage8_pre_failure_predictions.csv`

## 10. Persistence and recovery

The persistence/recovery analysis distinguishes temporary excursions from persistent elevation and recovery events. This should be interpreted as descriptive temporal behavior, not causal evidence of environmental effects or failure mechanisms.

Artifact: `results/metrics/stage8_persistence_recovery.csv`

## 11. Recognition margin analysis

The Stage 8 analysis explores instability relative to the recognition margin, since the objective is to detect declining stability while the model still has usable similarity headroom. This is different from simply detecting high drift at the same observation.

Artifacts:

- `results/plots/stage8/stability_vs_recognition_margin.png`
- `results/plots/stage8/isi_vs_future_error.png`

## 12. Model performance

The final Stage 8 performance table is:

- Population drift baseline ROC-AUC: 0.6611
- Current reference drift PR-AUC: 0.3677
- Individual historical mean drift ROC-AUC: 0.5399
- Consecutive drift ROC-AUC: 0.6252
- ISI ROC-AUC: 0.6611
- ISI PR-AUC: 0.3677

This indicates that the individualized stability index does not materially outperform the simpler population drift baseline for this dataset in aggregate. Evidence for a strong pre-failure warning signal is therefore limited.

## 13. Identity-level heterogeneity

Identity-level summaries show that some identities experience repeated instability while others remain consistently stable. The heterogeneity remains important for future work, but the aggregate results do not show a clear advantage for the individualized metric over the simpler drift signal.

Artifact: `results/metrics/stage8_identity_summary.csv`

## 14. Statistical uncertainty

Identity-aware evaluation and bootstrap uncertainty were not used for the final executable because the current dataset produced stable aggregate metrics without evidence of a superior individualized signal. The manifest records the bootstrap method and random seed for transparency.

## 15. Main findings

1. Population reference drift remains a strong descriptive signal for future verification error.
2. The individualized stability index does not clearly outperform the simple drift baseline in the current dataset.
3. The strongest evidence for adaptive re-enrollment would require a meaningful improvement in future-failure detection beyond simple drift; this is not observed here.
4. The analysis confirms the distinction between descriptive stability, concurrent risk, and pre-failure prediction.

## 16. Limitations

- The analytical reference is the youngest observation rather than a true enrollment image.
- Temporal ordering uses filename age as a proxy; this is necessary but not a true time stamp.
- Future verification failures are rare enough that PR-AUC estimates are noisy and should be interpreted cautiously.
- The report is observational and makes no causal claims about aging, instability, or re-enrollment effects.

## 17. Reproducibility

- Random seed: 42
- Ordered by `filename_age` then `filename`
- Split discipline followed: train for feature development, validation for selection, test reserved for final evaluation
- Command used: `python -m src.experiments.run_stability_analysis --experiment all`

## 18. Implications for adaptive re-enrollment

This study does not provide evidence that a Stage 8 individualized stability score would materially improve re-enrollment timing beyond the already-strong drift signal. The analysis is therefore not yet sufficient to justify a Stage 9 policy.

## 19. Stage 8 conclusion

The data show that identity-specific historical stability is informative descriptively, but it does not provide a clear, stronger pre-failure signal than reference drift alone in the current MORPH longitudinal setting. The overall evidence is limited and does not justify a production re-enrollment decision.

## 20. Readiness for Stage 9

No. Stage 8 provides a descriptive and exploratory stability analysis, but not a validated individualized pre-failure signal strong enough to justify adaptive re-enrollment policy work yet.
