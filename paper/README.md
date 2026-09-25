# Research Paper: Longitudinal Embedding Drift in Facial Representations

This directory contains the LaTeX source and generated figures for the study described in the project.

## Contents

- `main.tex` — IEEEtran conference paper source
- `references.bib` — bibliography
- `figures/` — vector/PDF and PNG previews used in the paper
- `output/` — compiled PDF output
- `tables/` — reserved for any table-specific LaTeX fragments, if needed later
- `generate_figures.py` — script that regenerates the figures using the project result artifacts

## Major result sources

The paper is based on the following project artifacts:

- `results/metrics/stage6_identity_stability_summary.csv` — identity-level drift summary
- `results/metrics/stage6_age_gap_drift_summary.csv` — age-gap drift means and spreads
- `results/metrics/stage6_longitudinal_trajectories.csv` — representative trajectories
- `results/metrics/stage7_verification_summary.csv` — verification metrics by split
- `results/metrics/stage7_age_gap_recognition.csv` — age-gap FRR and acceptance metrics
- `results/metrics/stage7_frozen_threshold.json` — frozen verification threshold
- `results/metrics/stage8_index_comparison.csv` — exploratory ISI comparison
- `data/metadata/stage6_drift_manifest.json` and `data/metadata/stage7_recognition_manifest.json` — reproducibility metadata

## Compile instructions

1. Ensure a LaTeX distribution is installed and available on PATH.
2. From this directory, run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled PDF will appear in the working directory as `main.pdf` or in `output/main.pdf` if copied there.

## Stage 8 status

The exploratory ISI analysis in Section VII is explicitly non-conclusive. The production audit showed that the corrected ISI formulation did not add independent predictive information beyond current drift. The manuscript therefore presents this section as exploratory and methodological, not as a validated contribution.

## Figure regeneration

Run:

```bash
python generate_figures.py
```

This regenerates the publication-quality figures using the project-metrics CSV files, ensuring that the document reflects the actual recorded MORPH results.
