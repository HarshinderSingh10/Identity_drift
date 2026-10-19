# Content verification log

_Updated in Phase 2. Every item below needs a decision from the authors before the site is published or presented._

The website content is taken from the **IEEE-format paper** ("Longitudinal Embedding Drift in Facial
Representations") as the primary source. The **synopsis** ("Identity Drift Detection in Face Recognition
Systems") describes a different experimental setup. The two are **not merged** anywhere on the site.
The items below need a decision from the team before the site is published or presented.

## A. Conflicts between the paper and the synopsis

| Topic | IEEE paper (used on site) | Synopsis (NOT used on site) |
|---|---|---|
| Primary dataset | MORPH (50,015 raw images; longitudinal) | AgeDB-30, CALFW, CPLFW, LFW (aligned verification benchmarks) |
| AgeDB role | Audited only; not used as main evidence (15,716 ambiguous/unresolved records) | Primary drift-analysis dataset (AgeDB-30) |
| Threshold | 0.2068 similarity / **0.7932 drift**, chosen on validation at 1% impostor FAR, then frozen | **0.65 drift**, chosen by visual inspection of the AgeDB-30 histogram |
| What the threshold means | A verification operating boundary, explicitly *not* a re-enrolment threshold | A re-enrolment trigger |
| Headline metrics | Test ROC-AUC 0.996201, EER 0.003745, FAR 0.008592, FRR 0.007270 | Accuracy 99.67% (LFW), 94.02% (CALFW), 92.68% (AgeDB-30), 88.73% (CPLFW) |
| Re-enrolment | Future work; not validated | "Proposes and implements" a system that recommends re-enrolment; "re-enrolment rate" 0.7%–22.5% |
| Causality | Explicitly non-causal | "Aging is the primary driver of identity drift"; "directly validating the core hypothesis" |
| Stability index | Identity Stability Index (ISI), exploratory, added nothing beyond current drift | Stability Index = mean cosine similarity over genuine pairs, per dataset |
| Pose vs. aging | Not studied | Claims pose is harder than aging (CPLFW) |
| Embedding pipeline | InsightFace detection + alignment of raw MORPH images | Pre-aligned 112x112 images, detector bypassed |

**Suggested resolution:** keep the paper as the single source for the website, and either update the
synopsis to match, or add a clearly labelled "earlier scope" note. Do not show synopsis accuracy figures
beside the paper's metrics.

## B. Inconsistencies and gaps inside the IEEE paper

Numbers marked **(our arithmetic)** were computed from values printed in the paper; they are not stated by the paper.

1. **Pair counts do not sum.** Table I: 84,133 eligible pairs. Table II: 59,545 + 12,303 + 12,289 = **84,137** (our arithmetic; difference of 4). The site shows both as printed.
2. **Two sets of age-gap mean drift values.** Section V-B text: 0.236 / 0.261 / 0.305 / 0.375 / 0.774 ("MORPH pairs analyzed here"). Table IV (test split): 0.2382 / 0.2693 / 0.3363 / 0.3662 / 0.6681. The n labels in Fig. 4 (77,762 + 4,676 + 922 + 660 + 113) sum to 84,133 (our arithmetic), which suggests the text values describe all eligible pairs, but the paper never says so. The site shows the text values in a separate, labelled table and does not merge them with Table IV.
3. **Table III FRR vs. Table II / Table IV (verified arithmetic, Phase 3).** Table III's FRR values are not whole-number fractions of the Table II pair counts in any split: train 0.005250 × 59,545 = 312.6; validation 0.005501 × 12,303 = 67.7; test 0.007270 × 12,289 = 89.3 (no integer k/N reproduces the printed 6-decimal values). Table IV's percentages do correspond to whole numbers of rejects (8, 2, 14, 10, 19 = 53 of 12,289, about 0.43%), and its pair counts sum to 12,289. So Table III does not appear to use the same genuine pair set as Table II/IV. The authors should state the genuine and impostor set sizes behind Table III. The site flags this on the Results page.
4. **Table V has no stated prediction target.** ROC-AUC 0.661054 / PR-AUC 0.367662 / Brier 0.037012 are given without saying which outcome is predicted. The site names none. PR-AUC and Brier are not defined in the paper.
5. **Image totals.** Table II images sum to 50,015 although 50,013 were processed (2 failures). Table II identities sum to 19,033, while only 13,119 identities have two or more observations (our arithmetic: roughly 5,900 identities contribute no pairs).
6. **Drift-bin analysis is qualitative only.** Section V-E and the abstract say failure rates rose "for high-drift comparisons", but no per-drift-bin numbers or table are printed (only Fig. 6, the genuine drift histogram with the boundary). The site shows no bin-level values.
7. **Unexplained internal terms.** "Stage 8 bootstrap", "corrected production artifact" (Section VII) and how the training-derived trajectory threshold 0.361454629 was derived are not described.
8. **Reference [7] fit.** Section II cites [6], [7] for "age-invariant recognition and template aging", but [7] is titled "Age estimation from face images: Human vs. machine performance". Shown exactly as printed; verify the citation.
9. **Missing publication details.** The PDF gives no venue, year, DOI, acknowledgements, contact details or author biographies, so none appear on the site.
10. **Spelling.** The paper uses American spelling ("enrollment", "characterize"); Phase 1 used British spellings in places. Phase 2 follows the paper.

11. **Which pairs feed which analysis is unclear.** The 84,133 eligible pairs cannot all be reference-anchored: with 13,119 identities having two or more observations and 50,015 images in total, reference-anchored pairs number at most 30,982 (our arithmetic). The eligible pairs are therefore same-identity pairs in general, while Fig. 7 and Table V use "reference drift" against the youngest observation. The paper does not say whether the identity-level drift summary (Section V-A) uses all pairs or reference drift.
12. **Identity-level summary scope.** The 13,119 identities in Section V-A equal the whole-cohort count in Section III-C, so the summary appears to cover all splits rather than the test split; not stated explicitly.
13. **Observation-count qualifier.** Section III-C says "326 identities with at least three observations spanning at least five years, 145 spanning at least ten years, and 26 spanning at least twenty years". It is unclear whether "at least three observations" also applies to 145 and 26. The site labels them without the qualifier.
14. **Threshold wording.** Section V-C says the test result "remained strong at the frozen verification threshold" and lists ROC-AUC and EER with FAR and FRR. ROC-AUC and EER do not depend on a threshold; only FAR and FRR do. The site labels them accordingly.

## C. Other items to verify

- **Removed in Phase 2 because the paper does not state them:** supervisor ("Sunita Kumari", listed on the synopsis as "Mr. Sunita Kumari, Assistant Professor"), degree (B.Tech CSE), session (2025-26), affiliating university (Dr. A.P.J. Abdul Kalam Technical University) and the phrase "final-year project". Re-add them to `src/config/site.ts` and the Team page only after the team confirms them and agrees they belong on a paper-based site.
- **Website name.** The header/footer name is now "Identity Drift Detection" (the project name used for this website). The exact paper title is the Home page heading. The longer synopsis project title is no longer used.
- **GitHub URL.** None provided; `siteConfig.githubUrl` is `null`, so the navbar shows a non-clickable "GitHub (coming soon)".
- **Paper PDF.** The paper is served from `public/paper/IEEE_major_project.pdf` and linked on the Team page. Delete it (and the link) if you do not want it public.
- **Figures.** Figs. 3, 4, 6, 7 are reproduced from the PDF's embedded images (`public/figures/`) because their underlying values are not tabulated. Fig. 5 is recreated from Table IV as an SVG chart; Fig. 1 is a reconstructed workflow list; Fig. 2 is not reproduced.
- **Privacy.** Student enrolment numbers and the project ID in the synopsis are intentionally not published.

## D. Source traceability

Every reusable statement and table lives in `src/data/researchContent.ts` with a `source` field (paper section, table or figure). Chart data (`ageGapChart`, `identityDriftChart`) are numeric copies of Table IV and Section V-A; they were checked programmatically against the table strings.
