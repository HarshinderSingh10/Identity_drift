# Research accuracy audit (Phase 3, Step 1)

**Authoritative artifact available:** the paper `IEEE_major_project.pdf` only. No experiment outputs (notebooks, CSV/JSON results, logs, figures' source data) were provided, so values could be checked against the paper (tables, text, figure labels, internal arithmetic) but **not** against raw experiment data. Anything that needs the raw outputs is listed under "Unresolved".

## 1. Scope inspected
Six pages (`Home`, `Research`, `Methodology`, `Results`, `Demo`, `Team`), `src/data/researchContent.ts`, `src/data/references.ts`, `src/lib/drift.ts`, `src/config/site.ts`, chart components, and `docs/CONTENT_VERIFICATION.md`.

## 2. Confirmed (matches the paper)
- All table values in `researchContent.ts` (Tables I–V, Section V-A statistics, Section V-B text means, coverage counts) and the study constants (50,015 / 50,013 / 19,033 / 84,133, seed 42, 1% FAR rule, threshold 0.206770330667, drift boundary 0.793229669333, trajectory threshold 0.361454629) were matched against the PDF text.
- `1 − 0.206770330667 = 0.793229669333` (exact).
- Chart data (`ageGapChart`, `identityDriftChart`) equals the table strings (checked by script).
- Table IV: acceptance + FRR = 100% in every row; FRR percentages correspond to whole-number rejects; pair counts sum to 12,289 (= Table II test pairs).
- Table II: identities sum to 19,033, images to 50,015; Fig. 4 n labels sum to 84,133 (= Table I eligible pairs).
- Percentile ordering is monotonic; coverage counts are monotonic and compatible with 84,133 all-pairs (lower bound 48,515).
- The drift formula, its [0, 2] range (demo), and the demo's 13 unit tests.

## 3. Discrepancies found and what was done
| # | Finding | Type | Action |
|---|---|---|---|
| 1 | Home labelled ROC-AUC (and implicitly EER) "at the frozen threshold"; both are threshold-independent | Site label error (mine) | Fixed on Home, Results, Methodology |
| 2 | Results scope note said all values use one frozen threshold | Site label error (mine) | Reworded: only FAR/FRR/acceptance use it |
| 3 | Home "Mean identity-level drift" was ambiguous | Label | Now "identity-level mean drift, averaged over 13,119 identities" |
| 4 | Embedding model shown as "buffalo_l (ArcFace-compatible)"; the paper says "ArcFace-compatible" about alignment | Misattribution (mine) | Reworded to "InsightFace buffalo_l model" |
| 5 | Methodology cited [3] for buffalo_l; the paper does not | Unsupported citation (mine) | Citation removed |
| 6 | Coverage rows 6–7 stated "at least three observations" for 145 and 26; the paper says it only for 326 | Over-specification (mine) | Qualifier removed, with note |
| 7 | Methodology listed SD/percentiles "per age-gap group"; paper gives only mean drift per group (plus Fig. 4 box plot) | Label | Reworded |
| 8 | Identity-level summary scope (all splits vs test) unstated | Clarity | Caption note added (inference, labelled as such) |
| 9 | Table III FRR is not an integer fraction of Table II counts in any split; Table IV implies 53/12,289 (0.43%) vs 0.007270 | **Paper inconsistency** | Not changed; documented, callout strengthened |
| 10 | 84,133 vs 84,137 pair totals | Paper inconsistency | Unchanged (already shown as printed) |

No numeric value on the site was changed: none could be shown wrong from authoritative data.

## 4. Unresolved (need authors / raw experiment outputs)
1. Genuine and impostor set sizes behind Table III, and why its FRR is not k/N for the Table II counts (item 9).
2. Whether Section V-A identity summary uses all pairs or reference drift; whether it covers all splits.
3. Which population the Section V-B means describe (Fig. 4 n-sum suggests all pairs).
4. The outcome predicted in Table V; definitions of PR-AUC/Brier; ISI = baseline exactly (correlation 1.0) suggests the ISI had no effect in the implementation.
5. Per-drift-bin numbers behind Section V-E and the abstract's "high-drift comparisons" claim.
6. 84,133 vs 84,137 pairs; 50,013 processed vs 50,015 in Table II images.
7. Derivation of the 0.361454629 trajectory threshold; "Stage 8 bootstrap"; reference [7] fit.
See `docs/CONTENT_VERIFICATION.md` section B for the full list.
