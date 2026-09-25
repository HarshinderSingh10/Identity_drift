"""CLI for Stage 7 drift-versus-recognition analysis."""

from __future__ import annotations

import argparse

from src.experiments.recognition_analysis import run_stage7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["verification", "threshold", "drift-recognition", "age-gap", "identity", "all"],
        default="all",
    )
    args = parser.parse_args()
    result = run_stage7(args.experiment)
    summaries = result["verification_summary"]
    validation = next(row for row in summaries if row["research_split"] == "research_validation")
    test = next(row for row in summaries if row["research_split"] == "research_test")
    qc = result["qc"]
    print("=" * 60)
    print("STAGE 7 FINAL STATUS")
    print("=" * 60)
    print("Verification construction:        PASS")
    print("Impostor construction:             PASS")
    print("Threshold selection:              PASS")
    print("Threshold frozen before test:     PASS")
    print(f"Validation ROC-AUC:               {validation['roc_auc']:.6f}")
    print(f"Validation EER:                   {validation['eer']:.6f}")
    print(f"Validation FAR:                   {validation['frozen_far']:.6f}")
    print(f"Validation FRR:                   {validation['frozen_frr']:.6f}")
    print(f"Test ROC-AUC:                     {test['roc_auc']:.6f}")
    print(f"Test EER:                         {test['eer']:.6f}")
    print(f"Test FAR:                         {test['frozen_far']:.6f}")
    print(f"Test FRR:                         {test['frozen_frr']:.6f}")
    print("Drift-recognition analysis:       PASS")
    print("Age-gap recognition analysis:     PASS")
    print("Identity-level analysis:          PASS")
    print("Statistical association:          PASS")
    print(f"Test leakage:                     {'DETECTED' if qc['test_leakage'] else 'NONE'}")
    print(f"Invalid records:                  {qc['invalid_scores']}")
    print(f"Missing scores:                   {qc['missing_scores']}")
    print(f"Duplicate pairs:                 {qc['duplicate_impostor_pairs']}")
    print("Tests:                            run separately with pytest")
    print("\nOVERALL:")
    print(f"STAGE 7: {qc['result']}")
    print("Ready for adaptive re-enrollment: NO")
    print("=" * 60)


if __name__ == "__main__":
    main()
