# Phase 2 Deep Testing Report

**Date:** 2026-06-09  
**Scope:** Feature, split, target, evaluation, bootstrap, and tuning correctness  
**Verdict:** **FAIL - Phase 2 is not complete**

## Executive Summary

All requested targeted pytest gates passed:

| Gate | Result |
|---|---|
| Feature registry, leakage audit, target encoder, split, bootstrap | 19 passed |
| Calibration chronology, calibration, acceptance schema aliases | 24 passed |
| Baseline, boosting, and advanced trainer outputs | 5 passed |
| Analysis outputs, advanced calibration, hyperparameter tuning | 11 passed, 6 warnings |

These green tests do not establish Phase 2 correctness. Independent source and
test reviews found one P0 methodology violation, multiple P1 defects, and major
missing counterfactual tests. Generated artifacts were not used as proof and may
be stale.

## Critical Findings

### P0 - 2023 selection labels leak into advanced-model calibration

`src/aac_adoption/models/train_advanced.py:151` splits
`split.validation`, while `src/aac_adoption/models/split.py:31` defines that
compatibility frame as 2022 calibration plus 2023 selection. The later portion is
used to fit the calibrator, then the calibrated candidate is evaluated on 2023
and marked selection-eligible around `train_advanced.py:217-270`.

This violates the required protocol:

```text
train 2013-2021 -> calibrate 2022 -> select 2023 -> test 2024-2025
```

2023 labels cannot both fit calibration and select the candidate.

## P1 Findings

1. **Threshold selection uses 2022 and 2023 together.**  
   `src/aac_adoption/analysis/threshold_analysis.py:257-284` uses
   `split.validation`, not `split.selection`. A reviewer probe confirmed that
   changing only 2022 labels changes selected thresholds.

2. **Standalone calibrated models lack 2023 selection metrics.**  
   `src/aac_adoption/models/calibrate.py:200-283` fits on 2022 but emits metrics
   from `split.test` only. It does not emit separate selection/test rows with
   correct eligibility.

3. **Calibrated metrics are omitted from model comparison.**  
   `src/aac_adoption/analysis/model_comparison.py:116-121` does not load
   `calibrated_classification_metrics.csv`. Calibrated candidates therefore
   cannot compete correctly.

4. **Final selection accepts incomplete provenance.**  
   `src/aac_adoption/analysis/model_selection.py:29-52` applies chronology and
   eligibility filters only when columns exist. Untagged rows can remain
   selectable, and calibrated chronology is assigned rather than proven.

5. **Leakage audit can report prohibited columns as safe.**  
   `src/aac_adoption/data/leakage_audit.py:6-55` audits only two intake fields
   and initializes other columns as safe. Targets, identifiers, raw timestamps,
   outcome fields, and future-derived fields can receive false-safe status.
   `scripts/generate_leakage_audit.py:79-105` checks `LEAKAGE_COLUMNS`, not the
   broader `PROHIBITED_MODEL_COLUMNS`.

6. **Canonical target failures are masked.**  
   `src/aac_adoption/analysis/hypothesis_tables.py:16-21` silently reconstructs
   canonical targets from aliases. `h3_age_evidence.py:29` and
   `h5_covid_evidence.py:33-39` can emit empty or partial evidence rather than
   fail when required targets are missing.

7. **Legacy tuning failures remain implicit.**  
   `src/aac_adoption/optimization/hyperparam_tuning.py:104-127,182-206`
   catches broad exceptions and can return `best_params=None` without an
   explicit failed status. `train_boosting.py:387-394` and
   `train_advanced.py:416-423` reject only `status == "failed"`; malformed or
   null parameter payloads can silently fall back to defaults.

8. **Standalone evaluation helper is broken.**  
   `src/aac_adoption/models/evaluate.py:145` calls `pd.DataFrame` without
   importing pandas. A direct reviewer probe raised `NameError`.

## P2 Findings

- `tests/test_split.py:19` preserves the dangerous combined
  `validation == 2022-2023` alias without proving downstream code uses explicit
  calibration and selection frames.
- `model_comparison.py:145-149` displays ROC-AUC-first ranking while final
  classification selection is PR-AUC-first.
- `threshold_analysis.py:276-281` catches prediction failures and returns,
  allowing missing threshold evidence without an explicit failed artifact.
- `h3_age_evidence.py:263` still says adopted timing may use
  `days_to_outcome`; canonical wording should require adopted-only
  `days_to_adoption`.
- `docs/target_definitions.md` lacks a canonical `adopted_in_*` horizon-target
  section.

## Contracts That Passed

- `split.py` defines train 2013-2021, calibration 2022, selection 2023, and test
  2024-2025.
- Random fallback requires explicit opt-in and is marked
  `random_development_only` / non-thesis.
- Core dataset construction correctly distinguishes `classification_target`,
  all-outcome `regression_target_days`, and adopted-only `days_to_adoption`.
- Adopted-only regression filters before splitting.
- Cluster bootstrap multiplicity is preserved by
  `src/aac_adoption/models/bootstrap.py`, with direct coverage in
  `tests/test_bootstrap.py`.

## Missing Required Regression Tests

1. Advanced calibrator fits exclusively on 2022; changing 2023 labels cannot
   change fitted calibration.
2. Thresholds use 2023 only and remain invariant when 2022 or 2024-2025 labels
   change.
3. Model A wins 2023 but loses test; model B loses 2023 but wins test; model A
   must remain selected. Repeat for regression.
4. Calibrated artifacts emit one 2023 selection row and one 2024-2025
   reporting-only row, then compete in model comparison.
5. Final selection rejects rows missing split, eligibility, thesis-evaluation,
   artifact, and calibration provenance.
6. Exact split-period and random-fallback metadata assertions.
7. OOF target encoding proves each row excludes its own target.
8. Leakage audit rejects every prohibited target, identifier, timestamp,
   outcome, and future-derived field.
9. All-pruned, all-failed, malformed, and null tuning results produce explicit
   failure and are rejected by trainers unless development fallback is explicit.
10. Analysis producers fail when canonical target columns are absent.

## Evidence Limits

- Full `python -m pytest -q` was not run.
- Long acceptance and pipeline regeneration were not run.
- Generated model and report artifacts may be stale.
- Static type/LSP tools were unavailable.

## Final Decision

Phase 2 must be marked **Blocked / Needs Attention**, not Complete. Task 7 and
cluster bootstrap have substantial passing evidence. Tasks 6, 6A, 8, 9, and 11
remain open because current code either violates the thesis methodology or lacks
tests capable of detecting the violation.
