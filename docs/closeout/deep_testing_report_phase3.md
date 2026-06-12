# Phase 3 Deep Testing Report

**Date:** 2026-06-09  
**Auditor:** ML Systems Auditor (subagent)  
**Scope:** Tasks 12 & 13 — Yearly Backtesting and PR-AUC / Report Generation  
**Project root:** `c:\Users\paula\Documents\mgr pjatk`

---

## Executive Summary

| Check | Status | Severity | Notes |
|-------|--------|----------|-------|
| Window chronological ordering (train < test) | ✅ PASS | — | Disjoint years prevent any future leakage. |
| First-window 'insufficient history' handling | ✅ PASS | — | Correctly logs warning and records status as `SKIPPED`. |
| Impossible-window guard | ✅ PASS | — | Handled gracefully via dataset checks and skipping. |
| Horizon target requires horizon dataset | ✅ PASS | — | Guard present in `yearly_backtesting.py` and checked. |
| PR-AUC computation | ✅ PASS | — | Standard `average_precision_score` from sklearn. |
| PR-AUC as primary metric | ✅ PASS | — | Used as primary sorting key for classification models. |
| Schema change guards | ✅ PASS | — | Report generation correctly rejects tables without `metric_split`. |
| Separation of 2023 selection and test rows | ✅ PASS | — | Separation done via `metric_split` filtering. |
| Regression test suites execution | ✅ PASS | — | All tests in suite pass successfully. |

---

## Task 12: Backtesting Window Chronological Validity

### File: `src/aac_adoption/models/yearly_backtesting.py`

#### Check 1: Chronological Ordering & Bounds
The backtesting implementation uses a rolling window partition where train ends before test start:
* **Train period:** `TRAIN_START_YEAR` (2013) to `test_year - 1`
* **Test period:** `test_year`
Since `train_end = test_year - 1`, the masks are completely disjoint. No future data leaks into earlier windows.

#### Check 2: First-Year Semantics
When there is insufficient history for the first window (e.g. `test_year = 2019` when the dataset starts in `2019`), the training subset evaluates to empty. The code gracefully skips training, logs a warning, and appends a skipped metadata row to the output with `"status": "SKIPPED"` and `"skip_reason": "Empty train set"`.

#### Check 3: Horizon Targets Dataset Check
In `run_yearly_backtesting` (lines 98-103), dataset validation guards are enforced:
* If target column starts with `adopted_in_` (horizon target), it enforces `horizon_modeling_dataset.csv`.
* If target is `classification_target` or `regression_target_days`, it enforces `modeling_dataset.csv`.
This prevents mixing datasets.

#### Check 4: Row Skipping Threshold
Rows/windows with insufficient samples (`<2` rows) are gracefully skipped and recorded as `SKIPPED` in the output DataFrame.

---

## Task 13: PR-AUC and Report Generation

### File: `src/aac_adoption/reporting/report.py`

#### Check 1: PR-AUC Computation
PR-AUC is correctly computed via sklearn's `average_precision_score` in `src/aac_adoption/models/evaluate.py`:
```python
metrics["pr_auc"] = average_precision_score(y_true, y_score)
```
This is the appropriate standard for imbalanced classification tasks.

#### Check 2: Classification Model Selection
In `src/aac_adoption/analysis/model_selection.py`, candidate models are sorted using `pr_auc` as the primary key:
```python
sort_keys = ["pr_auc", "brier_score", "expected_calibration_error", "roc_auc"]
ascending = [False, True, True, False]
```
PR-AUC is prioritized over ROC-AUC.

#### Check 3: Schema Guard & Split separation
`create_report_outputs` (lines 345-353) checks for the presence of `metric_split` columns in model comparison CSV files:
```python
if not {"metric_split"}.issubset(classification.columns):
    raise ValueError("Report generation must reject model comparison data without split/source metadata.")
```
Additionally, `_best_rows()` filters on `metric_split == "selection"` (representing the 2023 validation period) before determining the best models. This ensures selection and test rows are kept separate.

---

## Test Run Results

The following test suites were audited and verified passing:
* `tests/test_yearly_backtesting.py`
* `tests/test_report_outputs.py`
* `tests/test_backtesting.py`

### Test output:
```
python -m pytest tests/test_yearly_backtesting.py tests/test_report_outputs.py tests/test_backtesting.py -q
17 passed, 6 warnings in 99.32s (0:01:39)
```

---

## Bugs Found & Resolved

1. **`test_report_outputs.py` Schema Discrepancy**  
   * *Problem:* The test fixture was missing the `metric_split` column, which triggered the new schema checks and caused the test to fail.
   * *Solution:* Added `"metric_split": "selection"` to the fixture DataFrames. Updated the test assertion to expect the updated PR-AUC text instead of ROC-AUC.
   
2. **`test_yearly_backtesting_horizon_targets` Quick Mode expectation**  
   * *Problem:* The test was calling `run_yearly_backtesting` with `quick=True` but expecting all 6 test years in the output.
   * *Solution:* Updated the assertion to verify the correct quick-mode years (`[2019, 2023]`) and added validation test cases for horizon target dataset guards.

3. **`test_yearly_backtesting_empty_splits_skipped` Assertion**  
   * *Problem:* The test expected `len(result) == 0` for skipped splits, but `run_yearly_backtesting` returns skipped rows with status `"SKIPPED"`.
   * *Solution:* Adjusted assertion to expect the metadata rows with status `"SKIPPED"`.

4. **Pandas masking error on SKIPPED rows**  
   * *Problem:* Standard series contains masking crashed on rows where the model was `None` (for skipped years) due to NA values.
   * *Solution:* Added `na=False` to all `.str.contains` filters in the test suite.

---

## Recommendations

1. **Keep `na=False` in String Masks:** Always ensure that boolean indexers check for `NaN` values via `na=False` when filtering model columns in results.
2. **Horizon Target Unit-Test Coverage:** Keep validation tests active to ensure that future schema or target modifications trigger explicit path failures.
