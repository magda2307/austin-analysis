# Phase 3 Deep Testing Report
**Date:** 2026-06-09  
**Auditor:** ML Systems Auditor (subagent)  
**Scope:** Tasks 12 & 13 — Yearly Backtesting and PR-AUC / Report Generation  
**Project root:** `c:\Users\paula\Documents\mgr pjatk`

---

## Executive Summary

| Check | Status | Severity |
|-------|--------|----------|
| Window chronological ordering (train < test) | ✅ PASS | — |
| First-window 'insufficient history' handling | ⚠️ WARNING | Medium |
| Hardcoded test-year list (not dynamic) | ⚠️ WARNING | Medium |
| Impossible-window guard (train_end < train_start) | ⚠️ WARNING | Medium |
| Min-sample threshold (audit: <2 rows, spec says <10) | ⚠️ WARNING | Medium |
| Horizon target requires horizon dataset | ✅ PASS | — |
| PR-AUC correctly computed via `average_precision_score` | ✅ PASS | — |
| PR-AUC as primary classification metric in `_select_classification` | ✅ PASS | — |
| PR-AUC figure NOT generated in `create_report_outputs` | ❌ FAIL | High |
| `create_report_outputs` rejects missing `metric_split` column | ✅ PASS (contract) | — |
| Test `test_report_outputs` fixture missing `metric_split` → test always fails | ❌ FAIL | Critical |
| Test `test_report_outputs` text assertion stale ("ROC-AUC" vs "PR-AUC") | ❌ FAIL | High |
| Test `test_yearly_backtesting_horizon_targets` quick=True expects all 6 years | ❌ FAIL | Critical |
| Test `test_yearly_backtesting_empty_splits_skipped` expects len=0 but gets SKIPPED rows | ❌ FAIL | Critical |
| Test `test_yearly_backtesting_output_csv` round-trip equality may fail (float dtype) | ⚠️ WARNING | Medium |
| 2023 selection vs calibration rows correctly separated in split.py | ✅ PASS | — |
| Dashboard does not train — no `create_report_outputs` call in streamlit | ✅ PASS | — |

**Test run summary (observed):**
- `test_yearly_backtesting.py`: at least 3 FAIL (still running as of report time)
- `test_report_outputs.py`: 1 FAIL
- `test_backtesting.py`: 1 PASS

---

## Task 12: Backtesting Window Chronological Validity

### File: `src/aac_adoption/models/yearly_backtesting.py`

#### Check 1: Train ends before calibration/selection/test — PASS ✅

The backtesting implementation uses a rolling expand-window approach. For a given `test_year`:

```python
# Lines 127-131
train_end_year = test_year - 1
train_start_year = max(min_year, TRAIN_START_YEAR)
if train_start_year > train_end_year:
    train_start_year = min_year
train_start, train_end = get_train_years(test_year, max_train_year=train_end_year, min_year=train_start_year)
```

```python
# Lines 134-135
train_mask = subset_df["intake_year"].between(train_start, train_end)
test_mask = subset_df["intake_year"] == test_year
```

Since `train_end = test_year - 1`, `train_mask` and `test_mask` are disjoint by construction. **No future data can appear in training.** Chronological ordering: PASS.

> [!NOTE]
> There is no explicit calibration/selection period in the backtesting loop. It is a pure train→test evaluation per year, not the thesis 4-split (train/calibration/selection/test). This is intentional for the *backtesting* workflow.

#### Check 2: First-year semantics — WARNING ⚠️

When the earliest year in the dataset equals the first test_year, the code tries to compute `train_end_year = test_year - 1` which may be less than the dataset's minimum year. The guard at lines 129-130 detects this:

```python
if train_start_year > train_end_year:
    train_start_year = min_year
```

But this does **not** fix the problem — it sets `train_start = min_year` while `train_end = test_year - 1 = min_year - 1`. Calling `get_train_years(test_year, max_train_year=min_year-1, min_year=min_year)` returns `start=min_year, end=min_year-1`. The `between(min_year, min_year-1)` mask evaluates to **all False**, producing an empty train set. The empty-train guard at lines 160-179 then appends a SKIPPED row. This is **correct behavior** but the intermediate state creates a logically incoherent `train_start > train_end` period string (e.g., `"2019-2018"`), which is written to the output CSV. A consumer seeing `train_years = "2019-2018"` cannot interpret it correctly.

**Bug:** The SKIPPED row's `train_years` field may read `"2019-2018"` (start > end), which is misleading.

#### Check 3: Impossible windows — WARNING ⚠️

Related to Check 2. If dataset only has one year (e.g., 2020), quick mode will try 2019 and 2023. Neither is in the data, so both are skipped via `if test_year not in years: continue`. **No impossible window is trained.** However, if data has exactly 2 consecutive years starting at X, and X+1 is in quick test_years, then year X is tested with an empty train set (see above). This is handled via SKIP but the period string is problematic.

#### Check 4: Hardcoded test-year list — WARNING ⚠️

Lines 117-120:
```python
if quick:
    test_years = [2019, 2023]
else:
    test_years = [2019, 2020, 2021, 2022, 2023, 2024]
```

The test years are **hardcoded** — they do not dynamically derive from the data. Two issues:
1. If the dataset contains years outside 2019–2024 (e.g., 2025 data in future), it is silently ignored.
2. The normal mode skips years 2014–2018 (only tests from 2019 onwards). This does not affect production since thesis data starts in 2013, but means early windows are never evaluated.

#### Check 5: Failed rows when year has <10 samples — WARNING ⚠️

The task specification mentions `<10 samples` as the skip threshold. The actual code (lines 181-199) uses `< 2`:

```python
if len(train_df) < 2 or len(test_df) < 2:
    # ... SKIPPED result appended
```

Statistically, 2 rows is insufficient for meaningful model training or PR-AUC computation (which requires both classes to be present). The threshold should be at minimum 10 (likely higher). With only 2 rows, even if not skipped, the model would fit noise and metrics would be meaningless.

**The skip threshold discrepancy (`<2` vs `<10`) is a bug.** Rows with 2-9 samples in train or test are included silently, producing unreliable metrics without warning.

#### Check 6: Horizon target requires horizon dataset — PASS ✅

Lines 98-103:
```python
if data_path:
    filename = Path(data_path).name
    if target_column.startswith("adopted_in_") and filename != "horizon_modeling_dataset.csv":
        raise ValueError(f"Target '{target_column}' requires 'horizon_modeling_dataset.csv', got '{filename}'.")
    if target_column in ["classification_target", "regression_target_days"] and filename != "modeling_dataset.csv":
        raise ValueError(f"Target '{target_column}' requires 'modeling_dataset.csv', got '{filename}'.")
```

**Guard is present and raises a `ValueError` explicitly.** However, this check only fires when `data_path` is not `None`. If callers pass the DataFrame directly without `data_path`, the check is bypassed entirely. In production (`evaluate_backtesting.py` line 79), `data_path=str(data_path)` is always passed, so the check is active in normal usage. But unit tests that call `run_yearly_backtesting` without `data_path` have no protection against using the wrong dataset.

---

## Task 13: PR-AUC and Report Generation

### File: `src/aac_adoption/models/evaluate.py`

#### Check 7: PR-AUC correctly computed — PASS ✅

`classification_metrics()` (lines 43-72) computes PR-AUC via sklearn's `average_precision_score`:

```python
metrics["pr_auc"] = average_precision_score(y_true, y_score)
```

This is the standard area under the precision-recall curve (AP score), computed correctly from probability scores. `average_precision_score` uses the trapezoidal approximation and is appropriate for imbalanced datasets. **Computation is correct.**

### File: `src/aac_adoption/analysis/model_selection.py`

#### Check 8: PR-AUC as primary metric for classification selection — PASS ✅

`_select_classification()` (lines 46-97) uses the sort order `["pr_auc", "brier_score", "expected_calibration_error", "roc_auc"]` with `ascending=[False, True, True, False]` (line 68-69). PR-AUC is the **primary sort key**, with descending order (higher is better). The selection reason template (line 16) explicitly says "highest PR-AUC". **PR-AUC is correctly the primary selection metric.**

### File: `src/aac_adoption/reporting/report.py`

#### Check 9: PR-AUC figure NOT generated — FAIL ❌

`create_report_outputs()` generates the following classification plots (lines 366-402):
- `model_comparison_classification_roc_auc.png` — ROC-AUC
- `model_comparison_classification_f1.png` — F1

**No PR-AUC figure is generated.** Given that:
1. PR-AUC is the primary selection metric in `model_selection.py`
2. The summary text (line 180) reads "Best classification models by 2023 selection PR-AUC"
3. PR-AUC is more informative for imbalanced adoption data

The absence of a `model_comparison_classification_pr_auc.png` figure is inconsistent with the thesis methodology and the report's own summary text.

**Bug:** A PR-AUC figure should be generated alongside or in place of the F1 figure.

#### Check 10: Schema changes — PASS (with guard) ✅

`create_report_outputs()` (lines 345-353) explicitly rejects classification/regression tables that lack `metric_split`:

```python
if not {"metric_split"}.issubset(classification.columns):
    raise ValueError("Report generation must reject model comparison data without split/source metadata.")
```

This guard ensures that stale or schema-changed model comparison files are rejected with an explicit failure. **The guard is correctly implemented.** However, this also means all tests that provide fixture data without `metric_split` will fail (see Bug #3 below).

#### Check 11: 2023 selection vs test rows correctly separated — PASS ✅

`split.py` (lines 81-84) correctly separates periods:
```python
train = subset_df.loc[subset_df["intake_year"].between(2013, 2021)].copy()
calibration = subset_df.loc[subset_df["intake_year"] == 2022].copy()
selection = subset_df.loc[subset_df["intake_year"] == 2023].copy()
test = subset_df.loc[subset_df["intake_year"].between(2024, 2025)].copy()
```

In `_best_rows()` (report.py lines 137-152), `metric_split == "selection"` rows are filtered before ranking. The `_select_classification()` function in model_selection.py applies `metric_split == "selection"` filter via `_filter_candidates()`. 2022 (calibration) and 2023 (selection) are kept separate and the selection happens on 2023 rows only. **Correct implementation.**

---

## Test Run Results

### `tests/test_backtesting.py`
```
1 passed in 15.42s
```
**All tests pass.** This test is a minimal schema check only.

### `tests/test_report_outputs.py`
```
FAILED tests/test_report_outputs.py::test_create_report_outputs_writes_summary_and_figures
1 failed in 3.09s
```

**Root cause:** The test fixture at line 12-18 creates `model_comparison_classification.csv` without a `metric_split` column. The `create_report_outputs()` function (report.py line 347-348) raises `ValueError` for exactly this case. The test and production code are in direct conflict.

### `tests/test_yearly_backtesting.py`
At minimum **3 failures** observed (test was still running at report time). Based on code analysis:

| Test | Expected Result | Analysis |
|------|----------------|----------|
| `test_yearly_backtesting_output_schema` | PASS | Schema check only, quick mode |
| `test_get_test_years` | PASS | Returns years > 2013 from 2019–2024 fixture |
| `test_get_train_years` | PASS | Simple math check |
| `test_format_train_period` | PASS | String format check |
| `test_detect_categorical_features` | PASS | Column type check |
| `test_yearly_backtesting_catboost_classifier_metrics` | PASS | Metric presence check |
| `test_yearly_backtesting_histgradientboosting_classifier_metrics` | PASS | Metric presence check |
| `test_yearly_backtesting_catboost_regressor_metrics` | PASS | Metric presence check |
| `test_yearly_backtesting_histgradientboosting_regressor_metrics` | PASS | Metric presence check |
| `test_yearly_backtesting_bootstrap_confidence_intervals` | PASS | CI column presence |
| `test_yearly_backtesting_animal_subsets` | PASS | Subset name check |
| `test_yearly_backtesting_horizon_targets` | **FAIL** | See Bug #1 below |
| `test_yearly_backtesting_empty_splits_skipped` | **FAIL** | See Bug #2 below |
| `test_yearly_backtesting_multiple_targets` | PASS | Model type check |
| `test_yearly_backtesting_output_csv` | **FAIL/WARN** | See Bug #4 below |

---

## Bugs Found

### Bug #1 (Critical) — `test_yearly_backtesting_horizon_targets`: wrong expectation with `quick=True`
**File:** `tests/test_yearly_backtesting.py`, lines 278–288  
**Symptom:** Test asserts `test_years == [2019, 2020, 2021, 2022, 2023, 2024]` but uses `quick=True`.  
**Root cause:** When `quick=True`, `run_yearly_backtesting` only processes `test_years = [2019, 2023]` (yearly_backtesting.py line 118). The test assertion expects all 6 years from the fixture, which only happens in full (non-quick) mode. The test either should set `quick=False`, or the assertion should use `[2019, 2023]`.  
**Additionally:** The train_years assertion at line 287 expects `f"2013-{year-1}"` but for year 2019 with fixture data starting at 2019, the actual train period would be `"2019-2018"` (impossible range) → SKIPPED row. For year 2023 with fixture ending at 2024, the train start equals `max(2019, 2013) = 2019`, so train_period is `"2019-2022"`, not `"2013-2022"`.

### Bug #2 (Critical) — `test_yearly_backtesting_empty_splits_skipped`: incorrect assertion
**File:** `tests/test_yearly_backtesting.py`, lines 311–312  
**Symptom:** Test asserts `len(result) == 0`.  
**Root cause:** The fixture has years [2019, 2020]. Quick mode tries [2019, 2023]. Year 2023 is skipped (not in data). Year 2019 has train_end=2018 < min_year=2019, producing an empty train set → SKIPPED row is appended. This is done for all 3 subsets (combined, dogs, cats). So `len(result) == 3`, not 0. The test assertion `== 0` is incorrect — it should be `> 0` with `result["status"].eq("SKIPPED").all()`.

### Bug #3 (Critical) — `test_report_outputs` fixture missing `metric_split` column
**File:** `tests/test_report_outputs.py`, lines 12–18  
**Symptom:** `create_report_outputs()` raises `ValueError` before any assertions run.  
**Root cause:** The production code at `report.py:347` requires the `metric_split` column as a metadata guard. The test fixture does not include `metric_split`. The test must add `"metric_split": "selection"` to all rows in the classification and regression fixture DataFrames.  
**Secondary:** Even if this were fixed, the test assertion at line 75 — `assert "Best classification models by ROC-AUC" in summary_text` — uses stale text. The current `report.py:180` emits `"Best classification models by 2023 selection PR-AUC:"`. The assertion would still fail.

### Bug #4 (High) — `test_yearly_backtesting_output_csv` CSV round-trip equality
**File:** `tests/test_yearly_backtesting.py`, lines 349–366  
**Symptom:** `pd.testing.assert_frame_equal(result, saved_df)` may fail due to dtype differences after CSV round-trip.  
**Root cause:** Metric columns are coerced to numeric after collection (yearly_backtesting.py lines 466-469). However, CSV round-trip can change integer columns (e.g., `train_start_year`, `test_year`) from int64 to int64 OR cause object→float conversions for nullable columns like `pr_auc_lower/upper` (which only exist for some rows). The `assert_frame_equal` call does not specify `check_dtype=False`, so type mismatches on mixed-nullable columns may cause failure.

### Bug #5 (High) — No PR-AUC figure generated in `create_report_outputs`
**File:** `src/aac_adoption/reporting/report.py`, lines 366–403  
**Symptom:** The report summary text describes "Best classification models by 2023 selection PR-AUC" but no PR-AUC figure is generated.  
**Root cause:** `_save_grouped_metric_plot` is called only for `roc_auc` and `f1`, not `pr_auc`. A figure `model_comparison_classification_pr_auc.png` should be added.

### Bug #6 (Medium) — Minimum sample threshold is `<2` not `<10`
**File:** `src/aac_adoption/models/yearly_backtesting.py`, line 181  
**Symptom:** Windows with 2–9 rows in train or test are not skipped; they proceed to model training.  
**Root cause:** The threshold is `< 2` not `< 10`. Models trained on 2-9 samples produce statistically meaningless metrics and may fail silently (e.g., PR-AUC undefined if only one class present). The spec states <10 samples should be skipped.

### Bug #7 (Medium) — Train period string is `"START-END"` where START > END for first window
**File:** `src/aac_adoption/models/yearly_backtesting.py`, lines 129–132  
**Symptom:** When `train_start_year > train_end_year`, `format_train_period()` produces a string like `"2019-2018"`.  
**Root cause:** The empty-train guard at line 160 catches this, but the SKIPPED row is still written with the incoherent period string. A guard should produce `"N/A"` or `"<insufficient-history>"` for the period string in this case.

---

## Recommendations

1. **Fix `test_report_outputs.py` fixture** (Bug #3): Add `"metric_split": "selection"` to all rows in both CSV fixtures. Update the text assertion from `"Best classification models by ROC-AUC"` to `"Best classification models by 2023 selection PR-AUC"`.

2. **Fix `test_yearly_backtesting_horizon_targets`** (Bug #1): Either set `quick=False` and update expected `train_years` to use actual data start year, OR change expected test_years to `[2019, 2023]` for `quick=True`.

3. **Fix `test_yearly_backtesting_empty_splits_skipped`** (Bug #2): Change `assert len(result) == 0` to:
   ```python
   assert len(result) > 0
   assert result["status"].eq("SKIPPED").all()
   ```

4. **Add PR-AUC figure to `create_report_outputs`** (Bug #5): After line 372, add:
   ```python
   _save_grouped_metric_plot(
       classification, "pr_auc",
       figures / "model_comparison_classification_pr_auc.png",
       "Classification PR-AUC by model and subset", "PR-AUC",
   )
   ```
   Also add a test assertion for this figure in `test_report_outputs.py`.

5. **Raise minimum sample threshold to 10** (Bug #6): Change line 181 from:
   ```python
   if len(train_df) < 2 or len(test_df) < 2:
   ```
   to:
   ```python
   if len(train_df) < 10 or len(test_df) < 10:
   ```

6. **Fix SKIPPED period string for impossible windows** (Bug #7): Before line 132:
   ```python
   if train_start > train_end:
       train_period = "N/A (insufficient history)"
   else:
       train_period = format_train_period(train_start, train_end)
   ```

7. **Add `data_path` enforcement to tests**: Unit tests for horizon targets should include a `data_path` parameter pointing to a horizon dataset path, or the guard at lines 98-103 should apply to in-memory calls as well.

8. **CSV round-trip test**: Add `check_dtype=False` to `pd.testing.assert_frame_equal` in `test_yearly_backtesting_output_csv`, or explicitly document which columns are expected to change dtype.

---

## Code Evidence Cross-Reference

| Location | Line(s) | Issue |
|----------|---------|-------|
| `yearly_backtesting.py` | 118 | Hardcoded quick test years [2019, 2023] |
| `yearly_backtesting.py` | 120 | Hardcoded full test years [2019–2024], misses 2014–2018 |
| `yearly_backtesting.py` | 129-130 | Incoherent period when train_start > train_end |
| `yearly_backtesting.py` | 181 | Skip threshold <2, should be <10 |
| `yearly_backtesting.py` | 98-103 | Horizon dataset guard bypassed without data_path |
| `report.py` | 179 | Uses PR-AUC for selection, correct |
| `report.py` | 180 | Summary text: "2023 selection PR-AUC" |
| `report.py` | 347-348 | Correctly rejects missing `metric_split` |
| `report.py` | 366-402 | No PR-AUC figure generated |
| `model_selection.py` | 65-68 | Correctly sorts by PR-AUC first |
| `evaluate.py` | 58 | Correct: `average_precision_score` for PR-AUC |
| `test_report_outputs.py` | 12-18 | Fixture missing `metric_split` → always fails |
| `test_report_outputs.py` | 75 | Stale assertion text "ROC-AUC" vs actual "PR-AUC" |
| `test_yearly_backtesting.py` | 278 | quick=True but expects 6 years (not 2) |
| `test_yearly_backtesting.py` | 282 | expected_years inconsistent with quick mode |
| `test_yearly_backtesting.py` | 287 | train_years `"2013-{year-1}"` wrong for fixture start |
| `test_yearly_backtesting.py` | 312 | `len(result) == 0` incorrect; SKIPPED rows exist |
