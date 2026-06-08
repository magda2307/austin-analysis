# P2 Fix Review Report: Permutation Importance Scoring

**Date:** 2026-06-07  
**Reviewer:** Kilo (AI assistant)  
**File:** `src/aac_adoption/models/train_boosting.py`  
**Original Issue:** Line 218 used `scoring="roc_auc"` instead of PR-AUC

---

## Executive Summary

The P2 fix correctly addresses the permutation importance scoring inconsistency. After review, **no additional issues** were found. The fix:
- ✅ Aligns permutation importance with PR-AUC training objective
- ✅ Maintains thesis validation-only methodology
- ✅ Uses appropriate scoring for classification and regression
- ✅ Contains clear explanatory comment

---

## Answer to Review Questions

### 1. Is PR-AUC now used consistently for classification permutation importance?

**Status: ✅ YES**

**Evidence:**
- Line 218: `scoring="average_precision"` — correctly set
- `tune.py:102` (CatBoost CV): Uses `average_precision_score`
- `tune.py:181` (HistGradientBoosting CV): Uses `average_precision_score`
- `evaluate.py:118` (final metrics): Uses `average_precision_score`

**Conclusion:** PR-AUC is now consistently used for:
1. Training objective (tune.py)
2. Final evaluation (evaluate.py)
3. Permutation importance (train_boosting.py:218)

### 2. Does `average_precision` scoring match the training objective (tune.py:102)?

**Status: ✅ YES**

**Evidence from tune.py:**
- Line 102: CatBoost classification uses `average_precision_score` for CV scoring
- Line 107: Study direction `maximize` for PR-AUC
- Line 181: HistGradientBoosting classification uses `average_precision_score`
- Line 186: Study direction `maximize` for PR-AUC

**Technical alignment:**
- Permutation importance measures feature stability via score degradation
- Using `average_precision` ensures importance rankings reflect PR-AUC optimization
- This matches the actual metric used for model selection and comparison

### 3. Are there other permutation importance calls that need similar fixes?

**Status: ✅ NO ADDITIONAL FIXES NEEDED**

**Current permutation importance calls in codebase:**

| File | Line | Task | Scoring | Status |
|------|------|------|---------|--------|
| train_boosting.py | 218 | Classification | `average_precision` | ✅ Correct |
| train_boosting.py | 274 | Regression | `neg_mean_absolute_error` | ✅ Correct |

**Regression analysis:**
- Regression uses `neg_mean_absolute_error` for permutation importance
- tune.py:141 (CatBoost regression CV) uses MAE
- tune.py:229 (HistGradientBoosting regression CV) uses MAE
- Scoring aligns perfectly with training objective

**No other files** contain `permutation_importance` calls outside train_boosting.py.

### 4. Is the comment clear about why PR-AUC is used?

**Status: ⚠️ PARTIALLY CLEAR — SUGGEST IMPROVEMENT**

**Current comment (line 218):**
```python
scoring="average_precision",  # PR-AUC matches training objective
```

**Assessment:**
- ✅ Correctly identifies PR-AUC
- ✅ States it matches training objective
- ❌ Does not explain **why** PR-AUC was chosen over ROC-AUC
- ❌ Does not mention class imbalance consideration
- ❌ Does not cite specific tune.py locations

**Suggested enhancement:**
```python
scoring="average_precision",  # PR-AUC matches training objective (tune.py:102, 181).
                               # PR-AUC is preferred for imbalanced classification and
                               # aligns with model selection and final evaluation.
```

### 5. Does this fix maintain thesis methodology (validation-only metrics)?

**Status: ✅ YES**

**Validation-only methodology compliance:**

1. **Split enforcement (line 138-143):**
   ```python
   if split.validation.empty:
       raise ValueError(
           f"Permutation importance requires validation data for {split.animal_subset}. "
           f"Validation split is empty. This violates validation-only methodology. "
           f"Check time-based split parameters (validation_years=2022-2023)."
       )
   ```
   - Permutation importance **only** uses validation split
   - Test data fallback removed
   - Clear error message explains validation-only requirement

2. **Metric consistency:**
   - Training: PR-AUC (tune.py:102, 181)
   - Permutation importance: PR-AUC (train_boosting.py:218)
   - Final evaluation: PR-AUC (evaluate.py:118)

3. **Time-based split alignment:**
   - Train: 2013-2021
   - Validation: 2022-2023 (used for permutation importance)
   - Test: 2024-2025 (never used for permutation importance)

**Thesis safety:** ✅
- No data leakage
- Consistent validation-only metrics
- Methodology documented in error messages

---

## Code Review Details

### Modified File: train_boosting.py

**Classification (line 218):**
```python
append_table(
    _permutation_table(
        pipeline,
        split,
        feature_columns,
        "classification_target",
        metadata,
        scoring="average_precision",  # ✅ FIXED: Was "roc_auc"
        repeats=permutation_repeats,
        max_rows=permutation_max_rows,
    ),
    tables_dir / "permutation_importance_classification.csv",
)
```

**Regression (line 274):**
```python
append_table(
    _permutation_table(
        pipeline,
        split,
        feature_columns,
        "regression_target_days",
        metadata,
        scoring="neg_mean_absolute_error",  # ✅ Already correct
        repeats=permutation_repeats,
        max_rows=permutation_max_rows,
    ),
    tables_dir / "permutation_importance_regression.csv",
)
```

---

## Testing Verification

**Current test status:** 41 passed (20.60s)

No changes to testable behavior:

1. **Function signature unchanged:** `_permutation_table` still accepts `scoring` parameter
2. **Logic unchanged:** Permutation importance calculation logic identical
3. **Output format unchanged:** DataFrame structure identical
4. **Only metric swapped:** ROC-AUC → PR-AUC for classification

**Recommendation:** Add unit test to verify scoring parameter usage:

```python
# tests/test_train_boosting.py (suggested addition)
def test_classification_permutation_uses_pr_auc():
    """Verify permutation importance uses PR-AUC for classification."""
    # ... setup code ...
    result = train_boosting.train_boosting_classification(...)
    df = pd.read_csv(tables_dir / "permutation_importance_classification.csv")
    # Verify permutation importance computed with average_precision (PR-AUC)
    # not roc_auc, by checking metric consistency with tune.py behavior
```

---

## Final Verdict

| Check | Status | Notes |
|-------|--------|-------|
| PR-AUC consistency | ✅ PASS | Matches tune.py training objective |
| Validation-only methodology | ✅ PASS | Only uses validation split |
| Regression scoring | ✅ PASS | Already aligned with MAE objective |
| Comment clarity | ⚠️ ENHANCEMENT | Add more context why PR-AUC chosen |
| No other fixes needed | ✅ PASS | Only 2 permutation calls, both correct |

**Overall assessment:** ✅ **P2 FIX IS COMPLETE AND VALID**

The fix correctly addresses the original issue. Permutation importance now uses PR-AUC consistently with the training objective, maintains the validation-only methodology, and requires no additional fixes. A minor comment enhancement would improve future maintainability.

---

## Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| `src/aac_adoption/models/train_boosting.py` | 138-143, 218, 274 | Permutation importance implementation |
| `src/aac_adoption/models/tune.py` | 102, 107, 181, 186 | PR-AUC training objective evidence |
| `src/aac_adoption/models/evaluate.py` | 118 | PR-AUC in final metrics |
| `agentsbatch3/p2_fix_summary.md` | — | Original fix documentation |
| `agentsbatch3/p2_validator_findings.md` | — | Validator risk assessment |

---

*End of P2 Review Report*
