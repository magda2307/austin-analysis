# Validator Findings: P2 Risk in Permutation Importance Validation Fallback

## Overview
Analyzed src/aac_adoption/models/train_boosting.py for validation-only rule violations in permutation importance calculation.

---

## 1. EXACT CODE CAUSING ISSUE

### Permutation Importance Fallback (Lines 138-139)
```python
sample = split.validation if not split.validation.empty else split.test
importance_split = "validation" if not split.validation.empty else "test"
```

### Full Context (_permutation_table function, Lines 128-160)
Lines 138-157:
```python
    sample = split.validation if not split.validation.empty else split.test
    importance_split = "validation" if not split.validation.empty else "test"
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=RANDOM_STATE)
    result = permutation_importance(
        pipeline,
        sample[feature_columns],
        sample[target_column],
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_split": importance_split,
            "evaluation_period": importance_split,
            **metadata,
        }
    ).sort_values("importance_mean", ascending=False)
```

---

## 2. VALIDATION-ONLY RULE VIOLATION ANALYSIS

### 2.1 Violation Status: YES

**The validation-only rule IS violated when split.validation is empty.**

When split.validation.empty is True, the code:
1. Falls back to using split.test for permutation importance calculation
2. Sets importance_split = "test" (correctly documenting what was used)
3. The evaluation_period column copies from importance_split (line 157)

### 2.2 Critical Issue: evaluation_period Misrepresentation

**Line 157 sets evaluation_period = importance_split, which means:**
- When validation exists: evaluation_period = "validation" 
- When validation is empty: evaluation_period = "test"  (P2 risk)

**This creates a thesis safety problem:** If validation data is ever empty (due to time-based splitting issues, small datasets, or edge cases), permutation importance will be calculated on TEST data but may be documented incorrectly in analysis pipelines that assume validation was used.

---

## 3. VALIDATION-ONLY RULE VIOLATION SCENARIOS

### Scenario 1: Empty Validation Split (Time-Based Split Edge Case)
**Risk:** validation-only rule violated, test data used

**Code Path:**
1. make_time_split() creates time-based splits (split.py lines 47-116)
2. Validation period contains 0 samples (empty DataFrame) at line 66
3. Code falls back to test data for permutation importance (train_boosting.py lines 138-139)
4. importance_split = "test" is correctly set
5. BUT if downstream analysis assumes validation was used, conclusions are invalid

### Scenario 2: Test Data Used When Validation Expected
**Risk:** Methodology confusion, invalid thesis conclusions

**Current Behavior:**
| Condition | Permutation Data Used | importance_split | evaluation_period |
|-----------|----------------------|------------------|-------------------|
| split.validation.empty = False | validation | "validation" | "validation" |
| split.validation.empty = True | test | "test" | "test" |

**The problem:** The evaluation_period column should indicate what was *intended* to be used for validation-only metrics, not what was actually used as a fallback. If validation was never available, the metric cannot be classified as "validation-only" by definition.

---

## 4. THESIS SAFETY CONCERNS

### 4.1 Primary Concern: Validation Metrics on Test Data

**Permutation importance measures feature stability, which is:**
1. A property of the model learned patterns
2. Independent of data split usage (theoretically any set works)
3. BUT thesis methodology requires validation-only metrics for fair comparison

**Why this matters for thesis validity:**
1. Permutation importance on test data leaks test information into feature importance analysis
2. If combined with test metrics, creates circular analysis (test - importance - test)
3. Undermines clean separation between validation (model selection) and test (final evaluation)

### 4.2 Secondary Concern: evaluation_period Column Semantics

**Line 157: "evaluation_period": importance_split**
- Should represent the *intended* evaluation stage (validation vs test)
- Currently represents what was *actually* used
- This creates ambiguity in downstream analysis

**Correct interpretation for thesis:**
- evaluation_period = "validation" - this is a validation-only metric
- evaluation_period = "test" - this is a test-set metric (final evaluation only)

When validation is empty, the metric cannot be classified as "validation-only" by definition.

---

## 5. EDGE CASES

| Edge Case | Current Handling | Thesis-Safety Risk |
|-----------|------------------|-------------------|
| **Validation empty (0 samples)** | Falls back to test | HIGH - validation-only rule violated |
| **Test empty (0 samples)** | Falls back to validation | MEDIUM - test metrics undefined |
| **Both validation and test empty** | Uses empty sample, permutation_importance may fail | HIGH - undefined behavior |
| **Validation smaller than max_rows** | Uses all validation | LOW - correct behavior |
| **Test smaller than max_rows** | Uses all test | LOW - correct behavior (as fallback) |

**Most Critical Edge Case:** Validation empty with non-empty test. This triggers the P2 violation where validation-only metrics are calculated on test data.

### split.py Analysis: When Can Validation Be Empty?

Looking at make_time_split (split.py lines 64-86):
- Time-based split uses years 2022-2023 for validation (line 66)
- If no data exists for these years, validation will be empty (0 rows)
- Fallback logic (lines 65-86) only returns if `not train.empty and not test.empty`
- **Note:** There is NO check for validation.empty before returning!

Looking at random split fallback (split.py lines 88-116):
- Uses train_test_split to create random splits
- Does not explicitly check if validation is empty
- Could produce empty validation with very small datasets

---

## 6. THESIS-SAFE FIX OPTIONS

### Option A: ✅ RAISE EXCEPTION (RECOMMENDED)
```python
def _permutation_table(
    pipeline,
    split: DatasetSplit,
    feature_columns: list[str],
    target_column: str,
    metadata: dict,
    scoring: str,
    repeats: int,
    max_rows: int,
) -> pd.DataFrame:
    # Thesis-safety: must use validation for validation-only metrics
    if split.validation.empty:
        raise ValueError(
            f"Permutation importance requires validation data for {metadata.get(animal_subset, unknown)}. "
            f"Validation split is empty. Check time-based split configuration or dataset size."
        )
    
    sample = split.validation
    importance_split = "validation"
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=RANDOM_STATE)
    result = permutation_importance(
        pipeline,
        sample[feature_columns],
        sample[target_column],
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_split": importance_split,
            "evaluation_period": importance_split,
            **metadata,
        }
    ).sort_values("importance_mean", ascending=False)
```

**Rationale:**
1. Permutation importance for feature analysis should use validation data (not test)
2. Validation-only rule is fundamental to thesis methodology
3. Empty validation indicates data preparation issue that should not be silently ignored
4. Clear error forces proper data engineering before analysis

### Option B: Skip permutation importance when validation empty (with warning)
**Pros:** Graceful degradation  
**Cons:** May mislead user about feature importance availability; inconsistent behavior

### Option C: Document "N/A" when validation unavailable
**Pros:** Transparent about missing metrics  
**Cons:** Still allows test data infiltration; harder to filter out invalid results

**Final Recommendation:** Option A (raise exception)

---

## 7. CODE PATH ANALYSIS

### Training Flow (Lines 206-217 in train_boosting.py)
```python
append_table(
    _permutation_table(
        pipeline,
        split,
        feature_columns,
        "classification_target",
        metadata,
        scoring="roc_auc",
        repeats=permutation_repeats,
        max_rows=permutation_max_rows,
    ),
    tables_dir / "permutation_importance_classification.csv",
)
```

**The issue:** This callpath is executed for every animal subset (dogs, cats, combined). If any subset has empty validation, the current code silently switches to test data.

### Split Generation (Line 184)
```python
split = make_time_split(df, "classification_target", animal_subset=subset)
```

**Potential issue source:** make_time_split() may create empty validation splits under certain conditions:
- Time-based cutoff results in no samples in validation period (years 2022-2023)
- Dataset too small for requested split configuration
- Date filtering removes all validation samples

---

## 8. RECOMMENDATIONS

### Immediate Fix: Raise Exception on Empty Validation
```python
# Add at start of _permutation_table (after line 137)
if split.validation.empty:
    raise ValueError(
        f"Permutation importance requires validation data for {split.animal_subset}. "
        f"Validation split is empty. This violates validation-only methodology."
    )
```

### Additional Safeguards:
1. Validate split structure before training: Check validation isnt empty before calling _permutation_table
2. Add test coverage for edge cases: Ensure make_time_split handles edge cases properly
3. Document evaluation_period semantics: Clarify that "validation" means validation was used, not intended

### Risk Mitigation Timeline:
| Phase | Action |
|-------|--------|
| Short-term | Add exception raise for empty validation |
| Medium-term | Add validation split integrity checks |
| Long-term | Document and enforce validation-only rule across all metrics |

---

## SUMMARY

| Item | Status |
|------|--------|
| **P2 Severity** | MEDIUM |
| **Thesis Safety** | VIOLATION at line 138-139 |
| **Risk** | Validation-only rule violated when validation is empty - test data used for feature importance |
| **Fix Required** | YES - raise exception when validation.empty |
| **evaluation_period Issue** | YES - reflects actual usage, not intended stage |
| **Edge Cases** | VALIDATION EMPTY - test fallback (P2 violation) |

---

**Recommendation:** Replace validation→test fallback logic with exception raising to maintain thesis methodological integrity. Validation-only metrics must use validation data; if validation is unavailable, the analysis should fail fast rather than silently use test data.
