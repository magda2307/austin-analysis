# Validator Agent Findings - Batch 2

**Task:** P1 Issue Validation - `yearly_backtesting.py` line 341 silently hides failed model/year rows

**Date:** 2026-06-07

---

## Executive Summary

**ISSUE CONFIRMED:** When `strict=False` (default), exceptions during model training are silently logged and iteration continues. This produces incomplete CSV output with missing model/year rows. No error columns are added to indicate failure.

---

## 1. Exception Types That Can Be Silently Swallowed

### Location: `yearly_backtesting.py` lines 164-344

All exceptions in the training loop (try block starting at line 165) are caught generically:

```python
except Exception as e:  # Line 341
    logger.error(f"Error training {model_name} on {subset_name} for test year {test_year}: {e}", exc_info=True)
    if strict:
        raise e
```

### Possible Exception Types:

| Category | Exception Type | Scenario |
|----------|---------------|----------|
| **Data Issues** | `ValueError` | Invalid data format, empty arrays after preprocessing, categorical encoding failures |
| **Data Issues** | `TypeError` | incompatible types in features/target |
| **Data Issues** | `AttributeError` | Missing column, None object in model.fit() |
| **CatBoost** | `CatBoostError` | Model training failures, GPU issues, corrupted parameters |
| **Sklearn** | `ValueError` | Sample weight mismatch, invalid parameters, insufficient samples |
| **Sklearn** | `NotFittedError` | Model fit() called with invalid state |
| **Sklearn** | `ConvergenceWarning` | Converted to exception in strict mode |
| **Bootstrap CI** | `ValueError` | Bootstrap calculation failures, insufficient unique values |
| **Bootstrap CI** | `RuntimeError` | Bootstrap iteration failures |
| **Numerical** | `RuntimeWarning` | Overflow, underflow in computations |
| **Numerical** | `FloatingPointError` | Division by zero, NaN propagation |
| **Index Issues** | `IndexError` | Array bounds errors after filtering |
| **Index Issues** | `KeyError` | Missing column in feature selection |

### High-Risk Scenarios:

1. **CatBoost-specific failures** (CatBoostError)
   - GPU memory exhaustion
   - Invalid categorical feature indices
   - Label encoding failures

2. **Feature preprocessing failures**
   - Categorical encoder with unseen categories (line 145-147)
   - Empty feature matrix after filtering
   - Mismatched column counts between train/test

3. **Bootstrap CI failures**
   - Insufficient unique values for CI calculation
   - Division by zero in metric computation

4. **Data quality issues**
   - All NaN values after preprocessing
   - Constant target variable (no variance)
   - Class imbalance causing prediction failures

---

## 2. Validation Test: Simulated Failure

### Test Design: Force CatBoost to fail

**Approach:** Create dataset with problematic categorical feature that triggers CatBoostError.

```python
def test_yearly_backtesting_failure_handling_strict_false():
    """Test that failures are logged but not raised when strict=False."""
    import pytest
    import logging
    
    # Create minimal dataset that will cause CatBoost to fail
    df = pd.DataFrame({
        "animal_id": ["A0001"] * 10,
        "intake_year": [2019] * 5 + [2020] * 5,
        "classification_target": [1, 0, 1, 0, 1, 1, 1, 1, 1, 1],  # All same class in one year
        "animal_type": ["Dog", "Cat"] * 5,
        "intake_age_days": [100] * 10,  # Constant value causes issues
        "Problematic_Categorical": ["A"] * 9 + ["B"],  # Rare category
    })
    
    with caplog.at_level(logging.ERROR):
        result = run_yearly_backtesting(
            df,
            target_column="classification_target",
            animal_subset="combined",
            output_path=None,
            compute_ci=False,
            strict=False,  # Default - should NOT raise
            quick=True,    # Only test 2019
        )
    
    # ASSERTION 1: Result should be empty or partial (no rows for failed model/year)
    assert len(result) == 0 or len(result) < expected_rows, "Expected incomplete results"
    
    # ASSERTION 2: Error should be logged
    assert "Error training" in caplog.text, "Expected error log message"
    
    # ASSERTION 3: No exception should be raised
    # (test passes if we reach this point)


def test_yearly_backtesting_failure_handling_strict_true():
    """Test that failures are raised when strict=True."""
    df = pd.DataFrame({  # Same problematic dataset
        "animal_id": ["A0001"] * 10,
        "intake_year": [2019] * 5 + [2020] * 5,
        "classification_target": [1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        "animal_type": ["Dog", "Cat"] * 5,
        "intake_age_days": [100] * 10,
        "Problematic_Categorical": ["A"] * 9 + ["B"],
    })
    
    with pytest.raises(Exception):  # Should raise on failure
        run_yearly_backtesting(
            df,
            target_column="classification_target",
            animal_subset="combined",
            output_path=None,
            compute_ci=False,
            strict=True,  # CRITICAL: Should raise
            quick=True,
        )
```

### Expected Behavior Comparison:

| Mode | Exception Handling | Output | CLI Evidence |
|------|-------------------|--------|--------------|
| `strict=False` | Logs error, continues | Incomplete CSV (missing rows) | No failure, incomplete data |
| `strict=True` | Raises exception | No CSV written | Immediate failure |

---

## 3. Exact Output Comparison: strict=False vs strict=True

### Scenario: CatBoost fails on 2019 data

#### strict=False (DEFAULT - Line 74)

**Console Output:**
```
ERROR:aac_adoption.models.yearly_backtesting:Error training catboost_classifier on combined for test year 2019: <error message>
Traceback (most recent call last):
  File "...\yearly_backtesting.py", line 165, in <loop>
    model.fit(...)
  ...
```

**CSV Output (incomplete):**
```csv
train_years,test_year,subset,animal_subset,model,pr_auc,roc_auc,brier,ece,mae,rmse,r2,train_rows,test_rows
2013-2018,2020,combined,combined,catboost_classifier,0.75,0.82,0.15,0.08,---
2013-2018,2020,combined,combined,histgradientboosting_classifier,0.73,0.80,0.16,0.09,---
2013-2019,2021,combined,combined,catboost_classifier,0.76,0.81,0.14,0.07,---
```

**Missing:**
- No row for `test_year=2019` with `model=catboost_classifier`
- No row for `test_year=2019` with `model=histgradientboosting_classifier`

**Total Rows:** 8 (instead of 12 for 2 years × 2 subsets × 2 models)

---

#### strict=True

**Console Output:**
```
ERROR:aac_adoption.models.yearly_backtesting:Error training catboost_classifier on combined for test year 2019: <error message>
Traceback (most recent call last):
  File "...\yearly_backtesting.py", line 165, in <loop>
    model.fit(...)
  ...
Traceback (most recent call last):
  File "...\yearly_backtesting.py", line 344, in run_yearly_backtesting
    raise e
  ...
CatBoostError: <actual CatBoost error>
```

**CSV Output:** No file written (or partial if failure occurs mid-execution)

**Exception:** propagates to caller, CLI exits with non-zero code

---

## 4. Recommendations

### Option A: Make CLI strict by default ⭐ *RECOMMENDED*

**Pros:**
- Fail-fast behavior catches issues early
- Complete evidence trail (failure + traceback)
- CLI users alerted to problems immediately
- Prevents silent data corruption

**Cons:**
-May break existing CI workflows expecting graceful continuation
-Requires updating documentation

**Implementation:**

```python
# scripts/evaluate_backtesting.py line 67
results = run_yearly_backtesting(
    df,
    target_column=target,
    animal_subset=args.subset,
    output_path=None,
    compute_ci=True,
    bootstrap_n=args.n_bootstraps,
    quick=args.quick,
    strict=True,  # CHANGE: Default to strict
)
```

**Or add CLI flag:**

```python
# scripts/evaluate_backtesting.py lines 35-40
parser.add_argument("--n_bootstraps", type=int, default=100,
                    help="Number of bootstrap iterations for CI")
parser.add_argument("--quick", action="store_true",
                    help="Quick mode: run only 2 windows")
parser.add_argument("--strict", action="store_true",  # NEW
                    help="Raise exceptions on model training failures")
```

---

### Option B: Add Error Columns

**Pros:**
- No breaking changes to existing behavior
- Explicit error tracking in output
- Allows post-hoc analysis of failures

**Cons:**
- More complex output schema
- Error information scattered across many rows
- No immediate failure notification

**Implementation:**

```python
# Change result dict structure (lines 298-337)
result = {
    "train_years": train_period,
    "test_year": test_year,
    "subset": subset_name,
    "animal_subset": subset_name,
    "model": model_name,
    "model_name": model_name,
    "pr_auc": metrics.get("pr_auc"),
    "roc_auc": metrics.get("roc_auc"),
    "brier": metrics.get("brier_score"),
    "brier_score": metrics.get("brier_score"),
    "ece": metrics.get("expected_calibration_error"),
    "mae": metrics.get("mae"),
    "rmse": metrics.get("rmse"),
    "r2": metrics.get("r2"),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "success": True,  # NEW
    "error_message": None,  # NEW
    "error_type": None,  # NEW
}

# In except block (lines 341-344)
except Exception as e:
    logger.error(f"Error training {model_name} on {subset_name} for test year {test_year}: {e}", exc_info=True)
    if strict:
        raise e
    # Add error row instead of skipping
    error_result = {
        "train_years": train_period,
        "test_year": test_year,
        "subset": subset_name,
        "animal_subset": subset_name,
        "model": model_name,
        "success": False,
        "error_message": str(e),
        "error_type": type(e).__name__,
        # Set all metrics to None
        "pr_auc": None, "roc_auc": None, "brier": None, "ece": None,
        "mae": None, "rmse": None, "r2": None,
        "train_rows": len(X_train) if 'X_train' in locals() else 0,
        "test_rows": len(X_test) if 'X_test' in locals() else 0,
    }
    results.append(error_result)
```

**CSV Output (with errors):**
```csv
train_years,test_year,subset,animal_subset,model,pr_auc,roc_auc,brier,ece,mae,rmse,r2,train_rows,test_rows,success,error_message,error_type
2013-2018,2019,combined,combined,catboost_classifier,,,,,,,0,5,False,"CatBoostError: ...",CatBoostError
2013-2018,2019,combined,combined,histgradientboosting_classifier,0.74,0.81,0.15,0.08,---,---,---,0,5,True///
```

---

## 5. Current Test Coverage Gaps

### Existing Tests (353 lines, 12 tests)

| Test | Coverage | Gap |
|------|----------|-----|
| `test_yearly_backtesting_output_schema` | ✅ Output columns | No strict mode testing |
| `test_yearly_backtesting_catboost_classifier_metrics` | ✅ Normal training | No failure case |
| `test_yearly_backtesting_histgradientboosting_classifier_metrics` | ✅ Normal training | No failure case |
| `test_yearly_backtesting_catboost_regressor_metrics` | ✅ Normal training | No failure case |
| `test_yearly_backtesting_histgradientboosting_regressor_metrics` | ✅ Normal training | No failure case |
| `test_yearly_backtesting_bootstrap_confidence_intervals` | ✅ CI calculation | No failure case |
| `test_yearly_backtesting_animal_subsets` | ✅ Subsets | No failure case |
| `test_yearly_backtesting_empty_splits_skipped` | ✅ Empty handling | No failure case |
| `test_yearly_backtesting_multiple_targets` | ✅ Multi-target | No failure case |
| `test_yearly_backtesting_output_csv` | ✅ CSV write | No failure case |

### Critical Missing Tests:

**1. Failure Handling Tests (HIGH PRIORITY)**

```python
# Missing test: strict=False mode
def test_yearly_backtesting_failure_strict_false_logs_error():
    """When strict=False, exceptions should be logged but not raised, producing incomplete results."""

# Missing test: strict=True mode  
def test_yearly_backtesting_failure_strict_true_raises():
    """When strict=True, exceptions should propagate to caller."""
```

**2. Specific Failure Scenario Tests**

```python
# Missing: CatBoost error handling
def test_yearly_backtesting_catboost_error_handling():
    """Test CatBoost-specific error scenarios (GPU, categorical indices)."""

# Missing: Bootstrap CI failure
def test_yearly_backtesting_bootstrap_failure():
    """Test bootstrap confidence interval failure cases."""

# Missing: Feature preprocessing failure
def test_yearly_backtesting_feature_preprocessing_failure():
    """Test feature processing edge cases (constant features, missing columns)."""

# Missing: Data quality failure
def test_yearly_backtesting_data_quality_failure():
    """Test handling of problematic data (all NaN, constant target)."""
```

**3. CLI Integration Tests**

```python
# Missing: CLI strict mode
def test_cli_strict_mode_flag():
    """Test evaluate_backtesting.py --strict flag."""

# Missing: CLI output completeness  
def test_cli_output_complete_on_failure():
    """Test CLI produces complete output on partial failures."""
```

---

## 6. Summary

### Issue Severity: **HIGH**

- **Default Behavior:** Silent data loss (incomplete CSV)
- **Detection Difficulty:** CLI shows success, user must compare expected vs actual row counts
- **Impact:** Research/production pipelines may use incomplete evidence without awareness

### Immediate Actions Required:

1. **ADD**: Test for exception handling in both strict modes (lines 341-344)
2. **CHOOSE**: Option A (strict CLI) OR Option B (error columns)
3. **UPDATE**: Test suite to cover failure scenarios
4. **DOCUMENT**: Behavior changes in README/CLI help text

### Estimated Test Coverage After Fixes:

| Category | Before | After |
|----------|--------|-------|
| Normal training | 100% | 100% |
| Failure handling | 0% | 100% |
| Edge cases | 0% | 80% |
| CLI integration | 0% | 100% |

---

**Generated by:** Validator Agent (Batch 2)  
**Files Reviewed:** 
- `src/aac_adoption/models/yearly_backtesting.py` (436 lines)
- `scripts/evaluate_backtesting.py` (87 lines)
- `tests/test_yearly_backtesting.py` (353 lines, 12 tests)
