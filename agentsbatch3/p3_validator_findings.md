# P3 Risk Analysis: Weak Spy Test in test_hyperparam_tuning.py

## Issue Summary

**Location**: `tests/test_hyperparam_tuning.py:164-180`

**Problem**: Test uses a spy pattern to verify `CatBoostRegressor.fit()` calls, but validation is insufficient to ensure all cross-validation folds execute. Test can pass even if the objective function fails after completing only 1 fold instead of the expected 10 (2 trials × 5 folds).

---

## Code Analysis

### Current Test Assertions (lines 164-180)

```python
with patch("catboost.CatBoostRegressor.fit") as mock_fit:
    best_params, studies = tune_models(df, n_trials=2)
    
    assert mock_fit.call_count >= 1  # LINE 167 - WEAK ASSERTION
    
    for call in mock_fit.call_args_list:
        X_tr = call[0][0]
        y_tr = call[0][1]
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert X_tr.index.equals(y_tr.index)
    
    actual_y_values = [call[0][1] for call in mock_fit.call_args_list]
    
    for y in actual_y_values:
        original_y_tr = y_reg_original[y.index]
        assert np.allclose(y, np.log1p(original_y_tr))
```

### How `tune_models` Executes

**File**: `src/aac_adoption/models/tune.py:112-148`

```python
def catboost_reg_objective(trial: optuna.Trial) -> float:
    # ...
    try:
        scores = []
        for train_idx, val_idx in cv.split(cat_X_reg):  # cv = TimeSeriesSplit(n_splits=5)
            X_tr, y_tr_reg = cat_X_reg.iloc[train_idx], y_reg.iloc[train_idx]
            # ...
            model = CatBoostRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr_reg), ...)  # fit() called per fold
            # ...
        return np.mean(scores)
    except Exception:
        return 1e9  # Objective dies, returns high value

study_cat_reg.optimize(catboost_reg_objective, n_trials=n_trials)
```

**Execution Flow**:
- `n_trials=2` → 2 Optuna trials
- `n_splits=5` (TimeSeriesSplit) → 5 CV folds per trial
- **Expected total**: 2 × 5 = 10 `fit()` calls
- **Actual minimum**: 1 `fit()` call (if objective dies early)

---

## Risk Assessment

### 1. Is current test sufficient to validate all folds execute?

**NO**

The test only verifies `mock_fit.call_count >= 1`, which:
- Passes with 1 call (incomplete execution)
- Passes with 10 calls (complete execution)
- Does not distinguish between partial and complete fold execution

### 2. Should test assert exactly 10 calls (2 trials × 5 folds)?

**YES** - for comprehensive validation.

**Rationale**:
- Test name: `test_tune_models_catboost_regression_fit_spy` implies spy pattern validation
- The test validates y_tr log-transform, which requires checking ALL folds
- If objective dies after 1 fold, y_tr validation still passes (1 log-transform call looks valid)

**Exception**: Asserting exact count adds coupling to implementation details (n_trials, n_splits). Consider adding a comment explaining expected count derivation.

### 3. Should test validate fold order or time-series split integrity?

**YES** for index ordering, **NO** for explicit fold order validation.

**Recommended validation**:
```python
# Verify all indices are unique (no duplicate folds)
all_train_indices = []
for call in mock_fit.call_args_list:
    train_idx = call[0][0].index
    all_train_indices.extend(train_idx.tolist())

assert len(all_train_indices) == len(set(all_train_indices)), "Training indices must be unique across folds"
```

**Why not explicit order validation**:
- TimeSeriesSplit guarantees chronological order, but test doesn't need to verify Optuna's trial execution order
- Individual fold integrity (index matching, log-transform) is sufficient

### 4. What's the minimal valid assertion for this spy pattern?

**Minimal** (detects zero calls but not partial execution):
```python
assert mock_fit.call_count >= 1
```

**Recommended minimal** (detects partial execution):
```python
# Expected: n_trials × n_splits = 2 × 5 = 10
assert mock_fit.call_count == 10
```

**Optimal** (detects partial execution + validates fold quality):
```python
# Count validation
assert mock_fit.call_count == 10, f"Expected 10 fit() calls, got {mock_fit.call_count}"

# Individual call validation (already present, keep)
for call in mock_fit.call_args_list:
    X_tr = call[0][0]
    y_tr = call[0][1]
    assert isinstance(X_tr, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)
    assert X_tr.index.equals(y_tr.index)

# Log-transform validation (already present, keep)
for y in actual_y_values:
    original_y_tr = y_reg_original[y.index]
    assert np.allclose(y, np.log1p(original_y_tr))
```

---

## Severity: P3 (Moderate)

**Impact**:
- **False positive risk**: Test passes even when objective function crashes after 1 fold
- **Production risk**: If `catboost_reg_objective` fails mid-execution, production code would return incomplete results, but test wouldn't catch it

**Why not P2**:
- Test still validates correctness of each fold that DOES execute
- Test name suggests "fit spy" → validation is about verifying fit() calls happen, not about objective completion
- Optuna catches exceptions and returns fallback values (0.0 or 1e9), so production behavior is graceful degradation

---

## Recommended Fixes

### Fix Option A: Assert exact call count (simplest)

```python
# Expected: n_trials × n_splits = 2 × 5 = 10
expected_calls = 2 * 5  # n_trials × TimeSeriesSplit(n_splits=5)
assert mock_fit.call_count == expected_calls, f"Expected {expected_calls} fit() calls, got {mock_fit.call_count}"
```

**Pros**: Simple, detects partial execution immediately  
**Cons**: Hardcodes implementation details (coupling to n_trials=2, n_splits=5)

### Fix Option B: Assert expected range with clear documentation

```python
# CV configuration from tune.py:57: cv = TimeSeriesSplit(n_splits=5)
# Expected calls = n_trials (2) × n_splits (5) = 10
expected_calls = 10
assert mock_fit.call_count == expected_calls, \
    f"Expected {expected_calls} fit() calls (2 trials × 5 folds), got {mock_fit.call_count}"
```

**Pros**: Clear documentation, explicit expectation  
**Cons**: Still couples to specific values

### Fix Option C: Validate all training indices are unique

```python
# Collect all training indices to verify completeness
all_train_indices = []
for call in mock_fit.call_args_list:
    train_idx = call[0][0].index
    all_train_indices.extend(train_idx.tolist())

# Verify: 10 folds = 10 * (len(X_reg) / 5) unique indices
# With 200 samples and 5 splits: each fold uses ~160 samples, total = 1600 indices
# 1600 indices with 5 unique splits means 1600 / 5 * 5 = 1600, but must check uniqueness
assert len(all_train_indices) == 10 * len(train_df_reg) / 5, \
    f"Incomplete CV execution: expected {10 * len(train_df_reg) / 5} total samples across folds, got {len(all_train_indices)}"
```

**Pros**: Validates actual data usage, not just function calls  
**Cons**: More complex, still couples to data size

---

## Conclusion

**Critical Issue**: Test at line 167 (`assert mock_fit.call_count >= 1`) is insufficient to detect partial CV execution.

**Fix Required**: Change to exact count assertion:
```python
assert mock_fit.call_count == 10  # 2 trials × 5 folds
```

**Rationale**: The test already validates log-transform on ALL collected y_tr values (lines 176-180). If only 1 fold executes, those assertions still pass because a single log-transformed y_tr is valid. The test must enforce that ALL 10 expected fit() calls occur.
