# Weak Test Analysis - Batch 3

## Date: 2026-06-07

## Weak Test 1: test_hyperparam_tuning.py:L134

### 1. Why Weak
The test `test_tune_models_regression_feature_alignment` (L134-153) directly reimplements the split logic from `make_time_split` (L137-147) instead of verifying the function's behavior. This means:
- The test duplicates logic from the implementation, making it fragile to changes in `make_time_split` or internal split behavior
- It validates index alignment at the DataFrame level but doesn't verify that the **actual regression frame used by `CatBoostRegressor.fit`** receives correctly aligned data
- A broken `tune_models` function that incorrectly aligned features/targets internally could still pass because the test only checks the intermediate splitting logic, not the actual fit call

**Key issue**: The test validates the *splitting strategy* but not the *regression frame preparation* that `CatBoostRegressor.fit` actually consumes. At line 134-153, the test recreates what `tune_models` does at lines 44-50 in tune.py, including `prepare_catboost_frame` calls.

### 2. Required Spy/Mock
Spy/mock the `CatBoostRegressor.fit` method to verify:
- **What data is passed**: Verify `X_tr` and `y_tr` passed to `fit` are correctly aligned after `prepare_catboost_frame` transformation
- **Index alignment in actual fit**: Assert that `X_tr.index.equals(y_tr.index)` for **each CV fold** during the optimization loop (lines 128-144 in tune.py)
- **Call count**: Verify `fit` is called exactly 5 times (matching TimeSeriesSplit(n_splits=5))
- **Log transformation verification**: Verify that `y_tr` passed to fit is `np.log1p(y_tr_reg)` (line 134), not raw values

Implementation requirement:
```python
from unittest.mock import patch
with patch('catboost.CatBoostRegressor.fit', autospec=True) as mock_fit:
    best_params, studies = tune_models(df, n_trials=2)
    # Verify mock was called with aligned data for each trial
    assert mock_fit.call_count == 2 * 5  # 2 trials × 5 CV splits
    for call in mock_fit.call_args_list:
        X_tr, y_tr = call[0][0], call[0][1]
        assert X_tr.index.equals(y_tr.index)
        # Verify log transformation
        assert np.allclose(y_tr, np.log1p(original_y_tr))
```

### 3. Schema/Content Assertions
The test should validate that `prepare_catboost_frame` (train_advanced.py) output:
- **Schema**: Contains all categorical features converted to appropriate types (object/dtype expected by CatBoost)
- **Content**: No NaN values in categorical columns (catboost requires explicit handling)
- **Cross-validation integrity**: For each of 5 CV folds, the training fold features and targets share identical indices
- **Log transformation**: Regression target passed to `CatBoostRegressor.fit` is log-transformed (`np.log1p`)
- **CatBoost frame structure**: Output from `prepare_catboost_frame` matches expected feature set from `model_feature_columns`

Target assertions:
```python
# Verify log transformation in actual fit calls
actual_y_values = [call[0][1] for call in mock_fit.call_args_list]
assert all(np.allclose(y, np.log1p(original_y[y.index])) for y, original_y in zip(...))

# Verify categorical columns are preserved correctly
cat_cols = [col for col in X_tr.columns if col in categorical_features]
assert all(X_tr[col].dtype == 'object' for col in cat_cols)
```

---

## Weak Test 2: test_train_boosting_outputs.py:L73

### 1. Why Weak
The test at line 73 (`assert (tables_dir / "permutation_importance_regression.csv").exists()`) only verifies file existence, not the **content** of the permutation importance table. The critical requirement (from `_permutation_table` function at L128-160 in train_boosting.py) is that the `evaluation_period` column equals `"validation"` when validation data exists.

**Why this matters**: The `_permutation_table` function determines which split to use at lines 138-139:
```python
sample = split.validation if not split.validation.empty else split.test
importance_split = "validation" if not split.validation.empty else "test"
```

The `evaluation_period` column (line 157) is set to `importance_split`, which should be `"validation"` when validation period data exists. The test should verify this semantic correctness, not just file creation.

### 2. Required Assertions
Verify the permutation importance table schema and content:
- **Column presence**: `evaluation_period` column exists
- **Value verification**: `evaluation_period` == `"validation"` for regression permutations (when validation data exists)
- **Data source verification**: All rows use the same evaluation split (consistent `evaluation_period` value)

### 3. Schema/Content Assertions
The test should validate:
- **Schema assertions**:
  - Required columns: `feature`, `importance_mean`, `importance_std`, `importance_split`, `evaluation_period`, plus metadata columns
  - `importance_split` and `evaluation_period` columns are present
  - No NaN values in `evaluation_period` column
- **Content assertions**:
  - `evaluation_period` == `"validation"` (when validation split exists, which it does via `make_time_split` with years 2018-2025)
  - `importance_split` == `evaluation_period` (both should be "validation")
  - All regression permutation rows share same `evaluation_period` value
  - Feature names match expected feature set from `model_feature_columns`

Schema/content assertions:
```python
perm_reg = pd.read_csv(tables_dir / "permutation_importance_regression.csv")
assert "evaluation_period" in perm_reg.columns
assert "importance_split" in perm_reg.columns
assert set(perm_reg["evaluation_period"].unique()) == {"validation"}
assert set(perm_reg["importance_split"].unique()) == {"validation"}
assert perm_reg["evaluation_period"].equals(perm_reg["importance_split"])
assert len(perm_reg) == len(feature_columns)  # One row per feature
assert not perm_reg["evaluation_period"].isna().any()
```
