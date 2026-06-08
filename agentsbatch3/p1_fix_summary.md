# P1 Fix Summary: tune.py Exception Handling

## File Changed
- `src/aac_adoption/models/tune.py`

## Issues Fixed

### 1. Broad Exception Handling (Lines 104, 143, 183, 223)
**Problem:** `except Exception:` catches all errors and returns placeholder values (0.0 or 1e9), making tuning failures appear as valid trials.

**Solution:** 
- Import `CatBoostError` from `catboost`
- Replace broad except with specific exception handling
- Re-raise `CatBoostError` and `ValueError` as `optuna.TrialPruned`
- Let unexpected errors bubble up with clear context

## Changes Made

### Imports
```python
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostError
```

### catboost_clf_objective (Lines 73-106)
- Added docstring explaining error handling strategy
- CatBoostError → `optuna.TrialPruned`
- ValueError → `optuna.TrialPruned`
- Unexpected errors → bubble up

### catboost_reg_objective (Lines 112-148)
- Added docstring explaining error handling strategy
- CatBoostError → `optuna.TrialPruned`
- ValueError → `optuna.TrialPruned`
- Unexpected errors → bubble up

### hist_clf_objective (Lines 151-188)
- Added docstring explaining error handling strategy
- Added NOTE about why broad except is problematic
- ValueError → `optuna.TrialPruned`
- Unexpected errors → bubble up

### hist_reg_objective (Lines 191-228)
- Added docstring explaining error handling strategy
- Added NOTE about why broad except is problematic
- ValueError → `optuna.TrialPruned`
- Unexpected errors → bubble up

## Error Handling Strategy

1. **CatBoostError**: Raise `optuna.TrialPruned` - invalid model configuration
2. **ValueError**: Raise `optuna.TrialPruned` - data/configuration issues
3. **Unexpected errors**: Let propagate with clear error context

## Return Values Unchanged
- Classification: still maximizes average precision
- Regression: still minimizes mean absolute error
- Default return values (0.0, 1e9) not used anymore since exceptions are pruned
