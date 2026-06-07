# Batch 4 Fix Plan

## P1: Fix test_dashboard_data.py line 260 (CatBoost log-transformation)

**File:** `tests/test_dashboard_data.py`

**Problem:** 
- Mock returns `15.0` for regressor prediction
- Code at `data.py:402-403` applies `exp(raw_days) - 1.0` for catboost
- Result: `exp(15) - 1 = 3269016.37` instead of `15.0`

**Fix:**
Change line 260 from:
```python
mock_regressor.predict.return_value = np.array([15.0])
```
To:
```python
mock_regressor.predict.return_value = np.array([math.log1p(15)])
```

**Dependencies:** None - test-only change

---

## P2: Fix data.py line 352 models_dir override (wrong family directory)

**File:** `src/aac_adoption/dashboard/data.py`

**Problem:**
- When `models_dir is not None`, both classifier and regressor get same `models_dir`
- This bypasses `_infer_models_dir()` which returns correct family dirs:
  - "catboost" -> "models/advanced"
  - "boosting" -> "models/boosting"
  - others -> "models/baseline"
- Example: hist_gradient_boosting classifier (baseline) + catboost regressor (advanced) both get forced to same dir

**Current Code:**
```python
if models_dir is not None:
    clf_dir = models_dir
    reg_dir = models_dir
```

**Fix:**
Only override `models_dir` when both classifier and regressor are expected to be in the same dir. Otherwise, preserve `_infer_models_dir()` per model:

```python
if models_dir is not None:
    # Only override if same dir for both (e.g., debug/testing)
    # Otherwise preserve inferred dirs per model family
    clf_dir = models_dir
    reg_dir = models_dir
```

**Better fix - preserve per-model inference:**
```python
if models_dir is not None:
    # Override both to same path (useful for debug/testing)
    clf_dir = models_dir
    reg_dir = models_dir
else:
    # Already handled above - infer from model names
    pass
```

Or more explicit:
```python
# After inferring from model names, only override if models_dir provided
# Keep the per-model inference logic above and don't override both to same
# OR add conditional to only override when appropriate
```

**Decision needed:** Should `models_dir` override apply to both or should we add separate `classifier_models_dir` and `regressor_models_dir`? For minimal fix, keep current behavior but document it's for same-family models only.
