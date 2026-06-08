# Implementer Changes - Batch 2

## Files Modified

1. `tests/test_yearly_backtesting.py`
2. `src/aac_adoption/models/yearly_backtesting.py`
3. `scripts/compare_recency.py`

---

## Specific Line Changes

### P1: Fix test quick parameter mismatch (tests/test_yearly_backtesting.py:278)

**Change:**
```python
# Before (line 278):
quick=True,

# After (line 278):
quick=False,
```

**Reasoning:** The test `test_yearly_backtesting_horizon_targets` expects 6 windows [2019-2024] from `six_year_fixture`, but was using `quick=True`. The `quick` mode parameter likely reduces iterations which may affect whether all test windows complete successfully. Changing to `quick=False` ensures the full test runs with all 6 expected years.

---

### P2: Add failed rows to output when strict=False (src/aac_adoption/models/yearly_backtesting.py:343-365)

**Change:**
```python
# Before (lines 343-346):
except Exception as e:
    logger.error(f"Error training {model_name} on {subset_name} for test year {test_year}: {e}", exc_info=True)
    if strict:
        raise e

# After (lines 343-365):
except Exception as e:
    logger.error(f"Error training {model_name} on {subset_name} for test year {test_year}: {e}", exc_info=True)
    if strict:
        raise e
    else:
        failed_result = {
            "train_years": train_period,
            "test_year": test_year,
            "subset": subset_name,
            "animal_subset": subset_name,
            "model": model_name,
            "model_name": model_name,
            "pr_auc": None,
            "roc_auc": None,
            "brier": None,
            "brier_score": None,
            "ece": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "train_rows": len(X_train) if 'X_train' in locals() else 0,
            "test_rows": len(X_test) if 'X_test' in locals() else 0,
            "error": str(e),
        }
        results.append(failed_result)
```

**Reasoning:** When `strict=False`, the original code silently dropped failed rows after logging. This change adds a failed result entry to the output DataFrame with `None` for all metric columns and includes the error message in an `error` column. This ensures failures are visible in the output rather than being silently dropped.

---

### P3: Align CLI default to module default (scripts/compare_recency.py:54)

**Change:**
```python
# Before (line 54):
default=3,

# After (line 54):
default=4,
```

**Reasoning:** The module default at `src/aac_adoption/analysis/recency_comparison.py:37` is `validation_gap_years=4`. The CLI was using `default=3`, creating inconsistency. Aligning to 4 ensures CLI and module use the same default validation gap.

---

## Summary

- All three issues from validator findings have been addressed
- Changes maintain backward compatibility for correct behavior
- Error handling now properly surfaces failures in non-strict mode
