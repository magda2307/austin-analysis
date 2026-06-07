# P3 Fix Summary: Validation-Only Rule Violation

## Fix Details

### File Changed
`C:\Users\paula\Documents\mgr pjatk\src\aac_adoption\models\train_boosting.py`

### Line Numbers Changed
Lines 138-147 (added validation check after function signature)

### Exact Code Inserted
```python
if split.validation.empty:
    raise ValueError(
        f"Permutation importance requires validation data for {split.animal_subset}. "
        f"Validation split is empty. This violates validation-only methodology. "
        f"Check time-based split parameters (validation_years=2022-2023)."
    )
```

### Description
Added exception at the start of `_permutation_table` function (line 138) that raises ValueError when `split.validation` is empty. The error message is clear and actionable for thesis writing, explicitly stating:
1. The requirement (permutation importance needs validation data)
2. The problem (validation split is empty)
3. The methodology violation (validation-only methodology)
4. Actionable guidance (check time-based split parameters)

### P2 Risk Addressed
✅ **FIXED** - The validation-only rule violation at lines 138-139 has been resolved. Code no longer falls back to `split.test` when `split.validation` is empty; instead it raises a clear error message.

## Test Results
```
2 passed in 15.90s
```
Test command: `python -m pytest tests/test_train_boosting_outputs.py -q -k "permutation"`
