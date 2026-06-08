# P2 Fix Summary: Permutation Importance Scoring

## Issue
Line 218 in `train_boosting.py` used `scoring="roc_auc"` for permutation importance on classification task, which did not match the training objective.

## Context
- Classifier training uses PR-AUC (`average_precision`) as primary validation metric
- `tune.py:102` and `tune.py:181` use `average_precision_score(y_va, preds)` for classification tuning
- Permutation importance should match training objective for consistency

## Fix Applied
Changed line 218 in `src/aac_adoption/models/train_boosting.py`:
```python
# Before
scoring="roc_auc",

# After  
scoring="average_precision",
```

## Verification
- Classification: Now uses `average_precision` (matches training objective)
- Regression: Uses `neg_mean_absolute_error` (appropriate for regression, no change needed)
- No other permutation importance calls found requiring changes

## Files Changed
- `src/aac_adoption/models/train_boosting.py` (line 218)

## Notes
- PR-AUC is used as validation-only metric (matches training methodology)
- _permutation_table function structure unchanged
- No other scoring logic modified
