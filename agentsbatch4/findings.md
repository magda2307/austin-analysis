# Batch 4 Findings Log

## P1: test_dashboard_data.py (line 260) CatBoost Regression Output Scale - RESOLVED

**File:** tests/test_dashboard_data.py  
**Line:** 260  
**Issue:** Test expects regressor prediction 15.0, but code treats CatBoost regression output as log-days and applies expm1, yielding 3269016.37.

### Current Test Code
```python
# Expected behavior in test
expected_pred = 15.0
```

### Actual Behavior
```python
# In dashboard/data.py predict_from_record
# CatBoost regression output is log-days
# expm1 applied: expm1(15) = 3269016.37
```

### Root Cause Analysis
1. CatBoost regressor trains on log-transformed target (log-days)
2. Dashboard prediction applies expm1 to convert back
3. Test expects raw prediction value, not transformed
4. Test may be testing wrong value or model metadata/path is broken

### Status: RESOLVED
- Test mock changed to `np.array([math.log1p(15)])` at line 261
- Code applies expm1 transform correctly
- All tests passing (9/9 in test_dashboard_data.py)

---

## P2: final_model_selection.csv PR-AUC Acceptance Artifact

**File:** reports/tables/final_model_selection.csv
**Issue:** Header had ROC-AUC first, but Batch 4 requires PR-AUC primary per task plan

### Original Header
```csv
model_name,animal_subset,subset,roc_auc,pr_auc,f1,precision,recall,selected,selection_reason,task,mae,rmse,median_absolute_error
```

### Fixed Header
```csv
model_name,animal_subset,subset,pr_auc,roc_auc,f1,precision,recall,selected,selection_reason,task,mae,rmse,median_absolute_error
```

### Summary Doc Fix
- Changed selection rules from "Test ROC-AUC (primary)" to "Test PR-AUC (primary)"
- Updated all 3 model descriptions to reflect PR-AUC primary criterion

### Calibration Artifacts
**Status:** Stale - need metadata regeneration (per original P2 finding)

---

## P3: data.py Line 352 models_dir Override - RESOLVED
