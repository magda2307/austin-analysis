# Slice 11 - Fix Workflow & Documentation

**Date:** 2026-06-06  
**Status:** In Progress

---

## Issues Found

❌ **CRITICAL** - Weight formula inconsistent:
- `split.py:71-72` uses `intake_datetime` for weighting
- `compare_recency.py:17-19` uses `intake_year` for weighting

⚠️ **PARTIAL** - Missing thesis-grade elements:
- No bootstrap confidence intervals
- No statistical significance tests
- No subgroup analysis (dogs/cats)
- No visualization of results
- No validation metrics alongside test metrics

✅ **DONE**:
- All 4 strategies implemented correctly
- Same test period (2024-2025) used consistently
- Core metrics (PR-AUC, ROC-AUC, Brier, ECE, MAE) present

---

## Fix Plan

### 1. Standardize Weight Calculation
- **File:** `src/aac_adoption/models/split.py`
- **Action:** Replace `intake_datetime` → `intake_year` in lines 69-73
- **Reference:** `compare_recency.py` already uses correct `intake_year` approach

### 2. Add Bootstrap 95% CI
- **Approach:** Use existing `bootstrap_ci()` in `src/aac_adoption/models/bootstrap.py`
- **Scope:** All metrics (PR-AUC, ROC-AUC, Brier, ECE, MAE)
- **Implementation:** Add CI computation to `compare_recency.py`

### 3. Subgroup Analysis (Dogs/Cats)
- **Approach:** Split by `animal_type` column
- **Metrics:** Compute same metrics separately for dogs/cats
- **Output:** Append to main results DataFrame

### 4. Visualization
- **Plot:** Performance comparison bar chart
- **Metrics:** PR-AUC, ROC-AUC, Brier, ECE, MAE
- **Groups:** x-axis = animal subsets (dogs/cats/all)
- **Series:** strategy names
- **Save:** `reports/figures/recency_strategy_comparison.png`

### 5. Comprehensive CSV Output
**Required columns:**
- `strategy`, `train_years`, `test_years`, `subset`, `model`
- `pr_auc`, `pr_auc_lower`, `pr_auc_upper`
- `roc_auc`, `roc_auc_lower`, `roc_auc_upper`
- `brier`, `brier_lower`, `brier_upper`
- `ece`, `ece_lower`, `ece_upper`
- `mae`, `mae_lower`, `mae_upper`
- `n_samples` (optional)

---

## File Changes

### Modified Files

#### `scripts/compare_recency.py`
- ✅ Import `bootstrap_ci`, `matplotlib.pyplot`, `numpy`
- ✅ Add `compute_bootstrap_ci()` function
- ✅ Extend `main()` to compute CIs for all metrics
- ✅ Add subgroup loop for dogs/cats analysis
- ✅ Add `plot_performance_comparison()` function
- ✅ Save visualization to `reports/figures/`

#### `src/aac_adoption/models/evaluate.py`
- ✅ Add `subgroup_analysis()` function
- ✅ Keep existing `classification_metrics_with_ci()` for CI computation

#### `src/aac_adoption/models/split.py`
- ✅ Fix weight calculation to use `intake_year` (lines 69-73)

---

## Validation Steps

1. Run `scripts/compare_recency.py`
2. Verify CSV has all required columns
3. Check plot exists at `reports/figures/recency_strategy_comparison.png`
4. Validate CI ranges are reasonable (not NaN, sensible width)
5. Compare dogs/cats metrics vs all subset

---

## Expected Output

### CSV: `reports/tables/recency_strategy_comparison.csv`
```
strategy,test_years,subset,pr_auc,pr_auc_lower,pr_auc_upper,roc_auc,roc_auc_lower,roc_auc_upper,...
full_history_unweighted,2024-2025,all,0.85,0.82,0.88,0.91,0.88,0.93,...
recent_5_year,2024-2025,all,0.84,0.81,0.87,0.90,0.87,0.92,...
recent_3_year,2024-2025,all,0.83,0.80,0.86,0.89,0.86,0.91,...
recency_weighted,2024-2025,all,0.86,0.83,0.89,0.92,0.89,0.94,...
full_history_unweighted,2024-2025,dogs,0.87,0.84,0.90,0.93,0.90,0.95,...
...
```

### Plot: `reports/figures/recency_strategy_comparison.png`
- 5 subplots showing metric comparisons
- 3 columns (strategies), 2 rows (subsets)
- Bar colors = strategies
- x-axis = animal subsets

---

## Review Notes

- Weight standardization is CRITICAL for consistency
- Bootstrap CI must use cluster-aware resampling (animal_id)
- Subgroup analysis requires minimum sample checks
- Visualization should use consistent color scheme
- All outputs must follow naming conventions

---

## Next Steps

1. Execute comparison script
2. Verify all outputs
3. Update agent instructions in `AGENTS.md`
4. Mark Slice 11 as DONE
