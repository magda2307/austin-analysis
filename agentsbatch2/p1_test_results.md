# P1 Fix Test Results - Batch2

**Test Status:** PASS  
**Date:** 2026-06-07  
**Time:** 15:08:01+02:00

## Run Command
```bash
python scripts/compare_recency.py --quick --n-bootstraps 5 --iterations 20
```

## Output Files Created
- ✅ `reports/figures/recency_strategy_comparison.png` (plot)
- ✅ `reports/tables/recency_strategy_comparison.csv` (data)

## CSV Data Verification
The CSV contains 10 rows of data with the following structure:
- 4 strategies tested: full_history, recent_5yr, recent_3yr, recency_weighted
- 3 subsets: combined, dogs, cats
- Metrics: pr_auc, roc_auc, brier, ece, mae with confidence intervals

## P1 Fix Verification
The fix in `src/aac_adoption/analysis/recency_comparison.py:372-373` handles negative yerr values:
- Line 372: `yerr_lower.append(max(0.0, val - l))`
- Line 373: `yerr_upper.append(max(0.0, u - val))`

This prevents negative error bar values that would cause matplotlib errors.

## Test Execution Time
~1 minute 30 seconds (15:08:25 - 15:09:55)

## Errors Encountered
None

## Conclusion
✅ All tests passed. The P1 fix successfully handles negative yerr values and the script completes without errors.
