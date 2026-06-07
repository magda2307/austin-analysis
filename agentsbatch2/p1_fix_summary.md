# P1 Issue Fix Summary

## Problem
Python script `scripts/compare_recency.py --quick --n-bootstraps 5 --iterations 20` failed after writing CSV because plot yerr had negative values. The matplotlib errorbar function cannot handle negative error values, causing the script to crash when generating the performance comparison plot.

## Root Cause
In `src/aac_adoption/analysis/recency_comparison.py` at lines 372-373, the yerr calculation computed:
- `yerr_lower.append(val - l)` - differences between point estimate and CI lower bound
- `yerr_upper.append(u - val)` - differences between CI upper bound and point estimate

When CI bounds were ordered incorrectly (l > val or u < val), these values became negative, causing matplotlib to fail.

## Fix Applied
Added `max(0.0, ...)` clamping to ensure yerr values are always >= 0:

**Line 372:** Changed `yerr_lower.append(val - l)` to `yerr_lower.append(max(0.0, val - l))`

**Line 373:** Changed `yerr_upper.append(u - val)` to `yerr_upper.append(max(0.0, u - val))`

## Location
File: `src/aac_adoption/analysis/recency_comparison.py`
Function: `plot_performance_comparison` (lines 327-406)
Specific fix: Lines 372-373
