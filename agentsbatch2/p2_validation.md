# P2 Fix Validation Report

## Test Results Summary

| Test | Status | Output |
|------|--------|--------|
| `py_compile recency_comparison.py` | ✅ PASS | No output (success) |
| `py_compile compare_recency.py` | ✅ PASS | No output (success) |
| `scripts/compare_recency.py --help` | ✅ PASS | Returns usage info instantly |

## Issues Discovered

1. **Missing p2_implementation.md file** - File not found at `agentsbatch2/p2_implementation.md`. Implementation details must be inferred from code review.

2. **No `validation_gap_years` parameter** - The code at `src/aac_adoption/analysis/recency_comparison.py:70` uses hardcoded calculation `train_end = test_start - 3` with no configurable parameter.

## Validation Gap Behavior

**Current Implementation (line 70):**
```python
train_end = test_start - 3  # Leave a 2-year validation gap
```

**Expected behavior for test_period="2024-2025" (test_start=2024):**
- With default gap=3: train_end = 2021 ✅
- With custom gap=2: train_end = 2022 ✅
- With custom gap=1: train_end = 2023 ✅

**Current State:** Gap is hardcoded as 3, not configurable via parameter.

## Production Readiness

**Status: ⚠️ NOT PRODUCTION-READY**

### Required Fixes:
1. Add `validation_gap_years` parameter to `run_recency_comparison()` function
2. Add `--validation-gap` CLI argument to `scripts/compare_recency.py`
3. Update implementation docs in `agentsbatch2/p2_implementation.md`

### Code Changes Needed:
- `src/aac_adoption/analysis/recency_comparison.py:70`: Use configurable gap instead of hardcoded `3`
- `scripts/compare_recency.py`: Add argparse argument for validation gap

## Command Outputs

```
$ python -m py_compile src/aac_adoption/analysis/recency_comparison.py
(no output)

$ python -m py_compile scripts/compare_recency.py
(no output)

$ python scripts/compare_recency.py --help
usage: compare_recency.py [-h] [--data-path DATA_PATH] [--output OUTPUT]
                          [--figure-output FIGURE_OUTPUT]
                          [--n-bootstraps N_BOOTSTRAPS]
                          [--iterations ITERATIONS]
                          [--test-period TEST_PERIOD] [--quick]

Compare recency strategies for model training with bootstrap CI and subgroup
analysis.

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to the modeling dataset CSV file.
  --output OUTPUT       Path to save the output CSV results.
  --figure-output FIGURE_OUTPUT
                        Path to save the output comparison figure.
  --n-bootstraps N_BOOTSTRAPS
                        Number of bootstrap iterations. Defaults to 5 in quick
                        mode, 100 otherwise.
  --iterations ITERATIONS
                        Number of CatBoost iterations. Defaults to 20 in quick
                        mode, 300 otherwise.
  --test-period TEST_PERIOD
                        Test period range (e.g. 2024-2025).
  --quick               Enable quick mode with reduced bootstrap iterations
                        and model complexity.
```

**Validation Date:** 2026-06-07
**Verified By:** Kilo Validator
