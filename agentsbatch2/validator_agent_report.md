# Validator Agent Report - Batch 2

**Date**: 2026-06-07  
**Validator Agent**: Batch 2 Validator

---

## Issue P1: Test expects 6 windows but uses quick=True

**Location**: `tests/test_yearly_backtesting.py:281`

### Severity
**HIGH** - Test incorrectly expects different behavior than quick mode implements

### Root Cause Analysis

The test at line 278-283 calls `run_yearly_backtesting()` with `quick=True` (line 278), but expects all 6 years `[2019, 2020, 2021, 2022, 2023, 2024]` in the results. However, looking at `src/aac_adoption/models/yearly_backtesting.py:103-106`:

```python
if quick:
    test_years = [2019, 2023]  # Only 2 years!
else:
    test_years = [2019, 2020, 2021, 2022, 2023, 2024]  # All 6 years
```

The `quick` mode is explicitly designed to run only 2 test windows (2019 and 2023) to speed up testing. The test is asserting the wrong expected values.

### Recommended Fix Approach

**Option A: Fix the test** (RECOMMENDED)
- Change expected_years from `[2019, 2020, 2021, 2022, 2023, 2024]` to `[2019, 2023]`
- This preserves the test's intent to validate quick mode

**Option B: Remove quick=True from this test**
- If the test's goal is to validate all 6 years, remove the `quick=True` parameter
- This would make the test run slower but validate the complete behavior

**Rationale**: Option A is preferred because:
1. The test name suggests it's validating yearly backtesting behavior
2. Quick mode is explicitly designed for faster iteration
3. Having both quick and non-quick tests provides better coverage
4. The test comment/documentation should clarify which mode is being tested

**Action**: Update test at line 282 to use `expected_years = [2019, 2023]`

---

## Issue P2: strict=False silently drops failed rows

**Location**: `src/aac_adoption/models/yearly_backtesting.py:343-366`

### Severity
**MEDIUM** - Silent failure could hide data quality issues

### Root Cause Analysis

Current behavior (lines 343-366):
```python
except Exception as e:
    logger.error(f"Error training {model_name} on {subset_name} for test year {test_year}: {e}", exc_info=True)
    if strict:
        raise e
    else:
        failed_result = {  # Create row with None metrics
            "train_years": train_period,
            "test_year": test_year,
            ...
            "pr_auc": None,
            "roc_auc": None,
            ...
            "error": str(e),
        }
        results.append(failed_result)
```

When `strict=False`, the exception is logged and a failed result row is added with `None` metrics and the error message. When `strict=True`, the exception is re-raised.

Current implementation:
- ✅ Logs errors with full stack trace
- ✅ Adds row to output with error message
- ⚠️ Metrics are `None` - difficult to distinguish from actual missing data
- ⚠️ Caller must manually check for non-None metrics or error column

### Recommended Fix Approach

**Recommended**: Keep current behavior but **add documentation and helper validation**

The current approach (log + include with None values) is actually reasonable for batch processing where you want one failure to not stop everything. However, improvements needed:

1. **Add helper method** to filter failed results
2. **Update docstring** to clarify behavior
3. **Add warning** when failures are included

**Implementation**:
```python
# After results_df is created, add:
failed_count = results_df["error"].notna().sum()
if failed_count > 0:
    logger.warning(f"Results include {failed_count} failed row(s). Check 'error' column for details.")
```

**Alternative**: Add a `fail_fast` parameter that behaves like current `strict` (True = raise, False = skip row silently). This would separate two concerns:
- `strict`: Fail on error (current True behavior)
- `fail_fast`: Stop processing on first failure

**Action**: Documentation update + optional helper method to detect failures in results

---

## Issue P3: CLI default differs from module default

**Location**: `scripts/compare_recency.py:52` vs `src/aac_adoption/analysis/recency_comparison.py:37`

### Severity
**HIGH** - Configuration mismatch can cause inconsistent behavior

### Root Cause Analysis

**CLI default** (`scripts/compare_recency.py:52-56`):
```python
parser.add_argument(
    "--validation-gap-years",
    type=int,
    default=3,  # CLI defaults to 3
    help="Gap years between training end and test start (default: 3).",
)
```

**Module default** (`src/aac_adoption/analysis/recency_comparison.py:31-37`):
```python
def run_recency_comparison(
    ...
    validation_gap_years: int = 4,  # Module defaults to 4
    ...
):
```

The CLI passes `args.validation_gap_years` to the module, but if CLI is run without the argument, it defaults to 3. If the module is called directly (not via CLI), it defaults to 4.

This creates two problems:
1. **Inconsistent behavior**: Running the CLI vs importing and calling the module directly yields different defaults
2. **Hidden coupling**: CLI depends on module parameter but defaults don't match

### Recommended Fix Approach

**Option A: Make them match** (RECOMMENDED)
- Choose one value as canonical (4 is used in module, so likely intentional)
- Update CLI default to match: `default=4`
- Update help text: `help="Gap years... (default: 4)."`

**Option B: Document the split**
- If there's a reason for different defaults, document why
- Add explicit validation in CLI to warn if value differs from module intent
- Add comment explaining the split

**Rationale**: 
- Module default (4) is documented in the function signature and docstring
- The module's docstring explains: "validation_gap_years: Gap years between training end and test start (default=4, meaning train_end = test_start - validation_gap_years; validation window = gap - 1 years)"
- CLI should not override module defaults without clear justification
- Single source of truth prevents configuration drift

**Action**: Update CLI default from 3 to 4 in `scripts/compare_recency.py:54`

---

## Summary Table

| Issue | Severity | Current Behavior | Recommended Fix | Effort |
|-------|----------|------------------|-----------------|--------|
| P1 | HIGH | Test expects 6 years with quick=True | Update expected_years to [2019, 2023] | 1 line |
| P2 | MEDIUM | Errors logged, None values in row | Add warning when failures present | 3 lines |
| P3 | HIGH | CLI=3, Module=4 default mismatch | Make CLI default=4 | 1 line |

---

## Files to Modify

1. `tests/test_yearly_backtesting.py` - Line 282
2. `src/aac_adoption/models/yearly_backtesting.py` - Lines 369-371 (add warning)
3. `scripts/compare_recency.py` - Line 54

---

*Report generated by Validator Agent for Batch 2*
