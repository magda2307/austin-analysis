# Batch 2 P2 Test Plan

**Date:** 2026-06-07  
**Target:** Configurable validation gap in recency comparison

---

## Test Scenarios

### 1. Default Behavior (Backward Compatibility)

**Objective:** Verify default parameter maintains original behavior

**Test Steps:**
1. Run CLI without `--validation-gap-years` flag
2. Check log output or results for train_years range

**Expected Results:**
- For test period "2024-2025", train_end should be 2021
- Validation gap should be 2 years (2022, 2023)
- All strategy train_years should end at 2021

**Test Command:**
```bash
python scripts/compare_recency.py --test-period "2024-2025" --quick --n-bootstraps 2 --iterations 10
```

---

### 2. Custom Gap: Smaller Gap (2 years)

**Objective:** Verify smaller validation gap works correctly

**Test Steps:**
1. Run with `--validation-gap-years 2`
2. Verify train_end calculation

**Expected Results:**
- For test period "2024-2025", train_end should be 2022
- Validation gap should be 1 year (2023)
- All strategy train_years should end at 2022

**Test Command:**
```bash
python scripts/compare_recency.py --test-period "2024-2025" --validation-gap-years 2 --quick --n-bootstraps 2 --iterations 10
```

---

### 3. Custom Gap: Larger Gap (5 years)

**Objective:** Verify larger validation gap works correctly

**Test Steps:**
1. Run with `--validation-gap-years 5`
2. Verify train_end calculation

**Expected Results:**
- For test period "2024-2025", train_end should be 2019
- Validation gap should be 4 years (2020, 2021, 2022, 2023)
- All strategy train_years should end at 2019
- Recent strategies should start at 2015 (2019-4) for 5yr, 2017 (2019-2) for 3yr

**Test Command:**
```bash
python scripts/compare_recency.py --test-period "2024-2025" --validation-gap-years 5 --quick --n-bootstraps 2 --iterations 10
```

---

### 4. Different Test Periods

**Objective:** Verify gap works with various test periods

**Test Steps:**
1. Test with "2023-2024"
2. Test with "2025-2026"

**Expected Results:**
- "2023-2024" + default gap=3 → train_end=2020
- "2025-2026" + default gap=3 → train_end=2022
- Gap calculation should be consistent across different test periods

**Test Command:**
```bash
python scripts/compare_recency.py --test-period "2023-2024" --quick --n-bootstraps 2 --iterations 10
python scripts/compare_recency.py --test-period "2025-2026" --quick --n-bootstraps 2 --iterations 10
```

---

### 5. Edge Case: Gap = 1

**Objective:** Verify minimum gap handling

**Test Steps:**
1. Run with `--validation-gap-years 1`

**Expected Results:**
- For test period "2024-2025", train_end should be 2023
- Validation gap: 0 years (immediately before test)
- All strategy train_years should end at 2023

**Test Command:**
```bash
python scripts/compare_recency.py --test-period "2024-2025" --validation-gap-years 1 --quick --n-bootstraps 2 --iterations 10
```

---

### 6. CLI Help Documentation

**Objective:** Verify CLI help text includes new parameter

**Test Steps:**
1. Run `python scripts/compare_recency.py --help`
2. Search for `--validation-gap-years`

**Expected Results:**
- Flag appears in help output
- Default value (3) is shown
- Help text describes the parameter clearly

**Test Command:**
```bash
python scripts/compare_recency.py --help | Select-String "validation-gap"
```

---

### 7. Python API Usage

**Objective:** Verify function can be called directly with parameter

**Test Steps:**
1. Import `run_recency_comparison` function
2. Call with explicit `validation_gap_years` parameter
3. Check results

**Expected Results:**
- Function accepts parameter without error
- Results reflect the custom gap value
- Default parameter works when omitted

**Test Code:**
```python
from aac_adoption.analysis.recency_comparison import run_recency_comparison

# With explicit parameter
results = run_recency_comparison(df, validation_gap_years=2)

# Without parameter (should use default=3)
results_default = run_recency_comparison(df)
```

---

### 8. Strategy-Specific Validation

**Objective:** Verify all strategies respect the gap parameter

**Test Steps:**
1. Run comparison with custom gap
2. Check all strategies' train_years

**Expected Results:**
- "full_history": starts at 2013, ends at (test_start - gap)
- "recent_5yr": ends at (test_start - gap), starts at max(2013, end-4)
- "recent_3yr": ends at (test_start - gap), starts at max(2013, end-2)
- "recency_weighted": same as full_history

**Verification Method:**
```python
results = run_recency_comparison(df, validation_gap_years=2, quick=True)
assert all(results['train_years'].str.endswith('-2022'))
```

---

## Automated Test Suite Additions

### New Test Function

```python
def test_recency_comparison_validation_gap_configurable():
    """Test that validation_gap_years parameter is properly configurable."""
    import pandas as pd
    from aac_adoption.analysis.recency_comparison import run_recency_comparison
    
    df = load_test_dataset()
    
    results_default = run_recency_comparison(df, validation_gap_years=3, quick=True)
    results_custom = run_recency_comparison(df, validation_gap_years=2, quick=True)
    
    train_end_default = int(results_default['train_years'].iloc(0).split('-')[1])
    train_end_custom = int(results_custom['train_years'].iloc(0).split('-')[1])
    
    assert train_end_default == 2021, f"Expected 2021, got {train_end_default}"
    assert train_end_custom == 2022, f"Expected 2022, got {train_end_custom}"
```

---

## Regression Tests

### Existing Tests to Verify

1. `test_recency_comparison_output_schema`
   - Ensure new parameter doesn't break output schema
   
2. `test_recency_comparison_strategy_lengths`
   - Verify strategy train_years still计算 correctly
   
3. `test_recency_comparison_combined_subset`
   - Ensure combined subset results still valid

---

## Performance Tests

### None Required

The change is a parameter substitution, not a computational change. No performance impact expected.

---

## Acceptance Criteria

- [ ] Default behavior (gap=3) produces same results as before
- [ ] Custom gap values (1, 2, 5) produce correct train_end calculations
- [ ] CLI --help displays new parameter
- [ ] All strategies respect the gap parameter
- [ ] Backward compatibility maintained
- [ ] Python API accepts parameter correctly

---

**Test Plan by:** Kilo (Batch 2 P2)  
**Status:** READY FOR EXECUTION
