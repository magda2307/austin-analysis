# P2 Gap Analysis: `test_start - 3` in recency_comparison.py

**File:** `src/aac_adoption/analysis/recency_comparison.py`  
**Line:** 70  
**Issue:** Hardcoded training end calculation with implicit validation gap

---

## Current Behavior

Line 70 contains:
```python
train_end = test_start - 3  # Leave a 2-year validation gap before the test period (e.g., 2021 for 2024 test start)
```

### How It Works

With `test_period="2024-2025"` (default):
1. `test_years = [2024, 2025]` -> `test_start = 2024`
2. `train_end = 2024 - 3 = 2021`
3. Training data spans 2013 to 2021 (inclusive)
4. **Gap years 2022 and 2023 are excluded** (2-year validation gap)

### Strategy Implementation

| Strategy | Training Years (for 2024-2025 test) | Logic |
|----------|--------------------------------------|-------|
| `full_history` | 2013-2021 | `range(2013, train_end + 1)` |
| `recent_5yr` | 2017-2021 | `range(max(2013, train_end-4), train_end + 1)` |
| `recent_3yr` | 2019-2021 | `range(max(2013, train_end-2), train_end + 1)` |
| `recency_weighted` | 2013-2021 | Same as full_history, but weighted |

---

## Analysis: Configuration vs Documentation

### 1. Is This a Configuration Issue?

**No** - the gap is intentionally designed into the methodology:

- The validation gap (years 2022-2023) prevents temporal leakage between training and test periods
- This is a standard time-series cross-validation practice: train on past, validate on recent unseen data, test on future
- The gap year(s) serve as a buffer zone to ensure the model isn't trained on data too close to the test period

### 2. Is This an Intentional Design Decision?

**Yes** - but **poorly documented**:

- The code comment states "Leave a 2-year validation gap" correctly
- However, the `- 3` magic number does not clearly communicate:
  - Why 3 years subtracted to get a 2-year gap
  - That this gap size is a configurable methodology choice
  - How the gap interacts with test period specification

### 3. Is This a Bug?

**No** - the math is correct:
- `test_start = 2024`
- `train_end = 2024 - 3 = 2021`
- Excluded years: 2022, 2023 (exactly 2 years)

The gap of 2 years is correct. The issue is **lack of configurability** and **unclear documentation of the relationship** between:
- `test_start`
- `train_end`
- Gap size

---

## Type of Issue Identified

| Aspect | Assessment |
|--------|-----------|
| **Behavior** | Correct (2-year gap achieved) |
| **Intent** | Clear (validation gap intended) |
| **Documentation** | Partial (comment explains result, not relationship) |
| **Configurability** | Missing (gap size is hardcoded) |
| **Risk** | Medium (hardcoded may be wrong for different test periods) |

---

## Recommendations

### Recommendation 1: Add Gap Size Parameter (CONFIGURE)

Add `gap_years` parameter to `run_recency_comparison()`:

**Location:** `src/aac_adoption/analysis/recency_comparison.py`, line 31-37

**Change:**
```python
def run_recency_comparison(
    df: pd.DataFrame,
    n_bootstraps: int = 100,
    iterations: int = 300,
    test_period: str = "2024-2025",
    gap_years: int = 2,  # NEW: Configurable validation gap
    quick: bool = False,
) -> pd.DataFrame:
```

**Update comment on line 70:**
```python
# Calculate training end with configurable gap before test period
# gap_years=2 means skip 2 years between training end and test start
train_end = test_start - gap_years
```

**Rationale:**
- Different use cases may need different gap sizes
- Makes methodology explicit and configurable
- Default `gap_years=2` preserves current behavior

---

### Recommendation 2: Improve Comment Documentation (DOCUMENT)

**Location:** `src/aac_adoption/analysis/recency_comparison.py`, line 70

**Current:**
```python
train_end = test_start - 3  # Leave a 2-year validation gap before the test period (e.g., 2021 for 2024 test start)
```

**Improved:**
```python
# Calculate training end with 2-year validation gap before test period
# gap_years=2 means exclude 2 years (test_start-2 to test_start-1) between
# training data (ending at train_end) and test data (starting at test_start)
train_end = test_start - gap_years
```

**Also update docstring** to document the gap:
```python
        test_period: Test period years, e.g. "2024-2025"
        gap_years: Years to exclude between training end and test start (default: 2)
```

---

### Recommendation 3: Update CLI Wrapper (CLI)

**Location:** `scripts/compare_recency.py`, after line 45

**Add gap_years argument:**
```python
    parser.add_argument(
        "--gap-years",
        type=int,
        default=2,
        help="Years to exclude between training end and test start (validation gap).",
    )
```

**Update function call on line 77:**
```python
    results_df = run_recency_comparison(
        df=df,
        n_bootstraps=n_bootstraps,
        iterations=iterations,
        test_period=args.test_period,
        gap_years=args.gap_years,  # NEW
        quick=args.quick,
    )
```

---

### Recommendation 4: Check for Other Hardcoded Values (FILE WIDE SCAN)

**Other occurrences found:**

1. **Line 76, 81, 86** - Hardcoded year ranges (2013 as start year)
   - This is acceptable as 2013 is the dataset start year
   - Could optionally be exposed as `min_train_year` parameter

2. **Line 136** - `compute_recency_weights` uses hardcoded `start_year=2013, end_year=train_end`
   - This should use the same `train_years` from the strategy's train_years
   - **Bug potential:** If strategy train_years do not start at 2013, weights calculation is inconsistent

**Suggested fix for line 136:**
```python
# Instead of:
sample_weight = compute_recency_weights(df[train_mask], start_year=2013, end_year=train_end)

# Use:
min_train = min(strat["train_years"])
max_train = max(strat["train_years"])
sample_weight = compute_recency_weights(df[train_mask], start_year=min_train, end_year=max_train)
```

---

## Code Locations Requiring Changes

| File | Line | Change Type | Priority |
|------|------|-------------|----------|
| `src/aac_adoption/analysis/recency_comparison.py` | 31-37 | Add `gap_years` parameter | HIGH |
| `src/aac_adoption/analysis/recency_comparison.py` | 70 | Update comment + formula | HIGH |
| `src/aac_adoption/analysis/recency_comparison.py` | 136 | Fix recency weights to use actual train_years range | MEDIUM |
| `src/aac_adoption/analysis/recency_comparison.py` | 50-55 | Update docstring | LOW |
| `scripts/compare_recency.py` | after 45 | Add `--gap-years` CLI arg | HIGH |
| `scripts/compare_recency.py` | 77-83 | Pass `gap_years` to function | HIGH |

---

## Summary

**Classification:** This is **intentional design** (validation gap) that requires:
1. **Configuration** - Make gap size configurable via `gap_years` parameter
2. **Documentation** - Clarify the relationship between `test_start`, `train_end`, and gap size
3. **CLI exposure** - Add command-line argument for gap years

**Not a bug** - the current math produces the correct 2-year gap. The issue is that this methodology choice is **not configurable** and the **relationship is not clearly documented**.

---

## Acceptance Criteria

After implementing these changes:
- `run_recency_comparison(gap_years=3)` should produce 3-year gap (train_end = test_start - 3)
- `run_recency_comparison(gap_years=1)` should produce 1-year gap (train_end = test_start - 1)
- Default behavior (`gap_years=2`) matches current `test_start - 3` behavior
- `--help` shows `--gap-years` option
- Code comments explain the gap calculation clearly
