# Fix Divergent Row Issue - Batch 3 P2

## Changes Made

### 1. Added Divergent Row Data Fixture (lines 30-53)

Created `divergent_row_data` fixture that generates test data with edge cases:

```python
@pytest.fixture
def divergent_row_data():
```

**Characteristics:**
- 200 samples with varied categorical values
- Includes edge cases: extreme age values (0, 10000, 15000, 20000 days)
- Unusual combinations: critical conditions, multiple intake types
- Known ground truth for alignment verification

**Divergent Rows:**
- Indices 5, 25, 50, 75, 100, 125, 150, 175 contain extreme values
- Age values: 0 or very old (10000-20000 days)
- Regression targets: 0, 45, 60, or 90 days (outside normal 1-30 range)

### 2. Added Regression Feature Alignment Test (lines 132-151)

Created `test_tune_models_regression_feature_alignment` with three assertions:

**Assertion 1 - Index Match:**
```python
assert X_reg.index.equals(y_reg.index), "Regression feature frame index must match regression target index"
```
Verifies feature frame and target share identical row indices.

**Assertion 2 - Row Count Match:**
```python
assert len(X_reg) == len(y_reg), "Regression feature frame row count must match regression target row count"
```
Ensures feature frame and target have equal number of rows.

**Assertion 3 - No Misaligned Rows:**
```python
assert X_reg.index.intersection(y_reg.index).equals(X_reg.index), "No misaligned rows in regression feature frame and target"
```
Confirms all feature frame rows have corresponding target rows.

### 3. Added Missing Imports (lines 7-12)

```python
from aac_adoption.models.split import make_time_split
from aac_adoption.features.feature_sets import model_feature_columns
```

## How Divergent Rows Are Handled

### Problem Identified

The original `test_tune_models_runs_successfully` test at line 74 used random data with the same random seed but:
- No explicit verification that regression features align with regression targets
- No test for divergent rows where misalignment could occur
- Used consistent random seed making data patterns predictable but not testing edge cases

### Solution Implementation

1. **Divergent Row Fixture:** Creates data with known problematic rows to test edge cases
2. **Index Alignment Checks:** Verifies feature frame and target share the same row indices
3. **Count Alignment Checks:** Ensures equal row counts between features and targets
4. **Intersection Verification:** Confirms no rows are missing from either frame

### Testing Process

The new test follows the same workflow as `tune_models()`:
1. Creates time-based split for classification target
2. Creates time-based split for regression target
3. Extracts feature columns and target vector
4. Verifies alignment before and after data preparation

This ensures the regression feature frame always aligns with the regression target, even with divergent edge cases.
