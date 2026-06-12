# Phase 1 Deep Testing Report

## Executive Summary

An audit of the Phase 1 (Tasks 1–5) implementation of the AAC Thesis Closeout was performed. The core data processing logic for episode matching, rolling volume context features, and lagged weather features is logically sound and mathematically correct. Crucially, the implementation is designed to prevent data leakage and ensure outcome independence.

However, there are major discrepancies between the implementation files and the test suites. Out of 5 test files, 4 currently fail because the tests expect deprecated interfaces (e.g., tuple unpacking instead of the `MatchResult` dataclass, horizon target columns directly on the main modeling dataset instead of the separate horizon dataset, and missing parameters/mock weather dates in test setups).

## Task 1 & 2: Episode Matching Integrity

### 1. `MatchResult` Return Dataclass
- **Status**: **PASS** (Implementation) | **FAIL** (Tests Integration)
- **Code Logic Analysis**: `match_intakes_to_future_outcomes()` in [match_records.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/match_records.py#L73-L78) returns a `MatchResult` dataclass with `matched_episodes` (DataFrame), `unresolved_intakes` (DataFrame), and `unmatched_intakes` (int). 
- **Bug/Violation**: In [test_match_records.py](file:///c:/Users/paula/Documents/mgr%20pjatk/tests/test_match_records.py#L58), the tests attempt to unpack the return value as a tuple: `result, unmatched = match_intakes_to_future_outcomes(intakes, outcomes)`. This triggers a `TypeError: cannot unpack non-iterable MatchResult object`.

### 2. Boundary Enforcement
- **Status**: **PASS** (Code Logic) | **FAIL** (Tests Integration)
- **Code Logic Analysis**: The outcome matching checks `outcome_datetime >= intake_datetime` (ensured by the `while` loop sorting on line 123) and strictly checks `outcome_datetime < next_intake_time` on line 133. This guarantees that outcomes never cross a later intake boundary.
- **Bug/Violation**: When there are unresolved intakes, `match_intakes_to_future_outcomes()` requires `extract_end_date` to be provided to compute the follow-up days. In [test_match_records.py](file:///c:/Users/paula/Documents/mgr%20pjatk/tests/test_match_records.py#L113), the test `test_match_records_unmatched_intakes` fails because it omits `extract_end_date`, causing a `ValueError` to be raised.

### 3. Dataset Build Result & Supervised Targets
- **Status**: **PASS**
- **Code Logic Analysis**: [build_dataset.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/build_dataset.py#L115-L127) uses `DatasetBuildResult` to separate `dataset` (matched episodes) and `unresolved_intakes`. Supervised targets (`classification_target` and `regression_target_days`) are only computed on the matched episodes dataset (lines 139-140) and are not present on `unresolved_intakes`.

### 4. Count Conservation
- **Status**: **PASS**
- **Code Logic Analysis**: Enforced via a strict assertion on line 120 of [build_dataset.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/build_dataset.py#L120):
  ```python
  assert len(matched) + unmatched_intakes == len(clean_intake_df), "Intake count conservation failed"
  ```

### 5. Target Exclusion for Unresolved Intakes
- **Status**: **PASS**
- **Code Logic Analysis**: `unresolved_intakes` are enriched with intake features but never receive the supervised columns `classification_target` or `regression_target_days`.

---

## Task 1A: Horizon Cohort Logic

### 6. Horizon Labeling Truth Table
- **Status**: **PASS** (Implementation) | **FAIL** (Tests Integration)
- **Code Logic Analysis**: `build_horizon_dataset()` in [build_dataset.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/build_dataset.py#L295-L334) accurately implements the truth table using the following conditions:
  - `adopted_in_Hd = 1.0` if matched adoption occurs within H days (`cond1`)
  - `adopted_in_Hd = 0.0` if matched non-adoption occurs within H days (`cond2`), matched outcome occurs after H days (`cond3`), or unresolved intake has >= H days follow-up (`cond4`)
  - `adopted_in_Hd = NaN` if unresolved intake has < H days follow-up (default initialization)
- **Bug/Violation**: [test_horizon_targets.py](file:///c:/Users/paula/Documents/mgr%20pjatk/tests/test_horizon_targets.py#L41) asserts that these columns exist on `build_modeling_dataset(...).dataset`. In the code, horizon columns are only generated in the separate horizon dataset created by `build_horizon_dataset()`.

### 7. `HorizonDatasetBuildResult`
- **Status**: **PASS**
- **Code Logic Analysis**: Defined and used correctly in [build_dataset.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/build_dataset.py#L58-L65).

---

## Task 4: Intake Volume Context

### 8. Count Derivation from Raw Intakes
- **Status**: **PASS**
- **Code Logic Analysis**: Derived strictly from `raw_intakes` in [context_data.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/context_data.py#L216) to ensure the rolling volumes reflect the actual shelter density rather than the modeling subset.

### 9. Calendar Windows & Exclusions
- **Status**: **PASS** (with a minor logic gap for simultaneous duplicate timestamps)
- **Code Logic Analysis**: Computed in [rolling_features_cache.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/features/rolling_features_cache.py#L7-L33) using `end_indices - start_indices` which excludes the current index `i`.
- **Subtle Logic Gap**: If two intakes are recorded with the exact same timestamp `t`, the one sorted first is counted for the second one, despite both occurring at `t` (which violates the strict exclusive upper bound `[t-window, t)`).

### 10. Outcome Independence
- **Status**: **PASS**
- **Code Logic Analysis**: Computations rely strictly on intake timestamps and animal IDs; outcome timestamps/statuses are completely omitted.

### 11. Volume Threshold Application Sequence
- **Status**: **PASS**
- **Code Logic Analysis**: Applied on line 381 of [build_dataset.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/build_dataset.py#L381) after the context features are merged.

---

## Task 5: Weather Lagging

### 12. Weather Lagging by 1 Day
- **Status**: **PASS** (Implementation) | **FAIL** (Tests Integration)
- **Code Logic Analysis**: Handled in [context_data.py](file:///c:/Users/paula/Documents/mgr%20pjatk/src/aac_adoption/data/context_data.py#L208) via `intake_dates - pd.Timedelta(days=1)`.
- **Bug/Violation**: In [test_build_dataset.py](file:///c:/Users/paula/Documents/mgr%20pjatk/tests/test_build_dataset.py#L224), the test provides daily weather matching the intake date but asserts that weather is successfully joined. Because of the 1-day lag, it looks for weather on the prior day (which is not provided in the test mock data), returning `NaN` and failing the assertion.
- **Bug/Violation**: In [test_context_data.py](file:///c:/Users/paula/Documents/mgr%20pjatk/tests/test_context_data.py#L50), `add_context_features` is called using invalid parameter names (`weather` instead of `weather_daily`, `animal_requests` instead of `requests_311`) and misses the required `raw_intakes` argument.

### 13. Weather Nullability
- **Status**: **PASS**
- **Code Logic Analysis**: Uses nullable `boolean` pandas series (with `pd.NA`) for `is_extreme_heat` and `is_rainy_day`.

### 14. Weather Lag Metadata
- **Status**: **PASS**
- **Code Logic Analysis**: `context_weather_lag_days: 1` is successfully recorded in the metadata output.

---

## Test Run Results

The targeted tests were run individually with the following results:
1. `tests/test_match_records.py`: **FAIL**
   - *Error*: `TypeError: cannot unpack non-iterable MatchResult object`
   - *Error*: `ValueError: extract_end_date is required to process unresolved intakes`
2. `tests/test_build_dataset.py`: **FAIL**
   - *Error*: `AssertionError: assert 'is_censored' in Index([...])`
   - *Error*: `AssertionError: assert nan == 96` (due to weather lagging not mocked correctly in tests)
3. `tests/test_horizon_targets.py`: **FAIL**
   - *Error*: `AssertionError: assert 'adopted_in_7d' in Index([...])`
4. `tests/test_context_data.py`: **FAIL**
   - *Error*: `TypeError: add_context_features() got an unexpected keyword argument 'weather'`
5. `tests/features/test_rolling.py` (referenced as `test_rolling_features.py`): **PASS**

---

## Bugs Found

1. **`MatchResult` Unpacking Bug**: The test suite in `tests/test_match_records.py` is incompatible with the dataclass returned by `match_intakes_to_future_outcomes()`.
2. **Missing `extract_end_date` in matching tests**: `tests/test_match_records.py` fails to pass `extract_end_date` when testing unmatched intakes.
3. **Horizon Target Placement Mismatch**: `tests/test_horizon_targets.py` expects horizon columns on the core modeling dataset, whereas they are only populated in the separate horizon dataset.
4. **`is_censored` and `followup_days_censored` Absence**: `tests/test_build_dataset.py` asserts columns `is_censored` and `followup_days_censored` exist in the modeling dataset, but they are not produced by `build_modeling_dataset()`.
5. **Weather Lag Test Mocking Bug**: `tests/test_build_dataset.py` has a test mock bug where it populates weather for the day of intake, but due to weather lagging, the system queries weather for the prior day, resulting in a missing join (`NaN`).
6. **`add_context_features` Parameter/Argument Incompatibility**: `tests/test_context_data.py` calls the function with incorrect signatures (`weather`, `animal_requests` and no `raw_intakes`).
7. **Simultaneous Intake Count Overestimation**: In `rolling_features_cache.py`, if multiple intakes occur at the exact same timestamp, the sorting causes later ones to count prior ones as "prior" volumes, despite occurring at the same second.

---

## Recommendations

1. **Update `tests/test_match_records.py`**: Refactor unpacking code to access attributes of the `MatchResult` object (e.g. `result.matched_episodes`) and provide an `extract_end_date` argument.
2. **Align Horizon Dataset Expectations**: Update `tests/test_horizon_targets.py` to call `build_horizon_dataset()` to obtain the horizon targets, rather than expecting them on the base modeling dataset.
3. **Fix columns assertions in `tests/test_build_dataset.py`**: Remove `is_censored` and `followup_days_censored` assertions or compute them appropriately.
4. **Fix Weather Mocking Dates**: Adjust weather date offsets in `tests/test_build_dataset.py` to match the lagged date constraints.
5. **Correct parameter signatures in `tests/test_context_data.py`**: Feed `raw_intakes`, `weather_daily`, and `requests_311` parameter names appropriately.
6. **Refine rolling feature index sorting logic**: Exclude records that share the same timestamp exactly to align with a strict `[t-window, t)` definition.
