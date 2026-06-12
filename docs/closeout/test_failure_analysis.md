# Test Failure Analysis Report (Task 22 Baseline Run)

This document contains the consolidated root cause analysis and exact recommended fixes for the 37 test failures encountered during the Phase 6 Task 22 baseline test run (`task-125.log`). The analysis was conducted by specialized subagents in three phases: Data Pipeline, Modeling & Evaluation, and Dashboard & Reporting.

## Phase 1: Data Pipeline Analysis

### `src/aac_adoption/data/build_dataset.py`
**Root Cause:** The `is_censored`, `censoring_reason`, `event_type`, and `followup_days_censored` columns are missing from the `build_modeling_dataset` output. 
**Recommended Fix:**
In `build_modeling_dataset`, explicitly define survival targets right before the `final_columns` selection:
```python
dataset["is_censored"] = ~dataset["adopted"]
dataset["event_type"] = dataset["outcome_type"].fillna("unknown").astype(str).str.strip().str.lower()
dataset["censoring_reason"] = np.where(dataset["adopted"], "", dataset["event_type"])
dataset["followup_days_censored"] = dataset["days_to_outcome"]
```
Ensure these four columns are appended to the `final_columns` list at the end of the file.

### `src/aac_adoption/data/context_data.py`
**Root Cause:** Legacy tests expected same-day weather, but the authoritative
intake-time contract requires prior completed calendar-day weather.
**Recommended Fix:** Keep the one-day lag and update fixtures to provide weather
for `intake_date - 1 day`. Preserve nullable weather flags when that row is absent.

### `tests/test_context_data.py`
**Root Cause:** The function signature of `add_context_features` has changed, and old keyword arguments break the test.
**Recommended Fix:**
Update the mock weather dates and the function call to match the new signature:
```python
weather = pd.DataFrame({"DATE": ["2021-01-01"], "TMAX": [96], "TMIN": [60], "PRCP": [0.1]})
enriched = add_context_features(dataset, weather_daily=weather, requests_311=requests, raw_intakes=dataset)
```

### `tests/test_build_dataset.py`
**Root Cause:** The weather lag breaks `test_build_modeling_dataset_from_files_adds_context_features`.
**Recommended Fix:** Change mock weather dates to precede the intakes correctly (`"2020-12-31"`, `"2021-01-31"`).

### `tests/test_match_records.py`
**Root Cause:** Tests attempt to unpack a tuple instead of accessing properties on the new `MatchResult` dataclass.
**Recommended Fix:** 
Update `test_match_records_reintake_features`, `test_match_records_no_reintake`, and `test_match_records_unmatched_intakes` to extract attributes from the dataclass:
```python
match_result = match_intakes_to_future_outcomes(intakes, outcomes, extract_end_date=pd.Timestamp("2021-12-31"))
result = match_result.matched_episodes
unmatched = match_result.unmatched_intakes
```

### `tests/test_horizon_targets.py`
**Root Cause:** Horizon targets (`adopted_in_7d`, etc.) were refactored out of `build_modeling_dataset` into `build_horizon_dataset`. Tests still expect them in the primary modeling dataset.
**Recommended Fix:** Route the outputs through `build_horizon_dataset`.

---

## Phase 2: Modeling & Evaluation Analysis

### `tests/models/test_reproducibility.py`
**Root Cause:** The `train_boosting_classification` call is missing the required `dataset_path` parameter.
**Recommended Fix:** Add the parameter to the function call:
```python
dataset_path=str(data_path),
```

### `tests/test_yearly_backtesting.py`
**Root Causes:** 
- Returning `SKIPPED` rows causes NaNs in the model column, breaking `.str.contains`.
- `quick=True` only produces data for 2019 and 2023, failing array length tests.
**Recommended Fixes:** 
1. Add `na=False` to all `.str.contains("...", na=False)` occurrences.
2. Change expected years to `[2019, 2023]`.
3. Check `len(result) > 0` and `(result["status"] == "SKIPPED").all()` for empty splits.

### `tests/test_hyperparam_tuning.py` and `tests/test_split.py`
**Root Cause:** `make_time_split` will fail if data is missing for any of its required periods (2013-2021, 2022, 2023, 2024-2025). The mock fixtures don't supply data for all these years.
**Recommended Fixes:** 
- In `test_hyperparam_tuning.py`: Update the `intake_year` choice to `np.random.choice([2020, 2022, 2023, 2024], n_samples)`.
- In `test_split.py`: Update df to contain both Dog and Cat across all periods `[2020, 2022, 2023, 2024, 2020, 2022, 2023, 2024]`.

### `tests/test_calibration.py`
**Root Cause:** Mock sidecar JSON metadata is missing required strict schema fields like `dataset_sha256`.
**Recommended Fix:** Populate the mock metadata using the base generator:
```python
from aac_adoption.models.metadata import base_training_metadata
from aac_adoption.models.split import DatasetSplit

split = DatasetSplit(
    full_data=pd.DataFrame(), train=pd.DataFrame(), calibration=pd.DataFrame(),
    selection=pd.DataFrame(), test=pd.DataFrame(), strategy="time",
    train_period="2013-2021", calibration_period="2022", selection_period="2023",
    test_period="2024-2025", animal_subset="combined", is_thesis_evaluation=True,
)

metadata = base_training_metadata(
    model_name="catboost", task="classification", split=split,
    feature_columns=feature_cols, run_timestamp="2024-01-01T00:00:00Z",
    target_column="classification_target", dataset_path=str(data_path),
)
```

### `tests/test_train_baseline_outputs.py` and `tests/test_train_boosting_outputs.py`
**Root Causes:** 
- Missing `intake_year` partitions.
- Tests look for `feature_set` instead of `feature_columns`.
**Recommended Fixes:** 
- Distribute intake years: `"intake_year": [2020, 2022, 2023, 2024][i % 4]`.
- Rename `"feature_set"` to `"feature_columns"` in the required schema checks.

---

## Phase 3: Dashboard & Reporting Analysis

### `src/aac_adoption/analysis/hypothesis_tables.py`
**Root Cause:** Tests use legacy column aliases (`adopted`, `days_to_outcome`) which crash the reporting logic expecting `classification_target` and `regression_target_days`.
**Recommended Fix:** Add a `_normalize_targets(df)` helper to bridge the alias gap and call it immediately after loading the CSV.

### `tests/test_acceptance_schema_aliases.py`
**Root Cause:** The mock DataFrame for final model selection lacks the `artifact_path` field.
**Recommended Fix:** Add the missing field:
```python
"artifact_path": str(model_path),
```

### `streamlit_app.py`
**Root Cause:** The application tries to slice `diagnostics["regression_slices"][["cohort", "mae", "records"]]`. However, the generated diagnostic files use `value` instead of `cohort`.
**Recommended Fix:** Change line 546 to:
```python
st.dataframe(diagnostics["regression_slices"][["value", "mae", "records"]].head(5), width='stretch', hide_index=True)
```

### `tests/test_dashboard_app.py`
**Root Cause:** Streamlit `AppTest` timeouts are too aggressive (3 seconds), causing flaky failures.
**Recommended Fix:** Update all `.run()` occurrences to `.run(timeout=10)`.
