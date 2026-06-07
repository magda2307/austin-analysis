# Batch 3 Communication Log

---

**Date:** 2026-06-07  
**Task:** P2 - Add Schema Contract Assertions  
**Status:** COMPLETE

---

## Summary

Added explicit schema contract assertions to test_build_dataset.py to verify full dataset output structure.

---

## Changes

### test_build_dataset.py (lines 96-121)

**New Assertions Added:**

```python
assert "outcome_subtype" in dataset.columns
assert "sex_upon_outcome" in dataset.columns
assert "age_upon_outcome" in dataset.columns
assert "has_name" in dataset.columns
assert "is_censored" in dataset.columns
assert "censoring_reason" in dataset.columns
assert "event_type" in dataset.columns
assert "followup_days_censored" in dataset.columns

assert bool(dataset.loc[dataset["animal_id"] == "A1", "is_censored"].item()) is False
assert dataset.loc[dataset["animal_id"] == "A1", "censoring_reason"].item() == ""
assert dataset.loc[dataset["animal_id"] == "A1", "event_type"].item() == "adoption"
assert abs(dataset.loc[dataset["animal_id"] == "A1", "followup_days_censored"].item() - dataset.loc[dataset["animal_id"] == "A1", "days_to_outcome"].item()) < 0.001
assert dataset.loc[dataset["animal_id"] == "A2", "event_type"].item() == "transfer"

assert dataset.loc[dataset["animal_id"] == "A1", "adopted"].item() is True
assert dataset.loc[dataset["animal_id"] == "A1", "is_adopted"].item() is True
assert dataset.loc[dataset["animal_id"] == "A1", "target_adopted"].item() == 1
assert dataset.loc[dataset["animal_id"] == "A2", "adopted"].item() is False
assert dataset.loc[dataset["animal_id"] == "A2", "is_adopted"].item() is False
assert dataset.loc[dataset["animal_id"] == "A2", "target_adopted"].item() == 0
assert abs(dataset.loc[dataset["animal_id"] == "A1", "length_of_stay"].item() - dataset.loc[dataset["animal_id"] == "A1", "days_to_outcome"].item()) < 0.001
assert abs(dataset.loc[dataset["animal_id"] == "A2", "length_of_stay"].item() - dataset.loc[dataset["animal_id"] == "A2", "days_to_outcome"].item()) < 0.001
```

---

## Test Results

```
tests/test_build_dataset.py::test_build_modeling_dataset_filters_and_creates_targets PASSED
tests/test_build_dataset.py::test_validate_rejects_negative_los PASSED
tests/test_build_dataset.py::test_repeated_animal_matches_each_intake_to_next_unused_outcome PASSED
tests/test_build_dataset.py::test_build_modeling_dataset_from_files_adds_context_features PASSED
tests/test_build_dataset.py::test_build_modeling_dataset_keeps_raw_los_outliers PASSED

5 passed in 0.97s
```

---

## Documentation

Files created:
- ✅ agentsbatch3/p2_findings.md - Analysis and test plan
- ✅ agentsbatch3/p2_work_complete.md - Summary of changes

---

## Notes

All schema contract assertions now verify:
- Required columns from REQUIRED_MODELING_COLUMNS
- Optional outcome metadata columns (outcome_subtype, sex_upon_outcome, age_upon_outcome)
- Alias columns (has_name, is_named)
- Censoring columns (is_censored, censoring_reason, event_type, followup_days_censored)
- Derived target columns (adopted, is_adopted, target_adopted, length_of_stay)
- Value correctness (event_type lowercase, followup_days_censored equals days_to_outcome)
