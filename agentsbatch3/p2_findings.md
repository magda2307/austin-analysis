# P2 Findings - Schema Contract Assertions

**Date:** 2026-06-07  
**Issue:** P2 - tests/test_build_dataset.py (line 96) passes but does not assert full schema contract

## Analysis

The test `test_build_modeling_dataset_filters_and_creates_targets` in `tests/test_build_dataset.py:96` currently passes but lacks explicit assertions for:

1. **Optional outcome columns** - `outcome_subtype`, `sex_upon_outcome`, `age_upon_outcome`
2. **Alias columns** - `has_name`, `is_named`, `is_airport_location`
3. **Censoring columns** - `is_censored`, `censoring_reason`, `event_type`, `followup_days_censored`
4. **Target columns with aliases** - `adopted`, `is_adopted`, `target_adopted`, `length_of_stay`

## Required Changes

Add explicit assertions in `test_build_modeling_dataset_filters_and_creates_targets` to verify:
- All required columns present
- Column aliases work correctly
- Censoring metadata columns exist
- Event type values are correct (lowercase, accurate)
- Followup days match days_to_outcome

---

*Documented by: Agent Manager Subagent*  
*Date: 2026-06-07*
