# Batch 4 Focused Validator Test Results

**Date:** 2026-06-07  
**Agent:** Batch 4 Focused Validator

## Summary

| Test File | Tests Passed | Duration | Status |
|-----------|-------------|----------|--------|
| test_dashboard_data.py | 9/9 | 3.51s | PASS |
| test_ensemble.py | 14/14 | 3.79s | PASS |
| test_hyperparam_tuning.py | 4/4 | 32.87s | PASS |
| test_build_dataset.py | 5/5 | 1.26s | PASS |
| **Total** | **32/32** | 41.43s | **PASS** |

## Test Details

### test_dashboard_data.py (3.51s)
- test_best_model_rows_selects_expected_metrics ✓
- test_build_prediction_record_creates_model_features ✓
- test_build_profile_prediction_record_uses_representative_values ✓
- test_model_feature_columns_uses_artifact_metadata ✓
- test_similar_historical_cases_returns_outcome_mix ✓
- test_profile_global_shap_reasons_maps_profile_values ✓
- test_visibility_need_from_prediction_labels_quadrants ✓
- test_los_days_to_bucket ✓
- test_predict_from_record_handles_calibration ✓

**P1 Fix Verified:** Line 261 uses `math.log1p(15)` ✓

### test_ensemble.py (3.79s)
- test_weighted_ensemble_classifier ✓
- test_weighted_ensemble_from_dict ✓
- test_stacked_ensemble_classifier ✓
- test_weighted_ensemble_equal_weights ✓
- test_weighted_ensemble_regressor ✓
- test_stacked_ensemble_regressor ✓
- test_stacked_ensemble_oof_classifier ✓
- test_stacked_ensemble_oof_regressor ✓
- test_stacked_ensemble_classifier_oof ✓
- test_stacked_ensemble_regressor_oof ✓
- test_stacked_ensemble_classifier_fallback ✓
- test_stacked_ensemble_regressor_fallback ✓
- test_weighted_ensemble_classifier_string_labels ✓
- test_stacked_ensemble_classifier_string_labels ✓

**Warnings:** 5 (feature name warnings, non-blocking)

### test_hyperparam_tuning.py (32.87s ⚠️ over 30s)
- test_tune_histgradient_classification ✓
- test_tune_histgradient_regression ✓
- test_tune_empty_data ✓
- test_tune_models_runs_successfully ✓

**Note:** This test exceeded 30s threshold but was included for P2 fix validation.

### test_build_dataset.py (1.26s)
- test_build_modeling_dataset_filters_and_creates_targets ✓
- test_validate_rejects_negative_los ✓
- test_repeated_animal_matches_each_intake_to_next_unused_outcome ✓
- test_build_modeling_dataset_from_files_adds_context_features ✓
- test_build_modeling_dataset_keeps_raw_los_outliers ✓

## Code Fixes Verified

### P1: test_dashboard_data.py line 261
**Status:** ✓ PASSED
The test uses `math.log1p(15)` for proper log transformation.

### P2: data.py models_dir logic
**Status:** ✓ PASSED
Per-model family directories are preserved in the data.py models_dir logic.

## Conclusion

All 32 tests passed. Both Batch 4 fixes (P1 and P2) are verified working.
