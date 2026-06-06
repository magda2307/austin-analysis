# ML Hardening Done Reference

Date: 2026-06-06

Purpose: old/reference file for completed or mostly completed work from `docs/ml_code_review.md`, `docs/internal/plan_0606.md`, and the previous cavecrew task list. Active work now lives in `docs/internal/cavecrew_ml_hardening_tasks.md`.

Status rule:

- DONE: code exists and at least targeted tests/contract checks exist.
- PARTIAL: code exists, but reproducible artifact, report refresh, or methodology proof is still missing.
- STALE: old review claim no longer matches code.

## DONE - Selected-Model Diagnostics

Source review concern:

- Diagnostics hardcoded CatBoost, so reports could explain/evaluate a non-selected model.

Current state:

- `src/aac_adoption/diagnostics/model_diagnostics.py` resolves selected rows from `reports/tables/final_model_selection.csv`.
- Diagnostics can load artifacts from `models/advanced`, `models/boosting`, and `models/baseline`.
- Outputs include `reports/diagnostics/diagnostics_model_selection.csv` and `reports/diagnostics/diagnostics_validation_tactics.csv`.
- SHAP skips with a written note when selected model is not CatBoost.

Evidence:

- `tests/test_diagnostics_outputs.py`
- `docs/internal/plan_0606.md`, Slice 1

Residual:

- Regenerate real report artifacts after code changes.

## DONE - Validation-Selected Thresholds

Source review concern:

- Thresholds needed validation selection and test-only evaluation.

Current state:

- `src/aac_adoption/analysis/threshold_analysis.py` selects thresholds on validation and applies frozen thresholds to test.
- Output records `threshold_selection_period`, `evaluation_period`, validation metrics, test metrics, and `validation_tactic`.
- Policies include default, max-F1,Youden J, high-recall, balanced, and top-10%-capacity.

Evidence:

- `tests/test_acceptance_schema_aliases.py`
- `docs/internal/plan_0606.md`, Slice 2

Residual:

- Regenerate `reports/tables/final_classifier_thresholds.csv` from current code.

## DONE - Calibration Metrics Foundation

Source review concern:

- Probability reliability needed Brier/ECE measurement.

Current state:

- `src/aac_adoption/models/evaluate.py` includes `expected_calibration_error()`.
- `classification_metrics()` reports Brier score and ECE when scores exist.
- Final model selection preserves calibration metric columns.

Evidence:

- `tests/test_diagnostics_outputs.py`
- `tests/test_train_advanced_outputs.py`
- `tests/test_train_boosting_outputs.py`
- `tests/test_train_baseline_outputs.py`
- `docs/internal/plan_0606.md`, Slice 3

Residual:

- Formal calibration stage still needs full methodological verification.

## PARTIAL - Formal Calibrated Artifacts

Old review claim:

- `calibrate_classifiers()` was missing and CLI import failed.

Current state:

- `src/aac_adoption/models/calibrate.py` now defines `calibrate_classifiers()`.
- `scripts/calibrate_classifiers.py` calls it.
- `PrefitProbabilityCalibrator`, Platt/sigmoid tests, and calibrated artifact saving exist.

Evidence:

- `tests/test_calibration.py`
- `tests/test_calibration_advanced.py`
- `src/aac_adoption/models/calibrate.py`

Verification on 2026-06-06:

- Cavecrew investigator confirmed CLI help exits 0.
- Platt maps to sigmoid.
- Prefit calibrator uses `base_model.predict_proba`; base model is not refit.
- Advanced CatBoost validation is split into early-stop and calibration halves.
- Uncalibrated and calibrated metric rows are separate.

Residual:

- Regenerate real calibrated artifacts/metrics if current report outputs are stale.

## DONE - Duplicate Feature Cleanup

Source review concern:

- Duplicate aliases inflated feature set and complicated interpretation.

Current state:

- Model feature registry excludes duplicate aliases: `age_months`, `age_years`, `has_name`, `intake_quarter`, `intake_season`, `color_group`.
- Compatibility aliases may remain in reports/dashboard outputs.

Evidence:

- `src/aac_adoption/features/feature_sets.py`
- `tests/test_feature_sets.py`
- `docs/internal/plan_0606.md`, Slice 4

## DONE - PR-AUC Primary Selection

Source review concern:

- Classification selection needed PR-AUC priority for imbalanced adoption task.

Current state:

- `src/aac_adoption/analysis/model_selection.py` sorts classification by PR-AUC first and ROC-AUC second.

Evidence:

- `tests/test_acceptance_schema_aliases.py`
- `docs/internal/plan_0606.md`, Slice 5

Residual:

- Regenerate generated Markdown summaries that still mention ROC-AUC-first wording.

## DONE - Recency Weight Formula and Wiring

Source review concern:

- Formula weighted older years more, and weights were not passed to learners.

Current state:

- `src/aac_adoption/models/split.py` gives later train years higher `sample_weight`.
- Baseline and boosting pipelines pass `model__sample_weight`.
- CatBoost training passes `sample_weight`.

Evidence:

- `src/aac_adoption/models/split.py`
- `src/aac_adoption/models/train_baseline.py`
- `src/aac_adoption/models/train_boosting.py`
- `src/aac_adoption/models/train_advanced.py`
- README recent improvements mention corrected recency weights.

Residual:

- Strategy comparison is still open: full history vs 5-year vs 3-year vs weighted.

## DONE - Tuning Fold Leakage Fix

Source review concern:

- Tuning used last fold only and fit HGB preprocessing outside CV.

Current state:

- `src/aac_adoption/models/tune.py` uses `TimeSeriesSplit`.
- Objectives average across folds.
- HGB preprocessing is fit inside each fold.

Evidence:

- `src/aac_adoption/models/tune.py`
- README recent improvements mention leakage-free tuning.

Residual:

- Add/confirm tiny fixture test for regression frame/index alignment.

## DONE - LOS Target Leakage Verification

Source review concern:

- `length_of_stay`, `days_to_outcome`, and `regression_target_days` might be capped with full-dataset quantiles before split.

Current state:

- `src/aac_adoption/data/build_dataset.py` keeps LOS targets as raw aliases from `days_to_outcome`.
- No caller was found using `winsorize_outliers()` on LOS targets.
- Advanced regression log transforms happen on train/validation/test split frames, not in pre-split dataset build.

Evidence:

- Cavecrew investigator scan on 2026-06-06.
- `src/aac_adoption/data/build_dataset.py`
- `src/aac_adoption/models/train_advanced.py`
- `src/aac_adoption/models/train_adopted_regression.py`

Residual:

- If future target capping is added, fit caps on train only and store cap metadata.

## DONE - Permutation Importance Uses Validation First

Source review concern:

- Permutation importance used final test split, weakening untouched-test claim.

Current state:

- `src/aac_adoption/models/train_boosting.py` uses validation when present and records `importance_split`.
- Falls back to test only when validation is empty.

Evidence:

- `src/aac_adoption/models/train_boosting.py`
- `tests/test_train_boosting_outputs.py`

## PARTIAL - Horizon Targets and Follow-Up Safeguard

Source review concern:

- Horizon labels needed enough follow-up time.

Current state:

- `src/aac_adoption/data/build_dataset.py` creates `followup_days_available`.
- `adopted_in_7d`, `adopted_in_30d`, `adopted_in_60d`, and `adopted_in_90d` become `NaN` when unsafe.
- Fast observed adoptions remain valid even with short remaining follow-up.
- `followup_days_available` is preserved in the output modeling dataset.
- `scripts/generate_data_audit.py` writes `reports/tables/horizon_followup_audit.csv`.
- `reports/summary/data_audit.md` gets a Horizon Follow-Up Audit section.

Evidence:

- `tests/test_horizon_targets.py`
- README recent improvements mention horizon censoring.
- Targeted verification on 2026-06-06: `python -m pytest tests/test_horizon_targets.py -q` passed.

Residual:

- Run data audit on real files to refresh checked-in/generated artifacts.

## DONE - Leakage Risk Classification Foundation

Source review concern:

- Leakage audit needed risk levels, not shallow true/false checks.

Current state:

- `scripts/generate_leakage_audit.py` emits leakage/risk fields.
- Tests check risk levels for leakage columns.

Evidence:

- `tests/test_leakage_audit.py`
- `tests/test_acceptance_schema_aliases.py`

Residual:

- Confirm generated audit artifact is refreshed from current code.

## DONE - Artifact Path Metadata Injection

Source review concern:

- Sidecar JSON could miss own artifact path.

Current state:

- `src/aac_adoption/models/artifacts.py` injects `artifact_path` before writing JSON.
- Sidecar metadata also includes package versions and placeholder git SHA.

Evidence:

- `src/aac_adoption/models/artifacts.py`
- `tests/test_train_advanced_outputs.py`

Residual:

- Add real git SHA/data hash/run manifest if thesis requires stronger provenance.

## DONE - Task A: Refresh Censoring Summary Artifacts

Source review concern:

- Horizon follow-up audit code exists, but real reports may not be refreshed.
- Roadmap still marks end-of-dataset censoring/follow-up summary incomplete until artifact is regenerated.

Current state:

- `generate_data_audit.py` successfully executed and generated `reports/tables/horizon_followup_audit.csv`.
- `reports/summary/data_audit.md` includes Horizon Follow-Up Audit section.

## DONE - Task B: Audit Re-Intake Ambiguity

Source review concern:

- Matching records re-intake metadata, but roadmap says ambiguous intake-outcome pairs are not rejected/summarized.

Current state:

- `match_records.py` is updated to detect and flag `is_ambiguous_match` using a performant mapping algorithm.
- Data audit script correctly calculates and reports ambiguous episodes.

## DONE - Task C: Add Yearly Backtesting Artifact

Source review concern:

- `evaluate_backtesting.py` existed, but roadmap said yearly backtesting table was still TODO.

Current state:

- `evaluate_backtesting.py` outputs the exact required schema (train_period, test_year, subset, model, pr_auc, roc_auc, brier, ece, mae).
- Test suite verifies the output schema.
- Hooked into `scripts/run_full_pipeline.py`.

## DONE - Task D: Compare Recency Strategies

Source review concern:

- Recency weights were fixed and wired, but no strategy comparison existed.

Current state:

- `compare_recency.py` compares full-history, recent 5-year, recent 3-year, and recency-weighted training.
- Strategy results saved to `reports/tables/recency_strategy_comparison.csv`.
- Test asserts monotonic chronological weights constraints.

## DONE - Task E: Align Dashboard LOS Language

Source review concern:

- Dashboard could conflate generic LOS with days to adoption.

Current state:

- `streamlit_app.py` avoids exact-day certainty using `wait_bucket` in the sensitivity demo.
- Cleaned up terms: "Predicted wait" to "Predicted days to outcome", "Median wait" to "Median days to outcome".
- Updated methodology to correctly reference "Predicted days to adoption".

## STILL OPEN - Review Items Not Yet Accepted

Active task file tracks these:

- LOS target leakage verification.
- Yearly temporal backtesting.
- Recency strategy comparison.
- Dashboard/report language separation for LOS vs adoption timing.

## Stale Note Removed

Previous `docs/internal/cavecrew_ml_hardening_tasks.md` ended with a corrupt UTF-16-looking line saying all tasks completed on 2026-06-06. That claim is not true under the current roadmap acceptance rules. Completed work is listed here; remaining work is in the active task file.
