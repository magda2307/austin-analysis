# Plan 0606 - ML Thesis Strengthening Roadmap

This plan turns the 06.06 critique into an implementation roadmap for making the AAC adoption project methodologically stronger. Main goal: move from "many artifacts and models" to a tighter ML thesis built around valid targets, time-aware evaluation, calibrated probabilities, selected-model-consistent diagnostics, and honest uncertainty.

## Critical Before Thesis Submission

1. Fix selected-model diagnostics.
   - Update diagnostics generation so it reads `reports/tables/final_model_selection.csv`.
   - For each `subset + task`, load the selected model artifact before producing calibration curves, error slices, risk quadrants, red flags, and explanations.
   - Ensure diagnostics never describe CatBoost when HistGradientBoosting or another model was selected.

2. Make calibration a real modeling stage.
   - Train base classifiers on `2013-2021`.
   - Calibrate on `2022-2023`.
   - Evaluate final calibrated artifacts on `2024-2025`.
   - Save calibrated CatBoost and calibrated HistGradientBoosting classifier artifacts.
   - Report before/after ROC-AUC, PR-AUC, Brier score, Expected Calibration Error, and calibration gaps by species, age group, and breed group.

3. Add validation-selected thresholds.
   - Stop treating `0.5` as the main decision threshold.
   - Select thresholds on validation data only.
   - Save threshold choices for max F1, Youden J, target recall, target precision, and top-capacity use cases such as top 10% risk.
   - Apply selected thresholds unchanged to test data and report precision, recall, F1, and confusion matrices.

4. Clean duplicate feature representations.
   - Remove redundant model features such as repeated age columns, duplicate name flags, and overlapping color aliases.
   - Keep one modeling representation for each concept:
     - `age_days`
     - `age_group`
     - `is_named`
     - `primary_color`
     - `simplified_color_group`
     - `primary_breed`
     - `simplified_breed_group`
   - Keep aliases only where needed for reports or dashboard compatibility.
   - Document that redundant representation columns were excluded to avoid unstable interpretation and inflated importance.

5. Improve length-of-stay modeling.
   - Stop presenting one generic LOS regressor as adoption speed.
   - Add `length_to_any_outcome` model for all clean matched episodes.
   - Add adopted-only `days_to_adoption` model for adopted animals only.
   - Train duration regressors on `log1p(days)`.
   - Convert predictions back with `expm1`, clamp negative outputs to zero, and report MAE/RMSE on original days.

6. Add censoring safeguards near dataset end.
   - For horizon tasks, include only intakes with enough follow-up time before export date.
   - Example: for 90-day adoption analysis, include only intakes at least 90 days before export date.
   - For LOS evaluation, exclude final-period intakes when long-stay unresolved animals would be under-observed.
   - Document remaining censoring risk explicitly.

7. Audit episode matching ambiguity.
   - Check whether another intake for the same animal occurs between candidate intake and outcome.
   - If yes, mark the earlier intake as ambiguous, censored, or unmatched, then prefer matching the later intake.
   - Report clean matches, ambiguous matches, dropped matches, and percentage affected.

8. Expand leakage audit.
   - Flag obvious leakage fields such as outcome, outcome datetime, sex/age upon outcome, days to outcome, length of stay, days to adoption, adopted, target columns, and future/after/next columns.
   - Classify borderline predictors as `safe`, `probably_safe`, `needs_audit`, or `unsafe`.
   - Pay special attention to name flags, context features, and any fields that may have been cleaned or updated after intake.

## Strong ML Upgrades

1. Add horizon-based adoption classifiers.
   - Create targets for adoption within 7, 30, 60, and 90 days.
   - Train and evaluate horizon classifiers separately from eventual-adoption models.
   - Use horizon outputs in dashboard/reporting as operational intervention windows.

2. Add yearly temporal backtesting.
   - Evaluate rolling historical training windows:
     - train `2013-2018`, test `2019`
     - train `2013-2019`, test `2020`
     - train `2013-2020`, test `2021`
     - train `2013-2021`, test `2022`
     - train `2013-2022`, test `2023`
     - train `2013-2023`, test `2024`
   - Show whether performance changes after COVID or operational shifts.

3. Compare recency strategies.
   - Compare full-history training with recent 5-year, recent 3-year, and recency-weighted training.
   - Use 2024-2025 as final evidence for whether recent windows improve current performance.

4. Add survival analysis section.
   - Add Kaplan-Meier curves for adoption timing.
   - Add either Cox proportional hazards or AFT model if feasible.
   - Frame survival analysis as the appropriate censored time-to-event complement to regression.

5. Strengthen categorical handling.
   - For CatBoost, preserve raw categorical columns and pass explicit `cat_features`.
   - For HistGradientBoosting, use native categorical support where feasible, such as `categorical_features="from_dtype"` or a categorical mask.
   - Keep both `primary_breed` and `simplified_breed_group` for comparison instead of simplifying too early.

6. Add controlled hyperparameter tuning.
   - Tune CatBoost depth, learning rate, iterations, `l2_leaf_reg`, class weights, subsample, and random strength.
   - Tune HistGradientBoosting max iterations, learning rate, max leaf nodes, minimum samples per leaf, L2 regularization, and categorical handling.
   - Tune on validation only, never on test.
   - Save `best_params.csv`, `tuning_results.csv`, and `selected_model_reason.md`.

7. Make PR-AUC primary for classification.
   - Use PR-AUC as primary classification ranking metric.
   - Use ROC-AUC as secondary.
   - Use precision, recall, F1, and confusion matrices only at validation-selected thresholds.
   - Add lift at top 10%, lift at top 20%, precision@top_k, and recall@top_k.

8. Add uncertainty for duration outputs.
   - Avoid presenting exact single-day predictions as certain wait times.
   - Prefer median, P75, and P90 predicted days, or buckets:
     - `0-7 days`
     - `8-30 days`
     - `31-60 days`
     - `61-90 days`
     - `90+ days`
   - Phrase dashboard outputs as historical similarity/risk patterns, not deterministic forecasts.

9. Improve subgroup reliability.
   - Report subgroup metrics for dogs, cats, age group, intake type, intake condition, simplified breed group, simplified color group, COVID period, and year.
   - For each subgroup, report records, adoption rate, mean predicted probability, calibration gap, PR-AUC where enough data, and Brier score.
   - Do not interpret subgroup metrics below minimum sample thresholds such as `n < 100` or `n < 200`.

10. Add cluster-aware confidence intervals.
    - Bootstrap by `animal_id` where possible.
    - If row-level bootstrap remains, document that records are episode-level and not fully independent animal-level observations.

## Nice To Have

1. Add LightGBM with native categorical support if time remains.
2. Add XGBoost AFT survival objective only if survival modeling becomes a core thesis section.
3. Add quantile regression for P50/P80/P90 wait-time estimates.
4. Add feature drift or population stability index by year.
5. Add dashboard/reporting views for top-k campaign evaluation.
6. Do not add neural networks unless there is a separate, defensible research reason.

## Methodology Risks To Document

1. Adoption likelihood, adoption within a horizon, days to adoption, and days to any outcome are different targets.
2. Length of stay is non-negative, right-skewed, censored, outlier-heavy, and outcome-dependent.
3. Dataset-end censoring can make recent animals look like they had shorter waits.
4. Duplicate features can distort SHAP values and feature importance.
5. Calibration is required before calling model scores probabilities.
6. A chronological split is necessary but not enough; yearly backtesting gives stronger evidence.
7. Subgroup results need minimum sample-size rules.
8. Episode-level records are not always independent when the same animal appears multiple times.

## Validation Tactics By Part

1. Selected-model diagnostics.
   - Validate by writing a diagnostics model-selection table with `task`, `subset`, selected `model_name`, and loaded `artifact_path`.
   - Add a regression test proving diagnostics load HistGradientBoosting when `final_model_selection.csv` selects HistGradientBoosting.

2. Calibration stage.
   - Validate by comparing uncalibrated and calibrated test-period ROC-AUC, PR-AUC, Brier score, and ECE.
   - Add a table that records train, calibration, and test periods for every calibrated artifact.

3. Validation-selected thresholds.
   - Validate by saving threshold source period, chosen threshold, validation metric, and untouched test metric.
   - Test that no threshold is selected from test-period labels.

4. Duplicate feature cleanup.
   - Validate by exporting final feature lists and asserting duplicate aliases are absent from model-training features.
   - Keep report/dashboard alias checks separate from model feature checks.

5. LOS and adopted-only timing models.
   - Validate by saving separate task metadata for all-outcome LOS and adopted-only days-to-adoption.
   - Test that adopted-only models train only on adopted episodes and report metrics after inverse log transform.

6. Censoring safeguards.
   - Validate by writing included/excluded row counts per horizon and export-date cutoff.
   - Test that 30/60/90-day targets exclude intakes without enough follow-up time.

7. Episode matching ambiguity audit.
   - Validate by reporting clean, ambiguous, censored, unmatched, and dropped episode counts.
   - Add fixture tests for repeated intake before outcome.

8. Leakage audit.
   - Validate by classifying each suspicious column as `safe`, `probably_safe`, `needs_audit`, or `unsafe`.
   - Test that known outcome-derived fields fail model feature validation.

9. Horizon classifiers.
   - Validate by checking each horizon target has enough follow-up and separate metrics.
   - Compare horizon PR-AUC, lift, and calibration by horizon.

10. Yearly temporal backtesting.
    - Validate by writing one row per train-window/test-year pair.
    - Test that each test year is strictly after its training window.

11. Recency strategies.
    - Validate by comparing full-history, 5-year, 3-year, and recency-weighted models on the same final test period.
    - Record selected strategy and reason.

12. Survival analysis.
    - Validate by saving event/censoring counts and Kaplan-Meier strata counts.
    - Check survival analysis uses censoring instead of dropping unresolved episodes silently.

13. Categorical handling.
    - Validate by saving categorical feature lists in model metadata.
    - Test CatBoost receives raw categorical columns and sklearn pipelines receive expected encoded inputs.

14. Hyperparameter tuning.
    - Validate by saving full tuning results, best params, validation metric, and untouched test metric.
    - Test that tuning result rows do not use test-period scores for selection.

15. PR-AUC and ranking metrics.
    - Validate by making PR-AUC primary in model-selection tables and including top-k/lift outputs.
    - Test that PR-AUC exists for every classifier with probability scores.

16. Duration uncertainty.
    - Validate by saving wait-time buckets or quantile predictions plus coverage checks where available.
    - Avoid exact-day dashboard language unless uncertainty is shown.

17. Subgroup reliability.
    - Validate by saving subgroup sample sizes, calibration gaps, Brier score, and interpretation flags.
    - Test that small subgroups are flagged and excluded from strong interpretation.

18. Cluster-aware confidence intervals.
    - Validate by bootstrapping on `animal_id` and recording cluster count.
    - If row bootstrap is used, require an explicit limitation note.

## Acceptance Checklist

- `generate_diagnostics.py` uses selected model artifacts from `final_model_selection.csv`.
- Calibrated classifier artifacts exist and are evaluated on untouched test years.
- Thresholds are selected on validation data and applied to test data.
- Duplicate modeling features are removed from training feature lists.
- LOS and adopted-only timing models are separate.
- Horizon adoption targets exist for 7, 30, 60, and 90 days.
- End-of-dataset censoring rules are applied and documented.
- Re-intake matching ambiguity is audited and summarized.
- Yearly backtesting table exists.
- Survival analysis section exists or is explicitly documented as future work.
- Subgroup reliability includes calibration and sample-size safeguards.
- Leakage audit classifies suspicious features by risk level.

## Implementation Progress

### Slice 1 - Selected-Model Diagnostics

Status: implemented.

Changes:
- Diagnostics now resolve the selected model from `reports/tables/final_model_selection.csv` per `task + subset`.
- Diagnostics can load selected artifacts across `models/advanced`, `models/boosting`, and `models/baseline`.
- Diagnostics write `reports/diagnostics/diagnostics_model_selection.csv` so the loaded artifact can be audited.
- Diagnostics write `reports/diagnostics/diagnostics_validation_tactics.csv` so each generated diagnostic has an explicit validation tactic.
- SHAP generation now skips with a written note when the selected model is not CatBoost, preventing explanations for an unselected CatBoost artifact.

Harsh validation:
- Added a unit test proving diagnostics load `hist_gradient_boosting` when final selection chooses it.
- Added a unit test proving validation tactics cover generated diagnostic parts.
- Ran `python -m pytest tests/test_diagnostics_outputs.py -q`: passed.
- Ran `python -m pytest tests/test_diagnostics_outputs.py tests/test_acceptance_schema_aliases.py tests/test_artifacts.py -q`: passed.
- Ran `python -m compileall -q src/aac_adoption/diagnostics/model_diagnostics.py`: passed.

Residual risk:
- Full diagnostics generation was not rerun on the real dataset in this slice, so newly written CSV artifacts are validated by unit contract, not by a fresh end-to-end reports refresh.

### Slice 2 - Validation-Selected Thresholds

Status: implemented.

Changes:
- Threshold analysis now locates the selected classifier from `final_model_selection.csv` and keeps selected `model_name`, `animal_subset`, and `model_path` in the threshold artifact.
- Thresholds are selected on the validation split only, then applied unchanged to the test split.
- `final_classifier_thresholds.csv` now includes `threshold_selection_period=validation`, `evaluation_period=test`, validation metrics, test metrics, and a `validation_tactic` column.
- Added `youden_j` and `top_10_percent_capacity` threshold policies alongside default, max-F1, high-recall, and balanced precision/recall thresholds.
- Threshold scoring now handles selected CatBoost models with CatBoost preprocessing and sklearn pipelines with normal feature frames.
- Threshold summary text now states that operating thresholds are validation-selected and test-applied.

Harsh validation:
- Added a unit test proving threshold tables expose validation-source metadata and new policies.
- Added a unit test proving frozen validation-selected thresholds are evaluated separately on test scores.
- Added a temp end-to-end test that creates a fake selected classifier artifact, runs threshold analysis, and verifies the output CSV separates validation selection from test evaluation.
- Delegated harsh review found four issues: string selected flags, ignored explicit artifact paths, empty validation split behavior, and diagnostic regression/test misalignment.
- Fixed all four delegated-review findings.
- Added a unit test proving threshold model discovery accepts string `selected=true` and explicit `artifact_path`.
- Ran `python -m pytest tests/test_acceptance_schema_aliases.py -q`: passed.
- Ran `python -m pytest tests/test_acceptance_schema_aliases.py tests/test_diagnostics_outputs.py tests/test_artifacts.py -q`: passed.
- Ran `python -m pytest tests/test_acceptance_schema_aliases.py tests/test_diagnostics_outputs.py tests/test_artifacts.py tests/test_train_advanced_outputs.py tests/test_train_boosting_outputs.py tests/test_train_baseline_outputs.py -q`: passed.
- Ran `python -m compileall -q src/aac_adoption/analysis/threshold_analysis.py src/aac_adoption/diagnostics/model_diagnostics.py`: passed.

Residual risk:
- Real report artifacts were not regenerated in this slice. The implementation is contract-tested with temporary artifacts, but the checked-in `reports/tables/final_classifier_thresholds.csv` must be refreshed by the analysis pipeline.

### Slice 3 - Calibration Metrics Foundation

Status: implemented.

Changes:
- Added `expected_calibration_error()` to central model evaluation helpers.
- `classification_metrics()` now reports `brier_score` and `expected_calibration_error` whenever probability scores are available.
- Final model-selection outputs now preserve `brier_score` and `expected_calibration_error` columns when model comparison tables contain them.

Harsh validation:
- Added tests proving classifier metrics include PR-AUC, Brier score, and ECE.
- Added a test proving ECE is zero for perfectly calibrated fixed bins.
- Added a model-selection schema test proving calibration metric columns survive into `final_model_selection.csv`.
- Ran `python -m pytest tests/test_diagnostics_outputs.py tests/test_train_advanced_outputs.py tests/test_train_boosting_outputs.py tests/test_train_baseline_outputs.py -q`: passed.
- Ran `python -m pytest tests/test_diagnostics_outputs.py tests/test_acceptance_schema_aliases.py tests/test_train_advanced_outputs.py tests/test_train_boosting_outputs.py tests/test_train_baseline_outputs.py -q`: passed.
- Ran `python -m compileall -q src/aac_adoption/models/evaluate.py src/aac_adoption/analysis/model_selection.py`: passed.

Residual risk:
- This slice adds calibration measurement, not post-hoc calibrated classifier artifacts. Isotonic/Platt artifact generation remains a later slice.

### Slice 4 - Duplicate Feature Cleanup

Status: implemented.

Changes:
- Model feature set now keeps one primary representation for age (`age_days`, `age_group`) and excludes linear aliases `age_months` and `age_years`.
- Model feature set now keeps `is_named` and excludes duplicate `has_name`.
- Model feature set excludes redundant calendar aliases `intake_quarter` and `intake_season`.
- Methodology notes now document that duplicate aliases are kept only for reports/dashboard compatibility, not model training.

Harsh validation:
- Added a feature-set test proving duplicate aliases are absent from `INTAKE_TIME_FEATURES`.
- Updated feature-set tests to expect `age_days`, not `age_years`, as the numeric age modeling feature.
- Ran `python -m pytest tests/test_feature_sets.py tests/test_train_advanced_outputs.py tests/test_train_boosting_outputs.py tests/test_train_baseline_outputs.py -q`: passed.
- Ran `python -m compileall -q src/aac_adoption/features/feature_sets.py`: passed.

Residual risk:
- Historical docs and generated thesis guide files may still mention old aliases in explanatory sections. Core methodology notes and model feature tests now enforce the cleaned modeling set.

### Slice 5 - PR-AUC Primary Selection

Status: implemented.

Changes:
- Classification model selection now sorts by PR-AUC first and ROC-AUC second.
- Classification selection reason now frames PR-AUC as the primary metric for class-imbalanced adoption prediction.
- Final selection still preserves ROC-AUC, F1, Brier score, and ECE for secondary interpretation.

Harsh validation:
- Added a test where one model has higher ROC-AUC but another has higher PR-AUC; final selection correctly picks the higher PR-AUC model.
- Ran `python -m pytest tests/test_acceptance_schema_aliases.py -q`: passed.
- Ran `python -m compileall -q src/aac_adoption/analysis/model_selection.py`: passed.

Residual risk:
- Existing generated Markdown reports may still contain older ROC-first wording until report generation is rerun.
