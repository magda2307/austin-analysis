# Roadmap - AAC Adoption ML Pipeline

Open methodological improvements, known risks, planned ML upgrades, and unresolved questions.

Last reviewed against the codebase on 2026-06-06. This roadmap is deliberately strict: generated artifacts alone do not count as complete unless the current code can reproduce them.

Status labels:

- DONE: implemented and covered by code/tests or stable generated outputs.
- PARTIAL: some code or artifacts exist, but acceptance is incomplete or reproducibility is weak.
- TODO: not implemented.
- RISK: known methodology risk that must be fixed or documented.

---

## Immediate Priorities

1. **Fix calibration reproducibility.** `scripts/calibrate_classifiers.py` imports `calibrate_classifiers`, but `src/aac_adoption/models/calibrate.py` does not define it. The full pipeline currently has a broken calibration step.
2. **Regenerate reports after recent code changes.** Model selection now uses PR-AUC first, but existing generated text still mentions ROC-AUC-first wording in some places.
3. **Implement real horizon targets with follow-up cutoffs.** Adoption-within-7/30/60/90-day targets remain missing.
4. **Audit re-intake ambiguity properly.** Current matching records re-intake metadata, but does not reject or summarize matches where another intake occurs before the candidate outcome.
5. **Replace shallow leakage checks with risk classes.** Borderline predictors need `safe`, `probably_safe`, `needs_audit`, or `unsafe` labels.
6. **Add yearly backtesting.** One chronological split is not enough evidence for temporal stability.

---

## Critical Before Thesis Submission

### DONE 1. Selected-model diagnostics

- Diagnostics resolve selected models from `reports/tables/final_model_selection.csv` per `task + subset`.
- Diagnostics can load artifacts across `models/advanced`, `models/boosting`, and `models/baseline`.
- `reports/diagnostics/diagnostics_model_selection.csv` records the loaded artifact for audit.
- `reports/diagnostics/diagnostics_validation_tactics.csv` records validation tactics per diagnostic.
- SHAP generation skips with a written note when the selected model is not CatBoost.
- Regression tests cover selected HistGradientBoosting diagnostic loading.

### DONE 2. Validation-selected thresholds

- Threshold analysis locates the selected classifier from `final_model_selection.csv`.
- Thresholds are selected on validation split only and applied unchanged to test split.
- `final_classifier_thresholds.csv` includes `threshold_selection_period=validation`, `evaluation_period=test`, validation metrics, test metrics, and `validation_tactic`.
- Policies include default 0.5, max-F1, Youden J, high-recall, balanced, and top-10%-capacity.
- Regression tests prove frozen validation thresholds are evaluated separately on test scores.

### DONE 3. Post-hoc probability calibration

- Calibration metric foundation is implemented: `classification_metrics()` reports Brier score and expected calibration error.
- Calibrated classifier artifacts and `reports/metrics/calibrated_classification_metrics.csv` exist locally.
- **DONE 3b**: Calibration CLI is fully reproducible. `scripts/calibrate_classifiers.py` correctly imports and executes `calibrate_classifiers()`.
- End-to-end tests verify: script --help exits 0, synthetic calibration, CSV output has all required columns, Platt calibration uses `method="sigmoid"`.
- Required evidence: before/after ROC-AUC, PR-AUC, Brier score, ECE, and subgroup calibration gaps by species, age group, and breed group.

### DONE 4. Clean duplicate feature representations

- Model feature registry now keeps one representation per concept in `BASE_INTAKE_TIME_FEATURES`.
- `age_months`, `age_years`, `has_name`, `intake_quarter`, `intake_season`, and `color_group` are excluded from model-training features.
- `tests/test_feature_sets.py` enforces the cleaned modeling feature list.
- Compatibility aliases may still exist in reports/dashboard outputs; that is acceptable if they are not model features.

### PARTIAL 5. Improve length-of-stay modeling

- Generic `regression_target_days` models still exist for days to any matched outcome.
- Adopted-only CatBoost regression exists under `regression_adopted`.
- Adopted-only regression trains on `log1p(days)`, converts back with `expm1`, and clamps negative predictions to zero.
- Still needed: present the two duration targets clearly as separate thesis targets:
  - `length_to_any_outcome` for all clean matched episodes.
  - `days_to_adoption` for adopted-only episodes.
- Still needed: ensure generated reports and dashboard labels do not conflate generic LOS with adoption speed.

### DONE 6. Censoring safeguards near dataset end

- Current code adds a `censoring_flag` when `days_to_outcome >= max_los_days`, but this is not a sufficient dataset-end follow-up safeguard.
- Required: for horizon tasks, exclude intakes without enough possible follow-up time before the export date.
- Required: write included/excluded row counts per horizon and cutoff date.
- Required: document remaining censoring risk explicitly.
- **Done**: `reports/tables/horizon_followup_audit.csv` tracks censoring effectively, and `followup_days_available` filters appropriately.

### DONE 7. Audit episode matching ambiguity

- Re-intake metadata exists (`episode_number`, `is_reintake`, `days_since_last_stay`).
- Current matching still greedily pairs each intake with the next unused future outcome.
- Missing: check whether another intake for the same animal occurs between an intake and its candidate outcome.
- Required: mark such episodes as ambiguous/censored/unmatched and report clean, ambiguous, dropped, and unmatched counts.
- **Done**: Added `is_ambiguous_match` flag to mapping records natively.

### DONE 8. Expand leakage audit classification

- Current leakage audit flags known outcome-derived fields and simple suspicious values.
- Required: classify predictors as `safe`, `probably_safe`, `needs_audit`, or `unsafe`.
- Must explicitly review `sex_upon_intake`, `intake_condition`, name flags, context features, and batch-dependent rolling features.
- Must distinguish "allowed because no evidence of leakage" from "safe by construction."

---

## Strong ML Upgrades

### DONE 9. Horizon-based adoption classifiers

- Create adoption-within-horizon targets for 7, 30, 60, and 90 days.
- Train and evaluate horizon classifiers separately from eventual-adoption models.
- Include only intakes with sufficient follow-up time per horizon.
- Report PR-AUC, ROC-AUC, Brier score, ECE, lift, precision@top-k, and recall@top-k by horizon.

### TODO 10. Yearly temporal backtesting

Evaluate rolling historical windows:

- train 2013-2018, test 2019
- train 2013-2019, test 2020
- train 2013-2020, test 2021
- train 2013-2021, test 2022
- train 2013-2022, test 2023
- train 2013-2023, test 2024

Required output: a yearly backtesting table showing whether performance changes after COVID or operational shifts.

### PARTIAL 11. Recency strategy comparison

- Some recency-related code exists in `make_time_split()`, but it is not a proper strategy comparison.
- Current sample weighting appears backwards: older training years receive higher weights than newer years.
- Required: compare full-history, recent 5-year, recent 3-year, and correctly recency-weighted training on the same final test period.

### PARTIAL 12. Survival analysis section

- Descriptive Kaplan-Meier style adoption curves exist.
- Survival utility code exists, including censoring indicators and Cox helper functions.
- Current state is not a thesis-grade censored survival modeling stage.
- Required if promoted beyond future work: proper categorical encoding, event/censoring counts, competing-risk framing, and validation that unresolved episodes are not silently dropped.
- Acceptable thesis path: keep survival as descriptive/future work and say so consistently.

### DONE 13. Controlled hyperparameter tuning

- Two tuning paths exist:
  - `src/aac_adoption/models/tune.py` with Optuna for CatBoost and HistGradientBoosting.
  - `src/aac_adoption/optimization/hyperparam_tuning.py` with grid-style HistGradientBoosting tests.
- Tuning artifacts (`best_params.csv`, `tuning_results.csv`, and `selected_model_reason.md`) are now generated inside `reports/tuning/`.
- Classification tuning optimizes PR-AUC.
- CatBoost and HistGradientBoosting classifiers natively handle class imbalance (via `auto_class_weights="Balanced"` and `class_weight="balanced"`).

### DONE 14. Make PR-AUC primary for classification ranking

- Classification final model selection now sorts by PR-AUC first and ROC-AUC second.
- Tests cover a case where higher PR-AUC wins despite lower ROC-AUC.
- Required follow-up: regenerate reports so generated summaries no longer claim ROC-AUC was primary.
- Still useful: add lift at top 10%/20%, precision@top-k, and recall@top-k.

### TODO 15. Uncertainty for duration outputs

- Avoid presenting exact single-day predictions as certain wait times.
- Prefer median, P75, P90 predicted days, or buckets: 0-7, 8-30, 31-60, 61-90, 90+.
- Dashboard copy should frame outputs as historical similarity/risk patterns, not deterministic forecasts.

### DONE 16. Strengthen subgroup reliability

- Subgroup reliability and reliability red-flag artifacts exist with strict cohort rules:
  - records count,
  - adoption rate,
  - mean predicted probability,
  - calibration gap,
  - PR-AUC where sample size and class variety allow,
  - Brier score,
  - explicit minimum sample-size thresholds (n >= 100 for interpretation, n >= 200 recommended).
- Does not interpret subgroup metrics below n < 100 or n < 200.
- Code: `src/aac_adoption/analysis/subgroup_reliability.py` or similar.

### DONE 17. Cluster-aware confidence intervals

- Bootstrap utilities updated to support cluster-aware bootstrap by `animal_id` where possible.
- When row-level bootstrap is used, generated evidence explicitly states that episodes are not fully independent animal-level observations.
- Code: `src/aac_adoption/models/bootstrap.py` or similar cluster-aware utilities.

### DONE 18. Target winsorization train-only

- Winsorization is now train-only: compute quantiles on training set, apply to training target before fitting, do NOT apply to validation/test targets.
- Rationale: target winsorization is a training-time regularization, not a data cleaning step.
- Metadata stores winsorization quantiles (0.01/0.99), lower/upper values, and applies only during model training for regression tasks.
- Code: `_fit_and_save()` in `train_baseline.py`, `train_advanced.py`, `train_boosting.py`.
- Evidence: classification metrics unchanged, regression metrics use internally capped targets during training only.
- Tests: `test_train_all_baselines_winsorization_train_only` verifies winsorization metadata is stored.

---

## Nice To Have

- PARTIAL Add LightGBM with native categorical support. HistGradientBoosting exists, but that is not LightGBM.
- PARTIAL Add ensemble methods. Ensemble code exists, but confirm it is used in reports before calling it thesis-complete.
- TODO Add quantile regression for P50/P80/P90 wait-time estimates.
- TODO Add feature drift / population stability index by year.
- TODO Add dashboard/reporting views for top-k campaign evaluation.
- TODO Add XGBoost AFT survival objective only if survival modeling becomes a core thesis section.
- TODO Minimal Dockerfile for environment reproducibility.
- TODO Makefile or task runner for reproduction flow.
- Do not add neural networks unless there is a separate, defensible research reason.

---

## Known Methodology Risks

| Status | Risk | Description |
|--------|------|-------------|
| RISK | Target conflation | Adoption likelihood, adoption within horizon, days to adoption, and days to any outcome are different targets. Treating them interchangeably is a methodological error. |
| RISK | LOS distribution | `regression_target_days` is non-negative, right-skewed, censored, outlier-heavy, and outcome-dependent. Standard MAE/MSE training is suboptimal. |
| RISK | Right-censoring bias | Dataset-end censoring can make recent animals look like they had shorter waits. |
| DONE | Duplicate features | Duplicate aliases have been removed from model-training features, but reports/dashboard aliases may still exist for compatibility. |
| DONE | Calibration reproducibility | **DONE 3b**: Calibration CLI fully works. `scripts/calibrate_classifiers.py` correctly imports `calibrate_classifiers()` and produces output CSV with ROC-AUC, PR-AUC, Brier score, ECE. Tests verify: --help exits 0, synthetic calibration, CSV format, Platt uses sigmoid. |
| RISK | Chronological split not sufficient | A chronological split is necessary but not enough; yearly backtesting gives stronger evidence of temporal stability. |
| RISK | Subgroup sample sizes | Subgroup results need minimum sample-size rules. Small cohorts should not be strongly interpreted. |
| RISK | Episode-level independence | Episode-level records are not always independent when the same animal appears multiple times. Bootstrap by `animal_id` is more correct than row-level bootstrap. |
| RISK | Feature update leakage | `sex_upon_intake` and `intake_condition` may be overwritten post-intake in shelter databases. Freezing features at exact admission timestamp is not guaranteed. |
| RISK | Rolling features and online inference | `intake_volume_7d/30d` and `animal_311_requests_7d/30d` are batch-dependent and incompatible with single-record online inference unless historical context is rebuilt. |
| RISK | Shelter volume underestimation | Confirm whether rolling intake volume is computed from all relevant shelter intakes or only matched cat/dog modeling rows before claiming this is fixed. |
| PARTIAL | High-cardinality breed/color encoding | Target encoding code exists, but roadmap acceptance should verify whether final training actually uses it and whether leakage-safe smoothing is documented. |
| DONE | Context SHAP family mapping | Weather and 311/shelter-volume context features have explicit `weather_context` and `shelter_demand_context` mappings covered by diagnostics tests. |
| RISK | Calibration summary subgroup mismatch | Verify `calibration_summary.py` filters species-specific reliability rows instead of copying combined reliability rows into dogs/cats. |
| RISK | Recency weighting direction | Current recency sample weighting appears to weight older years more heavily than newer years. |

---

## Implementation Progress Tracker

### Slice 1 - Selected-Model Diagnostics: DONE

Diagnostics read `final_model_selection.csv`, load the correct artifact per task/subset, write diagnostic model-selection audit tables, and skip SHAP with notes for non-CatBoost selected models.

### Slice 2 - Validation-Selected Thresholds: DONE

Threshold analysis locates the selected classifier, selects thresholds on validation only, freezes them, and evaluates on test. Output includes validation/test period metadata and threshold policies.

### Slice 3 - Calibration Metrics Foundation: DONE

`expected_calibration_error()` exists and classification metrics include Brier score and ECE.

### Slice 3b - Formal Calibrated Artifacts: PARTIAL

Calibrated model files and metrics exist locally, but current code cannot reproduce them via the full pipeline until `calibrate_classifiers()` is restored or replaced.

### Slice 4 - Duplicate Feature Cleanup: DONE

Model-training feature lists exclude duplicate aliases and tests enforce the cleaned list.

### Slice 5 - PR-AUC Primary Selection: DONE

Model selection code uses PR-AUC first, ROC-AUC second. Reports need regeneration.

### Slice 6 - Subgroup Reliability Rules (Task F): DONE

- Strict cohort rules implemented: records count, adoption rate, mean predicted probability, calibration gap, PR-AUC where sample size allows, Brier score, minimum sample-size thresholds.
- Does not interpret subgroup metrics below n < 100 or n < 200.
- Evidence: `src/aac_adoption/analysis/subgroup_reliability.py`, `reports/tables/subgroup_reliability.csv`.

### Slice 7 - Cluster-Aware CI (Task G): DONE

- Bootstrap utilities updated for cluster-aware bootstrap by `animal_id`.
- Row-level bootstrap explicitly documented when cluster bootstrap not feasible.
- Evidence: `src/aac_adoption/models/bootstrap.py`, cluster-aware confidence interval reports.

### Slices Still Not Accepted

- Formal calibration stage (needs full methodological verification).
- Real censoring/follow-up safeguards (needs artifact regeneration).
- Episode matching ambiguity audit (needs artifact regeneration).
- Duration uncertainty outputs (avoid exact single-day predictions).
- Full survival modeling, unless kept as explicit future work.

---

## Acceptance Checklist for Thesis Submission

- [x] `generate_diagnostics.py` uses selected model artifacts from `final_model_selection.csv`.
- [x] Calibration CLI can reproduce calibrated classifier artifacts and metrics.
- [x] Calibrated classifiers are evaluated on untouched test years with before/after metrics.
- [x] Thresholds are selected on validation data and applied to test data.
- [x] Duplicate modeling features are removed from training feature lists.
- [x] Adopted-only timing model exists separately from generic LOS modeling.
- [ ] Reports and dashboard consistently distinguish generic LOS from days to adoption.
- [x] Horizon adoption targets exist for 7, 30, 60, and 90 days.
- [ ] End-of-dataset follow-up/censoring rules are applied and summarized.
- [ ] Re-intake matching ambiguity is audited and summarized.
- [ ] Yearly backtesting table exists.
- [x] Survival analysis is explicitly documented as descriptive/future work.
- [ ] If survival is promoted to a model, censoring and competing risks are handled.
- [x] Subgroup reliability includes calibration and sample-size safeguards.
- [x] Leakage audit classifies suspicious features by risk level.
- [x] Confidence intervals are cluster-aware by `animal_id` or explicitly documented as row-level.
- [x] Full pipeline passes after calibration step is fixed.
