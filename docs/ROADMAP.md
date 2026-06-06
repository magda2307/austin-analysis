# Roadmap — AAC Adoption ML Pipeline

Open methodological improvements, known risks, planned ML upgrades, and unresolved questions. Sourced from `plan_0606.md` (06-06 critique) and the `progress_and_future_work.md` review (2026-05-31).

Items marked ✅ are implemented. Items marked 🔲 are pending. Items marked ⚠️ are known risks that need documentation even if not immediately fixed.

---

## Critical Before Thesis Submission

### ✅ 1. Selected-model diagnostics
- Diagnostics now resolve the selected model from `reports/tables/final_model_selection.csv` per `task + subset`.
- Diagnostics can load artifacts across `models/advanced`, `models/boosting`, and `models/baseline`.
- `reports/diagnostics/diagnostics_model_selection.csv` records the loaded artifact for audit.
- `reports/diagnostics/diagnostics_validation_tactics.csv` records validation tactics per diagnostic.
- SHAP generation skips with a written note when the selected model is not CatBoost.
- Regression test: proves diagnostics load `hist_gradient_boosting` when final selection chooses it.

### ✅ 2. Validation-selected thresholds
- Threshold analysis locates the selected classifier from `final_model_selection.csv`.
- Thresholds selected on validation split only, applied unchanged to test split.
- `final_classifier_thresholds.csv` includes `threshold_selection_period=validation`, `evaluation_period=test`, validation metrics, test metrics, and `validation_tactic` column.
- Threshold policies: default (0.5), max-F1, Youden J, high-recall, balanced, top-10%-capacity.
- Regression test: proves frozen validation-selected thresholds are evaluated separately on test scores.

### 🔲 3. Post-hoc probability calibration (formal modeling stage)
- **Current state:** calibration gaps up to 0.21 for some subgroups (e.g. Terrier-type dogs). Probabilities are not yet formally calibrated.
- **Plan:** train base classifiers on 2013–2021, fit calibration (Isotonic Regression or Platt Scaling) on 2022–2023, evaluate calibrated artifacts on 2024–2025 test.
- **Metrics to report:** before/after ROC-AUC, PR-AUC, Brier score, Expected Calibration Error (ECE), and calibration gaps by species, age group, and breed group.
- **Artifacts:** save calibrated CatBoost and calibrated HistGradientBoosting classifier `.joblib` files.

### 🔲 4. Clean duplicate feature representations
- **Current state:** `age_days`, `age_months`, `age_years` are collinear; `has_name` and `is_named` are near-identical.
- **Plan:** keep one modeling representation per concept (`age_days`, `age_group`, `is_named`, `primary_color`, `simplified_color_group`, `primary_breed`, `simplified_breed_group`); keep aliases only where needed for reports or dashboard compatibility.
- **Validation:** export final feature lists and assert duplicate aliases are absent from model-training features (separate from report/dashboard alias checks).

### 🔲 5. Improve length-of-stay modeling
- **Current state:** one generic LOS regressor is presented for all outcomes.
- **Plan:**
  - Add `length_to_any_outcome` model for all clean matched episodes.
  - Add adopted-only `days_to_adoption` model for adopted animals only.
  - Train duration regressors on `log1p(days)`, convert back with `expm1`, clamp negative outputs to zero.
  - Report MAE/RMSE on original days.

### 🔲 6. Censoring safeguards near dataset end
- **Current state:** right-censoring bias exists at temporal split boundaries. Animals admitted near end of 2025 with short or no-yet-observed stays are either omitted or systematically show shorter LOS.
- **Plan:** for horizon tasks, include only intakes with sufficient follow-up time (e.g. for 90-day analysis, exclude intakes less than 90 days before export date). Document remaining censoring risk explicitly.

### 🔲 7. Audit episode matching ambiguity
- **Current state:** greedy matching does not verify whether a re-intake occurs between a matched intake and its outcome. This can artificially inflate LOS for the first intake.
- **Plan:** check whether another intake for the same animal occurs between candidate intake and matched outcome. If yes, mark earlier intake as ambiguous/censored/unmatched. Report clean/ambiguous/dropped counts.

### 🔲 8. Expand leakage audit classification
- **Current state:** leakage audit flags known outcome-derived fields.
- **Plan:** classify borderline predictors as `safe`, `probably_safe`, `needs_audit`, or `unsafe`. Pay special attention to `sex_upon_intake` and `intake_condition` (may be updated post-intake in shelter databases), name flags, and context features.

---

## Strong ML Upgrades (Planned)

### 🔲 9. Horizon-based adoption classifiers
- Create targets for adoption within 7, 30, 60, and 90 days.
- Train and evaluate horizon classifiers separately from eventual-adoption models.
- Use horizon outputs in dashboard/reporting as operational intervention windows.
- Include only intakes with sufficient follow-up time per horizon.

### 🔲 10. Yearly temporal backtesting
- Evaluate rolling historical training windows:
  - train 2013–2018, test 2019
  - train 2013–2019, test 2020
  - train 2013–2020, test 2021
  - train 2013–2021, test 2022
  - train 2013–2022, test 2023
  - train 2013–2023, test 2024
- Show whether performance changes after COVID or operational shifts.

### 🔲 11. Recency strategy comparison
- Compare full-history, recent 5-year, recent 3-year, and recency-weighted training.
- Use 2024–2025 as final evidence for whether recent windows improve current performance.

### 🔲 12. Survival analysis section
- Add Kaplan–Meier curves (descriptive, partially done) plus Cox proportional hazards or AFT model.
- Frame survival analysis as the appropriate censored time-to-event complement to regression.
- Required: use censoring instead of silently dropping unresolved episodes.

### 🔲 13. Controlled hyperparameter tuning
- CatBoost: depth, learning rate, iterations, `l2_leaf_reg`, class weights, subsample, random strength.
- HistGradientBoosting: max iterations, learning rate, max leaf nodes, minimum samples per leaf, L2 regularization, categorical handling.
- Tune on validation only, never on test.
- Save `best_params.csv`, `tuning_results.csv`, and `selected_model_reason.md`.

### 🔲 14. Make PR-AUC primary for classification ranking
- Use PR-AUC as primary classification ranking metric.
- Use ROC-AUC as secondary.
- Add lift at top 10%/20%, precision@top_k, recall@top_k.

### 🔲 15. Uncertainty for duration outputs
- Avoid presenting exact single-day predictions as certain wait times.
- Prefer median, P75, P90 predicted days, or buckets (0–7, 8–30, 31–60, 61–90, 90+ days).
- Phrase dashboard outputs as historical similarity/risk patterns, not deterministic forecasts.

### 🔲 16. Strengthen subgroup reliability
- Report per-cohort: records, adoption rate, mean predicted probability, calibration gap, PR-AUC (where n ≥ threshold), Brier score.
- Do not interpret subgroup metrics below minimum sample thresholds (n < 100 or n < 200).

### 🔲 17. Cluster-aware confidence intervals
- Bootstrap by `animal_id` where possible.
- If row-level bootstrap remains, document that records are episode-level and not fully independent animal-level observations.

---

## Nice To Have

- 🔲 Add LightGBM with native categorical support.
- 🔲 Add quantile regression for P50/P80/P90 wait-time estimates.
- 🔲 Add feature drift / population stability index by year.
- 🔲 Add dashboard/reporting views for top-k campaign evaluation.
- 🔲 Add XGBoost AFT survival objective (only if survival modeling becomes a core thesis section).
- 🔲 Minimal Dockerfile for environment reproducibility.
- 🔲 Makefile or task runner for the reproduction flow.
- Do **not** add neural networks unless there is a separate, defensible research reason.

---

## Known Methodology Risks

These are documented here so they can be addressed in the thesis discussion chapter even if the code is not changed.

| Risk | Description |
|------|-------------|
| ⚠️ Target conflation | Adoption likelihood, adoption within horizon, days to adoption, and days to any outcome are different targets. Treating them interchangeably is a methodological error. |
| ⚠️ LOS distribution | `regression_target_days` is non-negative, right-skewed, censored, outlier-heavy, and outcome-dependent. Standard MAE/MSE training is suboptimal. |
| ⚠️ Right-censoring bias | Dataset-end censoring can make recent animals look like they had shorter waits. |
| ⚠️ Duplicate features | `age_days/months/years` and `has_name/is_named` distort SHAP values and feature importance. |
| ⚠️ Uncalibrated probabilities | Calibration gaps up to 0.21 mean model scores cannot yet be treated as literal adoption probabilities. |
| ⚠️ Chronological split not sufficient | A chronological split is necessary but not enough; yearly backtesting gives stronger evidence of temporal stability. |
| ⚠️ Subgroup sample sizes | Subgroup results need minimum sample-size rules. Small cohorts (n < 100) should not be strongly interpreted. |
| ⚠️ Episode-level independence | Episode-level records are not always independent when the same animal appears multiple times. Bootstrap by `animal_id` is more correct than row-level bootstrap. |
| ⚠️ Feature update leakage | `sex_upon_intake` and `intake_condition` may be overwritten post-intake in shelter databases. Freezing features at exact admission timestamp is not yet guaranteed. |
| ⚠️ Rolling features and online inference | `intake_volume_7d/30d` and `animal_311_requests_7d/30d` are batch-dependent and incompatible with single-record online inference. |
| ⚠️ Shelter volume underestimation | Rolling intake volume is computed after cat/dog filtering and matching. Unmatched intakes and other animal types are excluded, understating actual shelter crowding. |
| ⚠️ High-cardinality breed/color encoding | `simplified_breed_group` collapses very distinct breeds into "other". Target encoding with smoothing or text embeddings would be more principled. |
| ⚠️ Context SHAP family mapping | Weather and 311 context features fall into `"other"` in SHAP family maps. Explicit `weather` and `shelter_demand` categories should be added. |
| ⚠️ Calibration summary subgroup mismatch | `calibration_summary.py` may copy `combined` reliability table for dogs/cats separately instead of filtering by species. |

---

## Implementation Progress Tracker (plan_0606 slices)

### Slice 1 — Selected-Model Diagnostics ✅

Changes completed: diagnostics read `final_model_selection.csv`, load correct artifact per task+subset, write `diagnostics_model_selection.csv` and `diagnostics_validation_tactics.csv`, SHAP skips for non-CatBoost selected models.

Residual risk: full diagnostics not rerun on real dataset in this slice; validated by unit contract only.

### Slice 2 — Validation-Selected Thresholds ✅

Changes completed: threshold analysis locates selected classifier, thresholds selected on validation only, applied to test, `final_classifier_thresholds.csv` includes period metadata and new policies (`youden_j`, `top_10_percent_capacity`).

Residual risk: checked-in `reports/tables/final_classifier_thresholds.csv` must be refreshed by running `scripts/run_analysis.py` on real data.

### Slices 3–18 — Remaining items from plan_0606 🔲

All remaining slices (calibration stage, feature cleanup, LOS modeling, censoring safeguards, episode matching audit, leakage audit expansion, horizon classifiers, temporal backtesting, recency strategies, survival analysis, categorical handling, hyperparameter tuning, PR-AUC primacy, duration uncertainty, subgroup reliability, cluster-aware CIs) are **not yet implemented**. See sections above for plans.

---

## Acceptance Checklist for Thesis Submission

- [ ] `generate_diagnostics.py` uses selected model artifacts from `final_model_selection.csv` ✅
- [ ] Calibrated classifier artifacts exist and are evaluated on untouched test years
- [ ] Thresholds are selected on validation data and applied to test data ✅
- [ ] Duplicate modeling features are removed from training feature lists
- [ ] LOS and adopted-only timing models are separate
- [ ] Horizon adoption targets exist for 7, 30, 60, and 90 days
- [ ] End-of-dataset censoring rules are applied and documented
- [ ] Re-intake matching ambiguity is audited and summarized
- [ ] Yearly backtesting table exists
- [ ] Survival analysis section exists or is explicitly documented as future work
- [ ] Subgroup reliability includes calibration and sample-size safeguards
- [ ] Leakage audit classifies suspicious features by risk level
