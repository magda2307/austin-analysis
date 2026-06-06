# ML Code Review

Date: 2026-06-06

Scope: full current repository review from an ML engineering standpoint, covering data construction, feature engineering, splitting, model training, tuning, calibration, interpretation, artifacts, tests, dashboard consistency, and reproducibility.

## Executive Summary

This project already has a strong thesis-oriented ML shape: intake-time feature registry, chronological split, multiple model families, artifact-first reporting, calibration/diagnostic concepts, SHAP evidence, leakage audit scripts, and broad test coverage.

Main gap: several implemented safeguards are either not wired into training or are weakened by validation/test reuse. Before relying on final thesis claims, fix the high-priority issues around target censoring, full-dataset target winsorization, calibration design, tuning leakage, sample weights, and the broken calibration pipeline step.

## Strengths

- Clear artifact-first architecture: scripts create CSV/Markdown/figure/model outputs and the Streamlit app reads artifacts instead of retraining.
- Explicit intake-time feature registry in `src/aac_adoption/features/feature_sets.py`, with `validate_no_leakage()` preventing known outcome-derived fields from becoming predictors.
- Chronological thesis split is the default (`2013-2021`, `2022-2023`, `2024-2025`), which is much better than random evaluation for shelter data with drift.
- Model ladder is methodologically useful: dummy, linear, random forest, histogram boosting, and CatBoost.
- Metrics include PR-AUC, ROC-AUC, Brier score, ECE, confusion matrix counts, MAE/RMSE/median AE/R2, and some bootstrap confidence intervals.
- Tests cover many artifact contracts and pipeline outputs, which is a good base for a thesis codebase.

## High-Priority Findings

### 1. Target winsorization leaks test-set distribution into training

Location: `src/aac_adoption/data/build_dataset.py:101-108`

`length_of_stay`, `days_to_outcome`, and `regression_target_days` are winsorized before the train/validation/test split. The 1st/99th percentiles are calculated on the full matched dataset, so future test-period LOS distribution can affect training labels.

Impact: regression metrics and downstream LOS interpretations can be mildly optimistic or distribution-conditioned on future data.

Fix: move target capping into the split/training layer. Fit caps on train only, store cap values in metadata, then apply the frozen caps to validation/test. Alternatively avoid target winsorization and use robust losses/metrics.

### 2. Right-censoring is acknowledged but not handled

Locations:

- `src/aac_adoption/data/match_records.py` around future-outcome matching
- `src/aac_adoption/models/split.py:66-68`
- `src/aac_adoption/models/train_advanced.py:188-213`

Unmatched intakes are excluded from the modeling frame. Recent/open stays are therefore missing, especially near the 2024-2025 test tail. A `censoring_flag` is created when `days_to_outcome >= max_los_days`, but it is not used by filters, weights, metrics, or model objectives.

Impact: classification and LOS estimates can be biased toward animals with observed outcomes. Long-stay/open cases are exactly the cases that matter for operational adoption support.

Fix: add a censor date and one of these strategies:

- For classification horizons, define observed-safe labels only when enough follow-up exists.
- For LOS, use survival modeling or censored regression, not plain regression on observed outcomes only.
- Exclude unsafe recent tail windows from final evaluation when follow-up is incomplete.
- Report attrition by intake year and animal subset.

### 3. Calibration pipeline is currently inconsistent and partly broken

Locations:

- `scripts/calibrate_classifiers.py:7`
- `scripts/run_full_pipeline.py:90-91`
- `src/aac_adoption/models/calibrate.py:20-28`
- `src/aac_adoption/models/calibrate.py:64-69`
- `src/aac_adoption/models/train_advanced.py:76`
- `src/aac_adoption/models/train_advanced.py:147-171`

Observed command:

```powershell
python scripts\calibrate_classifiers.py --help
```

fails with:

```text
ImportError: cannot import name 'calibrate_classifiers' from 'aac_adoption.models.calibrate'
```

Additional calibration issues:

- `calibrate_with_isotonic()` clones a fitted model and refits only on calibration data, so original training is ignored.
- `method == "sigmoid"` is forcibly changed to isotonic, so Platt calibration is not actually Platt.
- `post_hoc_calibration_pipeline()` fits with `cv=5` on `train+val`, so validation is no longer a clean calibration holdout.
- CatBoost uses validation for early stopping and then reuses the same validation set for calibration.
- Advanced classification rows mix uncalibrated accuracy/F1/ROC/PR with calibrated Brier/ECE.

Impact: probability reliability claims and final pipeline reproducibility are not trustworthy until this is repaired.

Fix: choose one clean design:

- Train base model on train.
- Use an early-stopping fold distinct from calibration, or disable early stopping when validation must calibrate.
- Fit calibrator on calibration split only using `FrozenEstimator` / `cv="prefit"`.
- Evaluate one coherent calibrated row and one coherent uncalibrated row, never mixed metrics.
- Restore or remove `scripts/calibrate_classifiers.py` from the full pipeline.

### 4. Recency weighting is inverted and not passed to model fitting

Locations:

- `src/aac_adoption/models/split.py:77-78`
- `src/aac_adoption/models/train_boosting.py:94`
- `src/aac_adoption/models/train_baseline.py` fit calls
- `src/aac_adoption/models/train_advanced.py` CatBoost fit calls

The current formula gives 2013 rows weight 1.5 and 2021 rows weight 1.0:

```python
1.0 + 0.5 * (2021 - x.year) / (2021 - 2013)
```

Also, the resulting `sample_weight` column is not passed into `Pipeline.fit()` or CatBoost.

Impact: the feature is currently either no-op or backwards from its name. If wired later without fixing the formula, older data receives more influence despite the intended drift handling.

Fix: either remove the claim/column, or pass weights explicitly and flip the formula so recent rows are larger. Add a test asserting 2021 weight > 2013 weight.

### 5. Hyperparameter tuning leaks and can misalign regression labels

Location: `src/aac_adoption/models/tune.py:45-174`

Issues:

- Comment says CV, but the objective uses only `splits[-1]`, making tuning a single holdout.
- Regression labels come from `reg_train_df`, while feature matrices/fold indices come from classification `train_df` / `cat_X`. If target filtering ever differs, labels and features misalign.
- Histogram boosting preprocessor is fit on full tuning train before the time split (`preprocessor.fit_transform(hist_X)`), leaking validation-fold imputation/category information into training.

Impact: tuned params may be unstable or overfit to a fold, and regression tuning can silently become invalid.

Fix: build separate classification and regression feature matrices and folds. Use a `Pipeline` inside each fold so preprocessors fit only on that fold's training window. Average objective across all chronological folds, or explicitly rename it holdout tuning.

### 6. Interpretation uses the final test split for permutation importance

Location: `src/aac_adoption/models/train_boosting.py:117-124`

Permutation importance samples from `split.test`.

Impact: the test set becomes both final evaluation and interpretation source. This is not target leakage into training, but it weakens the "untouched final test" claim.

Fix: compute permutation importance on validation by default. Keep test only for final performance reporting.

### 7. Stacked ensemble trains meta-learner on in-sample base predictions

Location: `src/aac_adoption/models/ensemble.py:67-72` and `src/aac_adoption/models/ensemble.py:97-102`

The base estimators are fit on `X, y`, then the meta-estimator is trained on predictions for that same `X`.

Impact: severe stacking overfit if this ensemble is used for reported results.

Fix: use out-of-fold predictions for the meta-learner. Also preserve DataFrame input; `check_array(X)` strips column names and can break pipelines/CatBoost models that expect named columns.

## Medium-Priority Findings

### 8. Intake-volume context is computed from matched modeling rows

Location: `src/aac_adoption/data/context_data.py:211-222`

`intake_volume_7d` and `intake_volume_30d` are based on the modeling dataset after matching/filtering, not the raw intake ledger.

Impact: context volume undercounts operational intake load, especially where unmatched/open intakes are removed.

Fix: compute shelter volume from all cleaned intakes before outcome matching. Join the prior-window values into matched modeling rows.

### 9. Time split can create empty validation while still reporting `strategy="time"`

Location: `src/aac_adoption/models/split.py:42-84`

`_has_time_split()` checks train and test years, but not 2022-2023 validation years. CatBoost/calibration code expects validation when doing early stopping/calibration.

Impact: some datasets can claim a time split with no validation support, causing inconsistent downstream behavior.

Fix: require non-empty train, validation, and test when validation is semantically required, or label the strategy as a no-validation chronological split.

### 10. Random fallback can mix future and past

Location: `src/aac_adoption/models/split.py:100-113`

When default thesis years are unavailable, code falls back to random split.

Impact: smaller/filtered datasets can leak temporal structure into evaluation.

Fix: if `intake_datetime` or `intake_year` exists, prefer chronological quantile fallback over random fallback. Use random only for tiny synthetic tests.

### 11. Advanced regression docstring says adopted-only, but code trains all rows

Location: `src/aac_adoption/models/train_advanced.py:188-207`

`filter_df` and `filter_col` are created but unused. The docstring says "adopted-only filtering", while the model uses all outcome types.

Impact: thesis wording can drift into "time to adoption" when the target is time to any matched outcome.

Fix: either implement an adopted-only model with explicit target `days_to_adoption`, or update the docstring and all UI/report wording to "time to outcome".

### 12. Dashboard model selection can disagree with final model selection

Locations:

- `src/aac_adoption/dashboard/data.py:115-135`
- `src/aac_adoption/dashboard/data.py:263-285`

Dashboard best-model rows rank classification by ROC-AUC, while the project documentation increasingly treats PR-AUC as primary for classification. Sensitivity prediction hardcodes `models/advanced` CatBoost rather than resolving final selected/calibrated artifacts.

Impact: dashboard can present a different "best" model than the thesis evidence pack.

Fix: read `reports/tables/final_model_selection.csv` and use the selected artifact path/metric. Prefer calibrated classifier for probability demos when available.

### 13. Artifact metadata is incomplete for some model files

Locations:

- `src/aac_adoption/models/artifacts.py:34-36`
- `src/aac_adoption/models/train_advanced.py:158-172`

The standard sidecar JSON is written before callers add `artifact_path`, so the sidecar can lack its own artifact path. Calibrated artifacts are saved with `joblib.dump()` and no standard sidecar metadata.

Impact: auditability and reproducibility are weaker for generated models.

Fix: let `save_model_artifact()` inject `artifact_path` before writing JSON. Add `save_calibrated_artifact()` with metadata: base model, calibration method, train/calibration/test periods, metrics source, feature list, package versions.

## Test and CI Gaps

- `tests/test_calibration.py` checks output length but misses the Platt-to-isotonic method bug.
- `tests/models/test_reproducibility.py` skips when local `data/processed/modeling_dataset.csv` is absent, so CI may not protect model semantics.
- Manifest/artifact tests skip when generated artifacts are absent; clean clones can pass without proving full artifact generation.
- No test currently asserts sample weights are passed into model fitting.
- No test catches full-dataset target winsorization.
- No test catches `scripts/calibrate_classifiers.py --help` import failure.

Recommended tests:

- Tiny fixture dataset with 2019-2025 rows for deterministic train/val/test checks.
- Test that target preprocessing caps are fitted on train only.
- Test `python scripts/calibrate_classifiers.py --help` exits 0.
- Test Platt calibration keeps `method="sigmoid"`.
- Test dashboard best-model selection follows final selection table.
- Test OOF stacking once ensemble is used.

## Reproducibility and Operations Gaps

- `pyproject.toml` uses broad `>=` production dependencies. The lock file exists, but reproducible install depends on humans choosing it.
- `.gitignore` ignores baseline/boosting/advanced model dirs, but not `models/calibrated/` or `tmp_models/`; generated `.joblib` artifacts are currently tracked.
- README states Docker/DVC/MLflow are not implemented. For a thesis, full MLflow may be optional, but a run manifest should be mandatory.

Recommended minimum:

- Add a `make`/script path that installs from `requirements-lock.txt`.
- Add CI smoke commands for dataset fixture -> train tiny model -> generate core artifacts.
- Track run metadata: git SHA, data file hash, feature list hash, package versions, train/val/test row counts, model params, calibration params.
- Ignore generated calibrated/tmp model artifacts or move them into a deliberate artifact store.

## Suggested Remediation Order

1. Fix broken calibration CLI and full-pipeline step.
2. Clean calibration design: separate early stopping, calibration, and final test metrics.
3. Move LOS winsorization out of dataset build; fit target caps on train only or remove capping.
4. Decide censoring strategy for open/recent intakes and horizon labels.
5. Fix recency weights and pass them to all applicable learners, or remove them.
6. Fix tuning fold/preprocessing leakage and regression label alignment.
7. Move permutation importance from test to validation.
8. Align dashboard selected-model logic with final selection artifacts.
9. Add tiny versioned fixtures and CI smoke tests.
10. Add artifact metadata/run manifest and tighten generated artifact ignores.

## Bottom Line

The project is not missing basic ML structure; it has unusually much of it already. What is missing is final methodological hardening: clean separation of train/validation/calibration/test roles, real handling of censoring, wired sample weights, non-leaky tuning/target transforms, and reproducible end-to-end calibration. Fix those and the codebase becomes much more defensible as a thesis ML pipeline.
