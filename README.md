# AAC Adoption ML Pipeline

Code foundation for the thesis project:

**Life-Saving Data: Analyzing Factors Affecting Adoptions at the Austin Animal Center via Machine Learning and Visualization**

## Scope and Claim

This thesis builds a **reproducible, leakage-safe, interpretable supervised machine learning pipeline** for analyzing adoption likelihood and length-of-stay patterns in Austin Animal Center dog and cat records using **intake-time features only**.

**What this project claims:**
- Intake-time predictors (breed, color, age, intake type, covid period, etc.) are **associated with** adoption likelihood and length-of-stay patterns in AAC records.
- The ML models demonstrate **predictive association**, not causal effects.
- Findings are **descriptive evidence** for the AAC dataset (2013-2025) and do not generalize beyond AAC without replication.
- The Streamlit dashboard is a **thesis demo and presentation layer**. It is not the main scientific contribution. The scientific contribution is the pipeline, the feature study, and the reproducible evidence pack.

**What this project does NOT claim:**
- That intake features *cause* adoption outcomes.
- That COVID *caused* adoption rates to change.
- That dark-colored animals are *discriminated against* (framed as descriptive association only).
- That the regression output is *adoption speed* - it is length of stay until any matched outcome.

**Required terminology:**
- Use: `predictive association`, `associated with`, `linked to model output`, `intake-time predictors`, `length of stay`, `time to outcome`, `descriptive time-to-adoption evidence`.
- Avoid: `causes adoption`, `proves`, `adoption speed` (unless subset is adopted animals only), `days to adoption` (unless explicitly filtered to adopted episodes).

See `docs/target_definitions.md` for formal target variable definitions.
See `docs/METHODOLOGY.md` for regression and causal-framing justifications.

This repository focuses first on a clean, reproducible data and ML pipeline for Austin Animal Center dog/cat adoption analysis.

## Current Status

Implemented:

- raw data downloader from Austin Open Data,
- reproducible modeling dataset builder,
- intake-time-only feature set,
- deterministic Found Location taxonomy and flags,
- optional intake-time context features from weather, 311 demand, and shelter volume,
- time-aware train/validation/test split,
- baseline models,
- histogram gradient boosting models,
- CatBoost advanced models,
- dog/cat/combined evaluation,
- model artifacts,
- EDA tables and figures,
- model comparison tables,
- H1/H2/H3/H4/H5 support tables,
- hypothesis evidence matrix and interpretation reports,
- first-level and SHAP-based interpretability outputs,
- calibration, threshold, residual, and risk diagnostics,
- SHAP and feature-family summaries,
- animal-centered profile research,
- model evidence pack with confidence intervals and cohort limitations,
- local explanation example artifacts that connect Animal Journey Cards, similar historical cases, and model reasons,
- subgroup reliability and descriptive time-to-adoption evidence,
- data audit, target-definition audit, leakage audit, matching examples, feature-quality audit, and environment snapshot artifacts,
- artifact manifest for thesis deliverables, including generator-declared local explanation artifacts,
- Streamlit thesis dashboard,
- formal report-generation scripts,
- full-pipeline runners for Python, PowerShell, and shell workflows.

Not implemented yet:

- Docker/DVC/MLflow,
- survival models beyond descriptive adoption-timeline views.

### Recent Pipeline Improvements

- **Recency Weights and Drift Mitigation**: The baseline model temporal weighting formula was corrected to emphasize recent years (e.g., 2021) over historical years (2013). This directly influenced model dynamics, improving the deterministic baseline ROC-AUC from `0.63158` to `0.66029`. The reproducibility tests and golden values have been updated to reflect this verified improvement.
- **Leakage-Free Tuning**: Optuna hyperparameter tuning now rigorously loops across multiple `TimeSeriesSplit` chronological folds, safely isolating data preprocessors completely inside each training step.
- **Horizon Censoring**: Horizon targets (7/30/60-day limits) now automatically detect and censor (`NaN`) records at the trailing edge of the dataset to prevent right-censoring bias.

## Current Repository Structure

The project is now organized as a full thesis evidence pipeline:

```text
src/aac_adoption/data/              data loading, cleaning, matching, context features
src/aac_adoption/features/          leakage-safe intake-time feature engineering and feature lists
src/aac_adoption/models/            baseline, boosting, CatBoost training, splitting, metrics, artifacts
src/aac_adoption/analysis/          hypothesis tables, H1/H3/H5 evidence, H2/H4 checks, model selection, thresholds, calibration
src/aac_adoption/diagnostics/       model diagnostics, SHAP summaries, feature-family diagnostics
src/aac_adoption/interpretation/    feature importance and explanation helpers
src/aac_adoption/reporting/         evidence-pack and thesis summary generation
src/aac_adoption/dashboard/         Streamlit data helpers, story visuals, trust/limits page
scripts/                            command-line entrypoints and full-pipeline runners
docs/                               thesis plans, methodology notes, target definitions, technical guide
reports/                            generated tables, figures, summaries, diagnostics, artifact manifest
models/                             generated model artifacts
tests/                              regression tests for data, features, models, reports, audits, dashboard helpers
THESIS_LATEX_FINAL/                 latex thesis source code and files
```

Important local structure changes from recent multi-agent work:

- `scripts/run_full_pipeline.py`, `scripts/run_full_pipeline.ps1`, and `scripts/run_full_pipeline.sh` orchestrate the end-to-end workflow.
- `scripts/backfill_pr_auc.py` backfills PR-AUC into older metric files when existing artifacts predate the metric.
- `scripts/generate_artifact_manifest.py` records generated thesis deliverables and whether they exist on disk.
- `scripts/generate_data_audit.py`, `scripts/generate_leakage_audit.py`, `scripts/generate_matching_examples.py`, `scripts/generate_environment_snapshot.py`, and `scripts/generate_feature_quality_audit.py` create reproducibility and methodology artifacts.
- `src/aac_adoption/analysis/` now contains dedicated modules for hypothesis evidence, final model selection, threshold analysis, calibration summaries, and reliability red flags.
- `src/aac_adoption/reporting/evidence_pack.py` now emits both Animal Journey examples and acceptance-facing local explanation examples with explicit non-causal limitation notes.
- `streamlit_app.py` now presents generated artifacts, methodology reports, trust/limits evidence, animal stories, risk exploration, campaign candidates, and a model sensitivity demo.
- `reports/artifact_manifest.csv` is a lightweight tracked manifest of generated thesis artifacts; large raw data, processed data, and model binaries remain local/generated assets.

## Technical Architecture

The repository follows an artifact-first architecture. Scripts create stable CSV, Markdown, PNG, and model artifacts; the Streamlit app reads those artifacts and does not retrain models interactively.

```text
Austin Open Data / local CSVs
        |
        v
data loading and standardization
        |
        v
clean intakes/outcomes -> match each intake to nearest valid future outcome
        |
        v
feature engineering -> leakage-safe feature registry -> target creation
        |
        v
time-aware train/validation/test split
        |
        v
baseline, boosting, and CatBoost model training
        |
        v
metrics, diagnostics, SHAP, hypothesis evidence, audits, reports
        |
        v
artifact manifest + Streamlit thesis dashboard
```

Core technical principles:

- **Artifact-first:** every important analysis output is written to `reports/` and can be inspected without rerunning the app.
- **Leakage-safe:** configured model features come only from intake-time data and prior context windows.
- **Date-aware:** intakes are matched to future outcomes only, and the default split is chronological.
- **Model-comparison oriented:** dummy, linear, random forest, histogram boosting, and CatBoost models are kept for methodological comparison.
- **Thesis-ready language:** generated evidence uses predictive/descriptive wording, not causal claims.
- **Dashboard as reader:** Streamlit consumes artifacts and saved models; it is a presentation layer.

## Data Flow and Contracts

Raw inputs:

- `data/raw/intakes.csv`
- `data/raw/outcomes.csv`
- optional `data/raw/context/austin_weather_daily.csv`
- optional `data/raw/context/austin_311_animal_requests.csv`

Column normalization:

- raw AAC column names are converted to snake_case in `src/aac_adoption/data/load_data.py`;
- `DateTime` becomes `intake_datetime` for intakes and `outcome_datetime` for outcomes;
- raw files are never modified.

Cleaning contract:

- required intake columns: `animal_id`, `animal_type`, `intake_datetime`;
- required outcome columns: `animal_id`, `animal_type`, `outcome_datetime`, `outcome_type`;
- only cats and dogs are kept;
- exact duplicate rows are removed;
- mixed AAC datetime formats are parsed without shifting local shelter clock time.

Episode matching contract:

- each intake is matched to the nearest unused future outcome for the same `animal_id`;
- outcome rows are not reused for the same animal;
- unmatched intakes are counted and excluded from the modeling frame;
- negative `days_to_outcome` values are rejected.

Processed outputs:

- `data/processed/modeling_dataset.csv`
- `data/processed/feature_columns.json`
- `data/processed/target_columns.json`
- optional `data/processed/context_feature_columns.json`

## Feature and Target Contract

Configured base intake-time features live in `src/aac_adoption/features/feature_sets.py`.

Feature families:

- identity and intake: `animal_type`, `intake_type`, `intake_condition`, `sex_upon_intake`;
- age: raw age string plus `age_days`, `age_months`, `age_years`, `age_group`;
- breed: raw breed, `primary_breed`, `is_mixed_breed`, `simplified_breed_group`;
- color: raw color, `primary_color`, `simplified_color_group`, `is_black_or_dark`;
- name: `has_name`, `is_named`;
- time: `intake_year`, `intake_month`, `intake_quarter`, `intake_season`, `covid_period`;
- found location: `found_location_kind`, `found_location_area`, and location flags;
- optional context: weather, rainy/heat flags, prior 311 animal requests, prior shelter intake volume.

Found Location taxonomy:

- raw `Found Location` is preserved in raw data but not used directly as a model feature;
- derived fields are deterministic, reproducible, and geocoder-free;
- categories include `austin_city`, `county_or_region`, `outside_jurisdiction`, `intersection`, `address_like`, and `other`;
- flags include Austin, outside jurisdiction, intersection, address-like, and airport markers.

Targets and metadata:

- classification target: `classification_target` / `target_adopted` / `adopted`;
- regression target: `regression_target_days`, equivalent to `days_to_outcome` / `length_of_stay`;
- adopted-only timing support: `days_to_adoption` is populated only for adopted records;
- metadata columns such as `outcome_type`, `outcome_datetime`, `sex_upon_outcome`, and `age_upon_outcome` are not predictors.

Leakage rules:

- `validate_no_leakage()` rejects outcome-derived feature columns;
- columns containing future-window naming such as `next_` or `_next_` are rejected;
- `animal_id` and `intake_datetime` may exist as metadata but are not ordinary model predictors;
- context windows use prior dates only.

Schema compatibility rules:

- some generated report tables intentionally include alias columns for acceptance and dashboard compatibility;
- examples include `subset` alongside `animal_subset`, `threshold_name` alongside `threshold_label`, and `subgroup_field` / `subgroup_value` alongside `cohort` / `value`;
- H3 adopted-only timing output includes `age_group` and `records` aliases for `group_value` and `all_records`;
- calibration output includes `subset` and `records` aliases so reviewer-facing checks can read the same table without knowing internal naming;
- reliability red-flag output keeps the original cohort columns and adds `subgroup_field`, `subgroup_value`, and `mean_predicted_probability` for the pasted acceptance contract;
- leakage audit output keeps detailed internal columns and adds user-facing aliases such as `role`, `allowed_as_feature`, `leakage_risk`, and `notes`;
- these alias columns are additive only: existing dashboard/report readers can keep using the original internal column names.

## Modeling Stack

Model families:

- dummy baselines for sanity checks;
- logistic regression and ridge regression for interpretable linear baselines;
- random forest baseline models with feature importance;
- scikit-learn `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`;
- CatBoost classifier/regressor for categorical-heavy shelter features.

Animal subsets:

- `combined`;
- `dogs`;
- `cats`.

Default split:

```text
train:      2013-2021
validation: 2022-2023
test:       2024-2025
```

If these years are unavailable, training falls back to deterministic random splits.

Primary model metrics:

- classification: ROC-AUC, PR-AUC, F1, precision, recall, accuracy;
- regression: MAE, RMSE, R2 and median absolute error where available;
- calibration and threshold artifacts are generated separately for operational interpretation.

Saved model artifacts:

- baseline models: `models/baseline/`;
- histogram boosting models: `models/boosting/`;
- CatBoost models: `models/advanced/`;
- artifact metadata includes feature lists and categorical feature lists.

## Analysis and Evidence Layers

Hypothesis layers:

- H1: intake circumstances vs appearance/breed/color, supported by descriptive tables, feature-family importance, and optional ablation;
- H2: seasonality, supporting descriptive check;
- H3: age, adoption likelihood, length of stay, adopted-only timing, and SHAP evidence;
- H4: dark coat color / black dog-cat syndrome, supporting descriptive check;
- H5: COVID-period population and outcome-pattern shift, descriptive and predictive evidence.

Model rigor layers:

- final model selection summary;
- classification threshold analysis;
- confusion matrix at selected threshold;
- calibration summary by subset;
- reliability red flags;
- error slices and placement-risk quadrants;
- subgroup reliability and bootstrap intervals;
- model failure modes;
- descriptive adoption milestones at 7, 30, 60, and 90 days.

Interpretability layers:

- logistic regression coefficients;
- random forest importance;
- permutation importance;
- SHAP global feature tables;
- SHAP feature-family summaries;
- local SHAP explanations for representative animal journey cards;
- local explanation examples that combine journey-card profiles, similar historical cases, SHAP/model reasons, and limitation notes;
- standalone feature-family importance plots.

Audit and reproducibility layers:

- data attrition audit;
- leakage audit;
- matching examples;
- target definitions documentation;
- feature quality audit;
- environment snapshot;
- artifact manifest.

## Script Catalog

Data and context:

| Script | Purpose |
|--------|---------|
| `scripts/download_raw_data.py` | Download historical or current AAC intake/outcome CSV exports. |
| `scripts/download_context_data.py` | Download Austin weather and 311 animal-service context data. |
| `scripts/build_dataset.py` | Build processed modeling dataset and feature/target metadata. |
| `scripts/build_modeling_dataset.py` | Compatibility entrypoint for dataset building. |
| `scripts/compare_context_models.py` | Compare base vs context-enriched model metrics. |

Training:

| Script | Purpose |
|--------|---------|
| `scripts/train_baseline.py` | Train dummy, linear, and random forest baselines. |
| `scripts/train_baselines.py` | Compatibility wrapper for baseline training. |
| `scripts/train_adopted_regression.py` | Train models predicting time-to-adoption (adopted subset only). |
| `scripts/tune_models.py` | Run Optuna hyperparameter tuning for boosting and CatBoost models. Generates `tuning_results.csv`, `best_params.csv`, and `selected_model_reason.md` in `reports/tuning/`. |
| `scripts/train_boosting.py` | Train histogram gradient boosting models natively handling class imbalance and calculate permutation importance. |
| `scripts/train_advanced.py` | Train CatBoost classifier/regressor artifacts. |
| `scripts/calibrate_classifiers.py` | Produce calibrated classifiers via Isotonic Regression wrappers. |
| `scripts/backfill_pr_auc.py` | Backfill PR-AUC values into older metric files. |

Analysis, diagnostics, and reports:

| Script | Purpose |
|--------|---------|
| `scripts/run_eda.py` / `scripts/make_eda_outputs.py` | Generate initial EDA tables and figures. |
| `scripts/run_analysis.py` | Generate model comparison, hypotheses, model selection, thresholds, calibration, and evidence matrix outputs. |
| `scripts/generate_diagnostics.py` | Generate calibration, error slices, risk quadrants, SHAP, and adoption milestones. |
| `scripts/generate_animal_research.py` | Generate animal archetypes, vulnerability profiles, contrasts, and profile error summaries. |
| `scripts/generate_evidence_pack.py` | Generate ML-rigor evidence pack, subgroup reliability, intervals, milestones, journey examples, and local explanation examples. |
| `scripts/generate_report_outputs.py` | Generate thesis-ready summary Markdown and key report figures. |
| `scripts/generate_feature_family_importance.py` | Create standalone feature-family SHAP importance summaries and plots. |

Audits and reproducibility:

| Script | Purpose |
|--------|---------|
| `scripts/generate_data_audit.py` | Create data attrition and dataset audit artifacts. |
| `scripts/generate_leakage_audit.py` | Create leakage-control report from configured feature/target metadata. |
| `scripts/generate_matching_examples.py` | Create human-readable examples of intake/outcome matching. |
| `scripts/generate_environment_snapshot.py` | Save library/runtime version snapshot. |
| `scripts/generate_feature_quality_audit.py` | Audit feature presence, missingness, and quality. |
| `scripts/generate_artifact_manifest.py` | Inventory generated thesis artifacts and disk presence. |

Orchestration:

| Script | Purpose |
|--------|---------|
| `scripts/run_full_pipeline.py` | Python end-to-end pipeline runner with step selection and skip flags. |
| `scripts/run_full_pipeline.ps1` | PowerShell wrapper for Windows. |
| `scripts/run_full_pipeline.sh` | Shell wrapper for Unix-like environments. |

## Artifact Layout

Core generated directories:

```text
reports/metrics/       model metric CSVs
reports/tables/        model comparison, hypothesis, audit, reliability, and evidence tables
reports/figures/       EDA, hypothesis, diagnostic, SHAP, and report figures
reports/diagnostics/   prediction samples, calibration bins, error slices, risk quadrants
reports/summary/       thesis-ready Markdown summaries and audit narratives
models/                trained model artifacts and metadata
data/processed/        processed modeling datasets and feature/target metadata
logs/                  optional full-pipeline run logs
```

Artifact manifest:

- `reports/artifact_manifest.csv` is tracked as a lightweight index;
- `reports/summary/artifact_manifest.md` is generated for human review;
- status values are ASCII `present` / `missing`;
- required thesis artifacts are tested for existence when manifest files are available;
- manifest text is normalized to avoid mojibake dash artifacts in generated Markdown and CSV cells;
- generator-declared artifacts such as local explanation examples are listed with their source script, chapter mapping, notes, and disk-presence status.

## Test Suite Map

The test suite is configured by `pyproject.toml`:

```text
pythonpath = ["src"]
testpaths = ["tests"]
```

Coverage areas:

- data loading, cleaning, matching, dataset build;
- feature engineering and feature-set leakage checks;
- split logic and model artifact helpers;
- baseline, boosting, and CatBoost output contracts;
- EDA, analysis, diagnostics, and report outputs;
- evidence pack and subgroup reliability functions;
- data audit, leakage audit, target definitions, and artifact manifest;
- dashboard data helpers and story content;
- acceptance/schema aliases for generated artifacts;
- local explanation example generation and manifest registration.

Current expected verification:

```bash
pytest
```

Known warning:

- pandas may emit a `FutureWarning` from evidence-pack concatenation when synthetic tests use empty/all-NA frames; this does not currently fail the suite.

## Documentation Map

Key docs:

| Document | Purpose |
|----------|---------|
| `docs/target_definitions.md` | Formal target definitions and leakage-safe target framing. |
| `docs/METHODOLOGY.md` | Methodology, matching, evaluation, and causal-language limits. |
| `docs/RESULTS.md` | Current result interpretation and reporting notes. |
| `docs/ARCHITECTURE.md` | Pipeline and system architecture. |
| `docs/ROADMAP.md` | Implemented work and remaining project risks. |
| `docs/PROJECT_CLOSEOUT_TASKS.md` | Detailed acceptance and closeout checklist. |
| `reports/summary/model_evidence_pack.md` | Generated evidence-pack interpretation. |

## Research Priority

Do not treat all thesis hypotheses as equal implementation priorities.

Central analytical threads:

- **H1:** intake type vs breed/color/appearance,
- **H3:** age and length-of-stay / time-to-outcome patterns (descriptive adoption timing among adopted animals),
- **H5:** COVID-period change in adoption rates and outcome patterns.

Secondary descriptive threads:

- **H2:** seasonality,
- **H4:** black dog/cat syndrome (dark coloring and adoption rate - descriptive association only).

The code should first support robust dataset construction, baseline modeling, and clean outputs for H1/H3/H5. H2 and H4 should remain available through prepared variables such as `intake_month`, `intake_season`, `color`, and `color_group`, but they should not dominate the first modeling work.

Not in current scope:

- complex MLOps,
- advanced hypothesis testing,
- production deployment.

## Expected Data

Place raw CSV files in:

```text
data/raw/
```

Expected inputs:

```text
data/raw/intakes.csv
data/raw/outcomes.csv
```

Raw data is not modified by the pipeline.

## Download Raw AAC Data

The downloader uses Austin Open Data / Socrata CSV exports.

For the thesis period, download the historical 2013-2025 files:

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw
```

This creates:

```text
data/raw/intakes.csv
data/raw/outcomes.csv
```

To replace existing raw files:

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
```

Available sources:

- `historical`: 2013-10-01 to 2025-05-05 AAC records.
- `current`: ShelterBuddy records from 2025-05-05 onward.

## Download External Context Data

Optional context features add intake-date weather, prior Austin 311 animal-service demand, and prior shelter intake volume. They remain intake-time-only and are used for predictive association, not causal claims.

```bash
python scripts/download_context_data.py --output-dir data/raw/context
```

This creates:

```text
data/raw/context/austin_weather_daily.csv
data/raw/context/austin_311_animal_requests.csv
```

## Build Modeling Dataset

Install dependencies:

For latest compatible versions:
```bash
pip install -r requirements.txt
```

For exact replication of the thesis environment (fully pinned versions):
```bash
pip install -r requirements-lock.txt
```

Run:

```bash
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
```

Output:

```text
data/processed/modeling_dataset.csv
data/processed/feature_columns.json
data/processed/target_columns.json
```

To build a context-enriched dataset:

```bash
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset_context.csv --context-data-dir data/raw/context
```

Context feature metadata is saved to:

```text
data/processed/context_feature_columns.json
```

To compare base and context-enriched models without overwriting metrics, train into separate metric/model folders and then create the context comparison table:

```bash
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv --metrics-dir reports/metrics_base --models-dir models/base_baseline --tables-dir reports/tables_base --output reports/metrics_base/baseline_metrics.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv --metrics-dir reports/metrics_base --models-dir models/base_boosting --tables-dir reports/tables_base
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv --metrics-dir reports/metrics_base --models-dir models/base_advanced
python scripts/train_baseline.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_baseline --tables-dir reports/tables_context --output reports/metrics_context/baseline_metrics.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_boosting --tables-dir reports/tables_context
python scripts/train_advanced.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_advanced
python scripts/compare_context_models.py --base-metrics-dir reports/metrics_base --context-metrics-dir reports/metrics_context --tables-dir reports/tables
python scripts/generate_report_outputs.py
```

## Train Baseline Models

After `data/processed/modeling_dataset.csv` exists, run:

```bash
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
```

This trains simple classification and regression baselines:

- `DummyClassifier`
- `LogisticRegression`
- `RandomForestClassifier`
- `DummyRegressor`
- `Ridge`
- `RandomForestRegressor`

The default thesis split is:

```text
train: 2013-2021
validation: 2022-2023
test: 2024-2025
```

If those years are not available, the trainer falls back to a fixed random split.

Metrics are saved to:

```text
reports/metrics/baseline_metrics.csv
reports/metrics/classification_metrics.csv
reports/metrics/regression_metrics.csv
```

Fitted model pipelines are saved to:

```text
models/baseline/
```

First-level interpretability tables are saved to:

```text
reports/tables/logistic_regression_coefficients.csv
reports/tables/random_forest_feature_importance.csv
```

To run a faster sampled training pass:

```bash
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv --max-rows 50000
```

## Train Gradient Boosting Models

After baselines are stable, train scikit-learn histogram gradient boosting models:

```bash
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
```

Outputs:

```text
reports/metrics/boosting_classification_metrics.csv
reports/metrics/boosting_regression_metrics.csv
reports/metrics/boosting_metrics.csv
models/boosting/
reports/tables/permutation_importance_classification.csv
reports/tables/permutation_importance_regression.csv
```

For a faster sampled training pass:

```bash
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv --max-rows 50000 --permutation-repeats 1
```

## Train Advanced CatBoost Models

For categorical-heavy shelter data, train CatBoost classifiers and regressors:

```bash
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
```

Outputs:

```text
reports/metrics/advanced_classification_metrics.csv
reports/metrics/advanced_regression_metrics.csv
reports/metrics/advanced_metrics.csv
models/advanced/
```

For a faster development run:

```bash
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv --max-rows 50000 --iterations 100 --depth 4
```

## Run Thesis Analysis Tables

After baseline and boosting metrics exist, run:

```bash
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
```

To include the optional feature-family ablation study training (which trains 6 separate models per subset):

```bash
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv --h1-ablation
```

Outputs:

```text
reports/tables/model_comparison_classification.csv (now with pr_auc and pr_auc_rank)
reports/figures/model_comparison_classification_pr_auc.png
reports/tables/final_model_selection.csv
reports/summary/final_model_selection.md
reports/tables/final_classifier_thresholds.csv
reports/figures/final_confusion_matrix.png
reports/summary/threshold_selection.md
reports/tables/h1_feature_family_importance.csv
reports/figures/h1_feature_family_importance.png
reports/tables/h1_feature_family_ablation.csv (only when run with --h1-ablation)
reports/summary/h1_interpretation.md (with importance and ablation tables)
reports/tables/h2_seasonality_summary.csv
reports/figures/h2_adoption_rate_by_season.png
reports/figures/h2_median_los_by_season.png
reports/summary/h2_interpretation.md
reports/tables/h3_age_evidence_matrix.csv
reports/tables/h3_adopted_only_age_speed.csv
reports/figures/h3_age_adoption_rate.png
reports/figures/h3_age_adopted_only_median_days.png
reports/figures/h3_adopted_only_median_days_to_adoption.png
reports/figures/h3_age_shap_summary.png
reports/summary/h3_interpretation.md
reports/tables/h4_dark_color_summary.csv
reports/figures/h4_dark_color_adoption_rate.png
reports/figures/h4_dark_color_median_los.png
reports/summary/h4_interpretation.md
reports/tables/h5_covid_evidence_matrix.csv
reports/tables/h5_covid_population_mix.csv
reports/figures/h5_covid_adoption_rate.png
reports/figures/h5_covid_median_los.png
reports/figures/h5_covid_intake_volume.png
reports/summary/h5_interpretation.md
reports/tables/calibration_summary_by_subset.csv
reports/summary/calibration_interpretation.md
reports/tables/model_reliability_red_flags.csv
reports/summary/model_reliability_red_flags.md
reports/tables/hypothesis_evidence_matrix.csv
reports/summary/hypothesis_evidence_matrix.md
```

CLI options:

- `--h1-ablation`: Run the slow feature-family ablation study training.
- `--models-dir`: Custom path to model artifacts directory (default: `models`).
- `--skip-survival`: Skip Kaplan-Meier descriptive survival curve generation.

H1, H3, and H5 remain the central thesis hypotheses. H2 and H4 are generated as supporting descriptive checks.

## Generate Model Diagnostics and Interpretability

After advanced model artifacts exist, generate reliability diagnostics, error slices, placement-risk quadrants, adoption timeline tables, and optional SHAP outputs:

```bash
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
```

Outputs include:

```text
reports/diagnostics/classification_thresholds.csv
reports/diagnostics/classification_calibration.csv
reports/diagnostics/classification_error_slices.csv
reports/diagnostics/regression_error_slices.csv
reports/diagnostics/placement_risk_quadrants.csv
reports/tables/shap_global_classification.csv
reports/tables/shap_feature_families_classification.csv
reports/tables/adoption_by_day_milestones.csv
reports/figures/diagnostic_calibration_curve.png
reports/figures/shap_summary_classification.png
```

All interpretation outputs describe predictive association with the model, not causal effects.

To regenerate standalone feature-family importance summaries from SHAP tables:

```bash
python scripts/generate_feature_family_importance.py
```

Outputs:

```text
reports/tables/feature_family_importance_classification.csv
reports/tables/feature_family_importance_regression.csv
reports/figures/feature_family_importance_classification.png
reports/figures/feature_family_importance_regression.png
```

## Generate Animal-Centered Research

After the modeling dataset exists, generate the animal story layer:

```bash
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
```

Outputs include:

```text
reports/tables/animal_archetypes.csv
reports/tables/vulnerable_profiles.csv
reports/tables/profile_contrasts.csv
reports/tables/profile_model_error.csv
reports/tables/health_behavior_profiles.csv
reports/figures/animal_archetypes_top.png
reports/figures/vulnerable_profiles.png
reports/figures/profile_contrasts_adoption_rate.png
```

This layer supports dog and cat profiles such as pit-bull-type vs other dogs, black/dark cats vs other cats, senior vs baby animals, named vs unnamed animals, and health or behavior-support proxy groups. Health and behavior fields are interpreted as shelter-record descriptors, not full personality measures.

The Streamlit Animal Stories tab also builds representative Animal Journey Cards from these profiles. When advanced model artifacts are available, each card adds CatBoost adoption probability, predicted wait, a prediction-derived visibility label, similar historical cases with outcome mix, and local SHAP reasons for the representative record.

## Generate Model Evidence Pack

After diagnostics and animal research artifacts exist, generate the model evidence pack:

```bash
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
```

Outputs include:

```text
reports/tables/model_evidence_pack.csv
reports/tables/model_limitations_by_cohort.csv
reports/tables/metric_confidence_intervals.csv
reports/tables/animal_journey_examples.csv
reports/tables/local_explanation_examples.csv
reports/tables/subgroup_reliability.csv
reports/tables/subgroup_metric_confidence_intervals.csv
reports/tables/subgroup_adoption_milestones.csv
reports/tables/model_failure_modes.csv
reports/summary/model_evidence_pack.md
reports/summary/subgroup_reliability.md
reports/summary/local_explanation_examples.md
```

The evidence pack is the main ML-rigor layer. It summarizes model choice, PR-AUC, bootstrap metric intervals, calibration and error limits by cohort, SHAP feature-family evidence, selected Animal Journey examples, local explanation examples, subgroup reliability, model failure modes, and descriptive adoption milestones at days 7, 30, 60, and 90. It uses association language, not causal language.

Local explanation examples are thesis/demo artifacts, not causal case studies. Each row in `reports/tables/local_explanation_examples.csv` is derived from an Animal Journey profile and includes:

- `explanation_type`: profile-level explanation framing;
- `profile_label`: representative animal profile label;
- `similar_historical_cases`: nearest/similar historical cohort summary;
- `shap_model_reasons`: local SHAP reasons or global SHAP fallback reasons linked to model behavior;
- `limitation_note`: explicit warning that examples are illustrative, non-causal, and not individual certainty.

The companion `reports/summary/local_explanation_examples.md` gives a short thesis-readable explanation of how to cite these examples safely.

Useful development options:

```bash
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv --bootstrap-samples 100 --milestone-min-records 50
```

## Generate Thesis Report Outputs

After analysis tables exist, generate thesis-ready summary text and figures:

```bash
python scripts/generate_report_outputs.py
```

Outputs:

```text
reports/summary/current_results.md
reports/figures/model_comparison_classification_roc_auc.png
reports/figures/model_comparison_classification_f1.png
reports/figures/model_comparison_regression_mae.png
reports/figures/model_comparison_regression_rmse.png
reports/figures/h1_intake_type_adoption_rate.png
reports/figures/h1_intake_condition_adoption_rate.png
reports/figures/h3_age_group_adoption_rate.png
reports/figures/h3_age_group_median_days.png
reports/figures/h5_covid_period_adoption_rate.png
reports/figures/h5_covid_period_median_days.png
```

## Generate Audit, Reproducibility, and Manifest Artifacts

Recent local work added audit and manifest scripts that make the thesis easier to defend and reproduce:

```bash
python scripts/generate_data_audit.py --data data/processed/modeling_dataset.csv
python scripts/generate_leakage_audit.py
python scripts/generate_matching_examples.py
python scripts/generate_environment_snapshot.py
python scripts/generate_feature_quality_audit.py --data data/processed/modeling_dataset.csv
python scripts/generate_artifact_manifest.py
```

Typical outputs include:

```text
reports/tables/data_audit.csv
reports/tables/leakage_audit.csv
reports/tables/local_explanation_examples.csv
reports/tables/matching_logic_examples.csv
reports/tables/environment_snapshot.csv
reports/tables/feature_quality_audit.csv
reports/artifact_manifest.csv
reports/summary/data_audit.md
reports/summary/leakage_audit.md
reports/summary/local_explanation_examples.md
reports/summary/matching_logic_examples.md
reports/summary/environment_snapshot.md
reports/summary/feature_quality_audit.md
reports/summary/artifact_manifest.md
```

The manifest uses `present` / `missing` status values and is displayed in the Streamlit Artifacts tab. Manifest generation also cleans common mojibake dash sequences before writing CSV/Markdown text, which keeps generated artifact notes readable in thesis appendices.

Acceptance-facing artifact contracts currently enforced by tests include:

- leakage audit aliases: `role`, `allowed_as_feature`, `leakage_risk`, `notes`;
- adopted-only H3 aliases: `age_group`, `records`;
- model-selection alias: `subset`;
- threshold alias: `threshold_name`;
- calibration aliases: `subset`, `records`;
- red-flag aliases: `subgroup_field`, `subgroup_value`, `mean_predicted_probability`;
- local explanation artifact presence and non-causal limitation text;
- manifest entries for `local_explanation_examples.csv` and `local_explanation_examples.md`.

## Run with Docker (Recommended)

To avoid local dependency and environment issues, the entire project is containerized. You can run all pipeline, testing, and dashboard commands inside an isolated Docker container using the included PowerShell helper script (requires [Docker Desktop](https://www.docker.com/products/docker-desktop/)):

```powershell
# Boot up the Streamlit dashboard on port 8501
.\docker.ps1 dashboard

# Boot up the Jupyter Notebook server on port 8888
.\docker.ps1 jupyter

# Run the full data and training pipeline
.\docker.ps1 pipeline

# Run the quick pipeline (skips slow steps)
.\docker.ps1 quick

# Run the full pytest suite
.\docker.ps1 test
```

*Note: The container automatically mounts your local `data/`, `models/`, `reports/`, and `logs/` directories, so all generated artifacts and models persist on your host machine exactly as if you ran them natively.*

## Run Full Pipeline Locally

For a full local refresh, use one of the orchestration scripts:

```bash
python scripts/run_full_pipeline.py
```

Windows PowerShell:

```powershell
.\scripts\run_full_pipeline.ps1
```

Shell:

```bash
./scripts/run_full_pipeline.sh
```

The full pipeline builds data, trains models, generates diagnostics, creates evidence packs, writes audit artifacts, and refreshes the artifact manifest. It does not change raw data unless the download step is run separately.

## Run Streamlit Thesis Demo

After the dataset, models, analysis tables, and report figures exist, run:

```bash
streamlit run streamlit_app.py
```

The demo reads existing artifacts instead of retraining models. It includes:

- story mode with workflow and approach-comparison visuals,
- animal journey cards and animal-centered contrast views,
- overview of current generated results,
- model comparison figures and tables,
- H1/H3/H5 hypothesis figures and tables,
- H2/H4 supporting descriptive hypothesis checks,
- reliability diagnostics, SHAP interpretability, risk explorer, campaign finder, and adoption timeline,
- Trust & Limits evidence-pack view with subgroup selector, calibration-gap chart, confidence intervals, model-struggle table, and adoption milestone chart,
- generated artifact manifest and thesis/methodology report reader,
- a simple model sensitivity form using combined CatBoost artifacts.

The dashboard uses model sensitivity language instead of causal counterfactual claims. Changing form inputs shows how the trained model responds to different records; it does not prove that changing a real animal's characteristic would change the outcome.

## Recommended Local Reproduction Order

```bash
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv --h1-ablation
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
python scripts/generate_data_audit.py --data data/processed/modeling_dataset.csv
python scripts/generate_leakage_audit.py
python scripts/generate_matching_examples.py
python scripts/generate_environment_snapshot.py
python scripts/generate_feature_quality_audit.py --data data/processed/modeling_dataset.csv
python scripts/generate_artifact_manifest.py
pytest
```

For current interpretation notes, see:

```text
docs/RESULTS.md
```

## Create Initial EDA Outputs

After the modeling dataset exists, run:

```bash
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
```

Tables are saved to:

```text
reports/tables/
```

Figures are saved to:

```text
reports/figures/
```

## Run Validation Tests

```bash
pytest
```

## Dataset Assumptions

AAC animals can appear multiple times. The first version of the pipeline treats each intake as a possible separate stay episode.

For each `animal_id`, each intake is matched to the nearest unused outcome whose `outcome_datetime` is greater than or equal to `intake_datetime`. This avoids matching one outcome to many intakes and prevents negative length-of-stay values.

Outcome fields are used only to create labels and target variables. Intake-time fields are used as predictors to reduce data leakage.

Feature columns are saved to `data/processed/feature_columns.json`; target columns are saved to `data/processed/target_columns.json`.

## Important Output Columns

The processed modeling dataset includes, where available:

- `animal_id`
- `animal_type`
- `intake_datetime`
- `outcome_datetime`
- `intake_type`
- `intake_condition`
- `outcome_type`
- `sex_upon_intake`
- `age_upon_intake`
- `breed`
- `color`
- `found_location_kind`
- `found_location_area`
- `is_austin_found_location`
- `is_outside_jurisdiction`
- `is_intersection_location`
- `is_address_like_location`
- `is_airport_location`
- `has_name`
- `is_named`
- `age_in_days`
- `age_in_months`
- `age_in_years`
- `age_days`
- `age_months`
- `age_years`
- `age_group`
- `intake_year`
- `intake_month`
- `intake_quarter`
- `intake_season`
- `covid_period`
- `color_group`
- `primary_color`
- `simplified_color_group`
- `is_black_or_dark`
- `primary_breed`
- `is_mixed_breed`
- `simplified_breed_group`
- `days_to_outcome`
- `length_of_stay`
- `adopted`
- `is_adopted`
- `target_adopted`
- `classification_target`

## Validation Checks

The build script checks:

- required columns exist,
- final dataset contains only cats and dogs,
- `days_to_outcome` is not negative,
- target columns are created correctly.
- leakage columns are not included in configured feature lists.

## Modeling Pipeline Critique & Future ML Improvements

*(Note: This section contains critical machine learning critiques and future improvements for developers and AI agents working on this repository.)*

### 1. Target Leakage via Temporal Feature Updates
- **Critique:** `sex_upon_intake` and `intake_condition` are treated as static intake-time features, but in shelter databases, these fields are often overwritten or updated *post-intake* (e.g., an animal neutered during its stay may have its record updated from "Intact" to "Spayed/Neutered", or a sick animal's condition may be updated). This leaks the length-of-stay outcome (longer stay increases the probability of medical treatment or neutering).
- **Improvement:** Implement strict audit rules or use raw intake transaction logs to freeze features at the *exact timestamp of admission*.

### 2. Lack of Post-Hoc Probability Calibration
- **Critique:** The pipeline reports high calibration gaps (up to 0.21 for Terrier-type dogs, and a mean gap of 0.112 across bins). The CatBoost and HistGradientBoosting classifiers output uncalibrated probabilities, meaning output scores cannot be treated as literal probabilities of adoption.
- **Improvement:** Apply post-hoc calibration scaling (e.g., Isotonic Regression or Platt Scaling) fit on the validation split (2022-2023) before evaluating on the test split.

### 3. Suboptimal Regression Modeling for Length of Stay (LOS)
- **Critique:** Shelter stay duration is highly right-skewed, zero-inflated, and strictly non-negative. Training regressors using standard MAE/MSE metrics leads to poor convergence, negative days-to-outcome predictions, and extreme sensitivity to long-stay outliers.
- **Improvement:** 
  - Log-transform the target: Train models on $y' = \log(y + 1)$ and invert predictions.
  - Frame as Survival Analysis: Use Accelerated Failure Time (AFT) models or Cox Proportional Hazards with censoring markers, treating non-adoptions as censored stays.

### 4. Concept Drift and Dataset Aging
- **Critique:** The time split uses a training window spanning 2013 to 2021. Shelter operations, intake policies, and public adoption demand changed drastically post-COVID. Weighting 2013 data equally to 2021 data degrades model performance for the 2024-2025 test period.
- **Improvement:** Implement recency-based sample weighting during model training, or transition to a sliding-window time split (e.g., training only on the most recent 3–5 years).

### 5. High Cardinality Categorical Encoding (Breed and Color)
- **Critique:** Categorical features like `breed` and `color` have hundreds of levels. The current pipeline maps them to simplified groups statically. This loses granular information (e.g., distinguishing specific terrier types) while still suffering from high-cardinality noise, and maps them to a very coarse set of 8 categories where `other` becomes a catch-all category grouping together highly distinct breeds (e.g. Rottweilers, Boxers, Poodles, Siamese cats, etc.).
- **Improvement:** Replace static mapping with target encoding (with smoothing) or use text embedding vectors derived from a pretrained model for breeds and colors.

### 6. Batch-dependent Dynamic Rolling Features (Online Prediction Failure)
- **Critique:** The rolling features `intake_volume_7d/30d` and `animal_311_requests_7d/30d` are computed dynamically from the input DataFrame batch itself. If a single animal's record is passed to the pipeline for inference (e.g., in a production API or the Streamlit model sensitivity form), the rolling counts will fall back to zero (or be calculated incorrectly) because the historical window is absent in the input batch. This makes the model architecture incompatible with real-time, online single-record inference.
- **Improvement:** Decouple rolling feature computation from the prediction batch. In production, rolling features should query a persistent historical context database or stream-processing layer.

### 7. Underestimation of Shelter Volume via Matched-Only Filtering
- **Critique:** The rolling intake volume feature is calculated *after* filtering for cats and dogs, and *after* matching intakes to outcomes. Thus, unmatched intakes (e.g., animals currently in the shelter or those that escaped/died without outcomes), other animal types (birds, wildlife, livestock), and active shelter residents are excluded from the rolling count. This results in a systematic underestimation of the actual shelter crowding/volume on any given day.
- **Improvement:** Compute daily shelter volume and rolling context features from the *raw, unfiltered* intake dataset before matching and subsetting.

### 8. Greedy Chronological Matching without Verification of Re-Intakes
- **Critique:** The matching logic matches the *first* intake of an animal to its *first* outcome after that intake. If an animal has a re-intake before the first stay has an outcome record (e.g., due to missing outcome records or data logging errors), the first intake is matched to the future outcome (artificially inflating the length of stay), while the second, more recent intake is ignored and left unmatched.
- **Improvement:** Validate that no intermediate intakes occur between an intake-outcome matched pair. If a second intake occurs, mark the first stay as right-censored/unmatched, and match the second intake to the outcome instead.

### 9. Right-Censoring Bias at Temporal Split Boundaries
- **Critique:** Staying episodes are matched only if they have a completed outcome before the data export date. For the test split (2024-2025), animals admitted near the end of 2025 will either be omitted due to no outcomes yet, or will only be included if they had a *short* stay. This introduces a strong right-censoring bias, artificially deflating length of stay metrics at the end of splits.
- **Improvement:** Implement survival analysis methods that handle right-censoring natively, allowing active/unresolved stays to contribute to the model's training and evaluation.

### 10. Inefficient Categorical Feature Handling in Gradient Boosting
- **Critique:** For scikit-learn's `HistGradientBoostingClassifier`, categorical features like `primary_breed` and `primary_color` are one-hot encoded manually. This results in a very wide, sparse feature matrix. Since tree-based models can natively split categoricals, this manual encoding is highly suboptimal compared to native categorical support (which is natively supported in HistGradientBoosting since version 0.24).
- **Improvement:** Pass categorical features directly (as integers or categories) and enable `categorical_features="from_dtype"` or pass the indices/boolean mask to the classifier.

### 11. Feature Space Redundancy and Multicollinearity
- **Critique:** The feature set includes highly collinear features, such as `age_days`, `age_months`, and `age_years` (which are direct linear transformations of each other), and `has_name` and `is_named` (which are identical). This introduces unnecessary noise, degrades linear models, and complicates feature importance/SHAP interpretations.
- **Improvement:** Prune redundant columns. Keep a single continuous age feature (`age_days`) and a single boolean name flag (`is_named`).

### 12. Model Selection and Diagnostics Pipeline Mismatch
- **Critique:** The diagnostics script (`generate_diagnostics.py`) hardcodes loading the CatBoost classifier (`catboost`) to produce calibration curves, error slices, and placement-risk quadrants. However, the model selection step selects `HistGradientBoosting` for `combined` and `cats` classification due to higher ROC-AUC. As a result, the reported diagnostic graphs, calibration tables, and reliability red flags in `reports/` do not represent the selected models.
- **Improvement:** Update the diagnostics pipeline to load the *selected* model for each subset and task from `final_model_selection.csv` instead of hardcoding CatBoost.

### 13. Fixed-Threshold Leaderboard Metrics Comparison
- **Critique:** Leaderboard comparisons (like F1, Precision, and Recall) are calculated using a hard decision threshold of 0.5 for all models. This is unfair to models calibrated differently or trained with balanced class weights, as their optimal threshold may differ significantly from 0.5. Models should be compared on threshold-independent metrics (like ROC-AUC and PR-AUC) or at their respective optimal F1 thresholds.
- **Improvement:** Tune the decision threshold for each model to maximize F1 on the validation set, and report test metrics at those optimal thresholds.

### 14. Regression Target Conflation (Stay Duration vs. Adoption Speed)
- **Critique:** The regressor is trained on `regression_target_days` for *all* outcomes (including fast exits like transfers, euthanasias, and owner returns). When this regressor is used to predict wait times for animal adoption cards, it systematically underestimates stay duration because it is trained on non-adoption fast exits.
- **Improvement:** Train a dedicated "adoption speed" regressor on the subset of adopted animals only, or use a multi-task learning/survival framework that explicitly models outcome type and duration simultaneously.

### 15. Aggregated Context Features in SHAP Feature Family Importance
- **Critique:** In the SHAP feature family mapping (`feature_families.py`), context features (weather, 311 requests, daily precipitation, shelter volume) are not mapped to specific categories, resulting in them all falling back to the `"other"` feature family. This obscures the collective contribution of environmental and shelter-demand features.
- **Improvement:** Add explicit context mappings in `FEATURE_FAMILY_TERMS` (e.g., `"weather": ["temp", "precipitation", "heat", "rainy"]` and `"shelter_demand": ["311", "volume"]`).

### 16. Subgroup Mismatch in Calibration Summaries
- **Critique:** In `calibration_summary.py`, the calibration gaps for `dogs` and `cats` are not filtered or recalculated by species; instead, the script copies the `combined` subgroup reliability table and labels them as `dogs` and `cats` separately, meaning the reported metrics are identical to the combined subset.
- **Improvement:** Modify `calibration_summary.py` to filter `subgroup_reliability.csv` by the respective species subset (or recalculate reliability metrics per species) before computing subset calibration statistics.

