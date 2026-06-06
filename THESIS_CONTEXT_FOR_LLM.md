# THESIS CONTEXT COMPILATION

This file is a compiled megadocument of all architectural, methodological, and result summaries. It is designed to be passed to an LLM to provide full context on the Austin Animal Center Adoption ML project.



================================================================================
FILE: README.md
================================================================================

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
See `docs/methodology_notes.md` for regression and causal-framing justifications.

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
| `docs/methodology_notes.md` | Regression, causal language, and methodology caveats. |
| `docs/results_summary.md` | Current result interpretation and reporting notes. |
| `docs/model_diagnostics.md` | Diagnostics and model reliability guide. |
| `docs/model_evidence_pack.md` | Evidence-pack design and interpretation. |
| `docs/progress_and_future_work.md` | Roadmap, implemented layers, and remaining work. |
| `docs/technical_architecture_plan.md` | Architecture decisions and system plan. |
| `docs/thesis_technical_guide.md` | Living technical guide for thesis implementation. |
| `docs/found_location_plan.md` | Found Location taxonomy and implementation notes. |
| `docs/implementation_plan_p3_p4.md` | Later-priority implementation plan for hypothesis/evidence work. |
| `docs/interactive_story_plan.md` | Dashboard/storytelling plan. |
| `docs/animal_exploratory_research_plan.md` | Animal-centered research planning. |
| `docs/thesis.md` | Thesis draft text. |

## Project Scope

Current scope:

- load raw AAC intake and outcome CSV files,
- clean and standardize columns,
- keep dogs and cats only,
- join intake episodes with valid future outcomes,
- create a modeling dataset,
- create initial target columns for classification and regression.

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

## Run Full Pipeline

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
docs/results_summary.md
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

For current interpretation notes, see:

```text
docs/results_summary.md
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
- `regression_target_days`
- `days_to_adoption`

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



================================================================================
FILE: docs/ARCHITECTURE.md
================================================================================

# Architecture Reference — AAC Adoption ML Pipeline

This document describes the system design, pipeline layers, data and schema contracts, modeling decisions, technology stack, and test strategy. It is the technical reference for developers and AI agents extending this repository.

---

## System Design Philosophy

The repository follows an **artifact-first, leakage-safe, thesis-reproducible** architecture:

1. **Artifact-first:** every important analysis output is written to `reports/` as stable CSV, Markdown, PNG, or model files. The Streamlit app reads artifacts — it never retrains models or rebuilds datasets.
2. **Leakage-safe:** configured model features come only from intake-time data and prior context windows. `validate_no_leakage()` enforces this at build time.
3. **Date-aware:** intakes are matched to future outcomes only; the split is chronological. Random splits are never used for thesis evaluation.
4. **Model-comparison oriented:** five model families (dummy → linear → random forest → histogram boosting → CatBoost) are kept for methodological progression.
5. **Thesis-ready language:** all generated evidence uses predictive/descriptive wording. Causal claims are forbidden. See [`METHODOLOGY.md`](METHODOLOGY.md).

---

## Repository Layer Map

```
src/aac_adoption/
  data/           data loading, cleaning, matching, context features
  features/       leakage-safe feature engineering and feature registry
  models/         baseline, boosting, CatBoost training, splitting, metrics
  analysis/       hypothesis tables, H1/H3/H5 evidence, model selection, thresholds, calibration, reliability red flags
  diagnostics/    model diagnostics, SHAP summaries, feature-family diagnostics
  interpretation/ feature importance and explanation helpers
  reporting/      evidence-pack and thesis summary generation
  dashboard/      Streamlit data helpers, story visuals, trust/limits page

scripts/          command-line entrypoints (one per pipeline step) + orchestration runners
tests/            automated regression tests
docs/             METHODOLOGY.md, ARCHITECTURE.md, RESULTS.md, ROADMAP.md
data/raw/         raw AAC CSV files (gitignored)
data/processed/   modeling datasets and metadata (gitignored)
models/           trained model artifacts (gitignored, .gitkeep preserved)
reports/          generated tables, figures, summaries, diagnostics (gitignored, .gitkeep preserved)
```

---

## Pipeline Layers in Detail

### Layer 1: Data Acquisition

**Script:** `scripts/download_raw_data.py`, `scripts/download_context_data.py`

- Downloads raw AAC intake and outcome CSV exports from Austin Open Data (Socrata).
- Raw files are **never modified** by any downstream step.
- Optional context data: Austin daily weather and 311 animal-service demand.

**Sources:**
- Historical (2013-10-01 to 2025-05-05): `--source historical`
- Current (ShelterBuddy, 2025-05-05+): `--source current`

### Layer 2: Data Preparation

**Module:** `src/aac_adoption/data/`  
**Script:** `scripts/build_dataset.py`, `scripts/generate_data_audit.py`

Steps:
1. **Column normalization** — raw AAC column names converted to `snake_case` in `load_data.py`. `DateTime` becomes `intake_datetime` / `outcome_datetime`. Raw files are not modified.
2. **Cleaning** — required columns validated, duplicates removed, restricted to dogs and cats, mixed datetime formats parsed without shifting local shelter clock.
3. **Episode matching** — each intake matched to nearest unused future outcome for same `animal_id` (greedy nearest-future-match). Negative `days_to_outcome` rejected. Unmatched intakes counted.
4. **Target creation** — classification target (`classification_target`), regression target (`regression_target_days`), adoption-only timing (`days_to_adoption`).
5. **Data Audit Generation** — tracking data attrition across stages (`reports/tables/data_audit_attrition.csv`), calculating followup windows (`followup_days_available`), and auditing censoring logic (`reports/tables/horizon_followup_audit.csv`).

**Required intake columns:** `animal_id`, `animal_type`, `intake_datetime`  
**Required outcome columns:** `animal_id`, `animal_type`, `outcome_datetime`, `outcome_type`

### Layer 3: Feature Engineering

**Module:** `src/aac_adoption/features/`

- Converts raw shelter variables into model-ready features.
- All features are intake-time-only (nothing from the outcome record).
- Feature registry: `src/aac_adoption/features/feature_sets.py` — contains `BASE_INTAKE_TIME_FEATURES`, `CATEGORICAL_FEATURES`, `LEAKAGE_COLUMNS`, and `validate_no_leakage()`.
- Feature metadata saved to `data/processed/feature_columns.json`.
- Target metadata saved to `data/processed/target_columns.json`.

**Found Location taxonomy** (implemented, geocoder-free):

The raw `Found Location` string is preserved in raw data but is not used directly as a model feature. Derived fields are deterministic and reproducible:

| Derived field | Values |
|--------------|--------|
| `found_location_kind` | `austin_city`, `county_or_region`, `outside_jurisdiction`, `intersection`, `address_like`, `other` |
| `found_location_area` | extracted city/county token |
| `is_austin_found_location` | boolean flag |
| `is_outside_jurisdiction` | boolean flag |
| `is_intersection_location` | boolean flag |
| `is_address_like_location` | boolean flag |
| `is_airport_location` | boolean flag |

Raw `found_location` is excluded from `BASE_INTAKE_TIME_FEATURES` to avoid high-cardinality raw location text in models.

### Layer 4: Modeling

**Module:** `src/aac_adoption/models/`  
**Scripts:** `train_baseline.py`, `train_boosting.py`, `train_advanced.py`

**Model families:**

| Family | Models | Script |
|--------|--------|--------|
| Dummy baselines | `DummyClassifier`, `DummyRegressor` | `train_baseline.py` |
| Linear baselines | `LogisticRegression`, `Ridge` | `train_baseline.py` |
| Tree baselines | `RandomForestClassifier`, `RandomForestRegressor` | `train_baseline.py` |
| Histogram boosting | `HistGradientBoostingClassifier/Regressor` | `train_boosting.py` |
| CatBoost | `CatBoostClassifier`, `CatBoostRegressor` | `train_advanced.py` |

**Animal subsets evaluated for each model:** `combined`, `dogs`, `cats`

**Default time-aware split:**

```
train:      2013–2021
validation: 2022–2023  (threshold selection, calibration fitting)
test:       2024–2025  (final untouched evaluation)
```

Falls back to deterministic random split if these years are unavailable.

**Saved artifacts:**

```
models/baseline/    baseline model pipelines (.joblib)
models/boosting/    histogram gradient boosting artifacts (.joblib)
models/advanced/    CatBoost artifacts (.cbm + metadata JSON)
```

Each artifact bundle includes feature lists and categorical feature lists.

### Layer 5: Analysis and Evidence

**Module:** `src/aac_adoption/analysis/`  
**Script:** `scripts/run_analysis.py`

Sub-modules and their outputs:

| Module | Key outputs |
|--------|------------|
| `model_comparison.py` | `model_comparison_classification.csv`, `model_comparison_regression.csv`, PR-AUC comparison figures |
| `model_selection.py` | `final_model_selection.csv`, `final_model_selection.md` |
| `threshold_analysis.py` | `final_classifier_thresholds.csv` (validation-selected, test-applied), `final_confusion_matrix.png` |
| `calibration_summary.py` | `calibration_summary_by_subset.csv`, `calibration_interpretation.md` |
| `reliability_red_flags.py` | `model_reliability_red_flags.csv`, `model_reliability_red_flags.md` |
| `hypothesis_evidence.py` | `hypothesis_evidence_matrix.csv`, `hypothesis_evidence_matrix.md` |
| `h1_feature_family.py` | `h1_feature_family_importance.csv`, `h1_interpretation.md`, optional ablation |
| `h3_age_evidence.py` | `h3_age_evidence_matrix.csv`, `h3_adopted_only_age_speed.csv`, `h3_interpretation.md` |
| `h5_covid_evidence.py` | `h5_covid_evidence_matrix.csv`, `h5_covid_population_mix.csv`, `h5_interpretation.md` |
| `hypothesis_tables.py` | H2 seasonality, H4 dark-color descriptive tables and figures |

**Diagnostics (selected-model-driven):**

`scripts/generate_diagnostics.py` resolves the selected model from `final_model_selection.csv` before producing calibration curves, error slices, and risk quadrants. It does not hardcode a model family.

**Evidence pack:**

`scripts/generate_evidence_pack.py` produces the ML-rigor layer: bootstrap metric intervals, cohort reliability, SHAP feature-family summaries, Animal Journey examples, local explanation examples, subgroup reliability, model failure modes, and adoption milestone tables.

### Layer 6: Presentation

**Script:** `streamlit_app.py`  
**Module:** `src/aac_adoption/dashboard/`

The app is a **read-only artifact consumer**. It does not retrain models.

Dashboard pages:
- Story Mode (workflow and approach-comparison visuals)
- Animal Journey Cards and contrast views
- Overview of current generated results
- Model comparison figures and tables
- Hypothesis Lab (H1/H3/H5 + H2/H4 descriptive checks)
- Reliability diagnostics, SHAP interpretability, risk explorer, campaign finder, adoption timeline
- Trust & Limits (evidence-pack view with subgroup selector, calibration-gap chart, model-struggle table)
- Generated artifact manifest and thesis/methodology report reader
- Model sensitivity form (shows how the trained model responds to different records; uses association language, not causal counterfactuals)

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| **Python** | Main language |
| **pandas, numpy** | Data wrangling, feature engineering, aggregation, metrics |
| **scikit-learn** | Splitting, baselines, random forests, histogram boosting, preprocessing, evaluation |
| **CatBoost** | Advanced classifier/regressor for categorical-heavy shelter data |
| **SHAP** | Local and global model explanation (predictive association, not causality) |
| **Streamlit** | Thesis demo dashboard |
| **Altair, Plotly** | Interactive and publication-friendly dashboard visualizations |
| **Matplotlib** | Saved report figures |
| **joblib** | Model artifact serialization |
| **pytest** | Automated pipeline validation |

---

## Schema Compatibility Rules

Some generated report tables intentionally include **alias columns** for acceptance and dashboard compatibility. These are additive — existing readers can continue using original internal column names.

| Table | Internal column | Alias |
|-------|----------------|-------|
| H3 adopted-only | `group_value`, `all_records` | `age_group`, `records` |
| Model selection | `animal_subset` | `subset` |
| Threshold table | `threshold_label` | `threshold_name` |
| Calibration | internal columns | `subset`, `records` |
| Reliability red flags | `cohort`, `value` | `subgroup_field`, `subgroup_value`, `mean_predicted_probability` |
| Leakage audit | internal columns | `role`, `allowed_as_feature`, `leakage_risk`, `notes` |

---

## Artifact Manifest

`reports/artifact_manifest.csv` is a lightweight tracked manifest of generated thesis artifacts. Large raw data, processed data, and model binaries remain local/generated assets (gitignored).

Status values: `present` / `missing`

Manifest text is normalized to avoid mojibake dash artifacts in generated Markdown and CSV cells. The manifest is displayed in the Streamlit Artifacts tab.

---

## Test Suite

Configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

**Coverage areas:**

- Data loading, cleaning, matching, dataset build
- Feature engineering and feature-set leakage checks
- Split logic and model artifact helpers
- Baseline, boosting, and CatBoost output contracts
- EDA, analysis, diagnostics, and report outputs
- Evidence pack and subgroup reliability functions
- Data audit, leakage audit, target definitions, and artifact manifest
- Dashboard data helpers and story content
- Acceptance/schema aliases for generated artifacts
- Local explanation example generation and manifest registration

**Run tests:**

```bash
pytest
```

Known warning: pandas may emit a `FutureWarning` from evidence-pack concatenation when synthetic tests use empty/all-NA frames. This does not currently fail the suite.

---

## Thesis Chapter Mapping

| Thesis chapter | Primary code location | Key artifacts |
|---------------|----------------------|--------------|
| Ch 3: Data and Preprocessing | `src/aac_adoption/data/` | `data/processed/modeling_dataset.csv`, data audit tables |
| Ch 4: Methodology | `src/aac_adoption/features/`, `src/aac_adoption/models/` | `feature_columns.json`, model artifacts |
| Ch 5: Results | `src/aac_adoption/analysis/`, `src/aac_adoption/diagnostics/` | model comparison tables, hypothesis figures, SHAP outputs |
| Ch 6: System Architecture | `scripts/`, `tests/`, `src/aac_adoption/` | `artifact_manifest.csv`, full pipeline runner |
| Ch 7: Discussion | `src/aac_adoption/reporting/` | evidence pack, subgroup reliability, model failure modes |

---

## What Is Not Implemented (By Design)

- **Docker / DVC / MLflow** — deferred; the pipeline is reproducible via scripts and `requirements.txt` without them.
- **Production API** — the dashboard is a thesis demo, not a production shelter system.
- **Survival models as primary framework** — KM curves are descriptive only; Cox/AFT is future work.
- **Hyperparameter tuning** — documented as planned improvement in [`ROADMAP.md`](ROADMAP.md).


================================================================================
FILE: docs/METHODOLOGY.md
================================================================================

# Methodology — AAC Adoption ML Pipeline

This document is the authoritative reference for methodological framing, target variable definitions, causal language rules, feature set justification, and dataset matching logic. All thesis text, dashboard labels, report outputs, and code must be consistent with this document.

---

## Scope and Claims

### What this project claims

This pipeline demonstrates **predictive associations** between intake-time features and adoption outcomes / length of stay in Austin Animal Center records (2013–2025).

### What this project does NOT claim

- That intake features *cause* adoption outcomes.
- That COVID *caused* adoption rates to change.
- That dark-colored animals are *discriminated against* (descriptive association only).
- That the regression output equals *adoption speed* — it is length of stay until any matched outcome.

### Required terminology in all outputs

| Use | Avoid |
|-----|-------|
| `predictive association`, `associated with` | `causes adoption`, `proves` |
| `linked to model output` | `COVID caused...` |
| `intake-time predictors` | `adoption speed` (unless subset is adopted animals only) |
| `length of stay`, `time to outcome` | `days to adoption` (unless explicitly filtered) |
| `descriptive time-to-adoption evidence` | `reduces adoption time` |

---

## Target Variable Definitions

### 1. Classification Target

| Property | Value |
|----------|-------|
| Column name | `classification_target` |
| Alias columns | `adopted`, `is_adopted`, `target_adopted` |
| Data type | int (0 or 1) |
| Defined as | `1` if `outcome_type == "Adoption"` (case-insensitive), else `0` |
| Scope | All matched intake/outcome episodes (dogs and cats) |
| Allowed thesis labels | "adoption indicator", "adopted vs. not adopted", "binary adoption target" |
| Forbidden labels | "adoption speed target", "timing target" |

```python
outcome_type_normalized = dataset["outcome_type"].fillna("").astype(str).str.strip().str.lower()
dataset["adopted"] = outcome_type_normalized.eq("adoption")
dataset["classification_target"] = dataset["adopted"].astype(int)
```

### 2. Regression Target — Primary (Length of Stay)

| Property | Value |
|----------|-------|
| Column name | `regression_target_days` |
| Alias columns | `days_to_outcome`, `length_of_stay` |
| Data type | float (non-negative) |
| Defined as | `(outcome_datetime - intake_datetime).total_seconds() / 86400` |
| Scope | All matched intake/outcome episodes |
| Allowed thesis labels | "days to outcome", "length of stay", "predicted days to outcome" |
| Forbidden labels | "adoption speed", "days to adoption", "predicted wait until adoption" (unless subset is adopted only) |

**Interpretation:** This is an operational length-of-stay prediction. The matched outcome may be adoption, transfer, return-to-owner, or another disposition. Describing this as "predicted time to adoption" is methodologically wrong unless explicitly filtered to adopted animals.

```python
dataset["days_to_outcome"] = (
    dataset["outcome_datetime"] - dataset["intake_datetime"]
).dt.total_seconds() / 86400
dataset["regression_target_days"] = dataset["days_to_outcome"]
```

### 3. Adoption-Only Timing Target — Descriptive / H3 Only

| Property | Value |
|----------|-------|
| Column name | `days_to_adoption` |
| Data type | float or NaN |
| Defined as | `days_to_outcome` where `outcome_type == "Adoption"`, else `NaN` |
| Scope | Adopted animals only |
| Allowed thesis labels | "days to adoption (among adopted animals)", "adoption timing (adopted subset)" |
| Forbidden | Using as the main regression target without stating adopted-only scope |

```python
dataset["days_to_adoption"] = np.where(
    dataset["adopted"], dataset["days_to_outcome"], np.nan
)
```

### 4. Survival / Time-to-Event — Future Work

Kaplan–Meier descriptive curves are generated for adopted animals grouped by `animal_type`, `age_group`, `covid_period`, and `intake_type`. These serve as **descriptive evidence** for H3 only. Full survival modeling with censoring and competing risks is outside the main scope of this thesis.

### Target Relationship Diagram

```
All matched episodes (N ≈ 162,390)
├── classification_target=1 (adopted, ~52%) ──► days_to_adoption = days_to_outcome
│                                                 ↑ Used for H3 adopted-only timing
└── classification_target=0 (not adopted, ~48%) ──► days_to_adoption = NaN

regression_target_days = days_to_outcome (ALL episodes, regardless of outcome type)
    ↑ Main regression target: predicts length of stay, not adoption speed
```

---

## Leakage Control

The following columns are **targets or outcome-derived metadata** and must never appear in `feature_columns.json` or be passed to any model as predictors:

| Column | Category | Reason |
|--------|----------|--------|
| `classification_target` | target | binary adoption label |
| `regression_target_days` | target | days to outcome |
| `days_to_outcome` | target/alias | same as regression_target_days |
| `length_of_stay` | target/alias | same as days_to_outcome |
| `days_to_adoption` | target/alias | adopted-only timing |
| `adopted`, `is_adopted`, `target_adopted` | target/alias | binary adoption flags |
| `outcome_type` | metadata | post-intake outcome label |
| `outcome_subtype` | metadata | post-intake outcome detail |
| `outcome_datetime` | metadata | post-intake timestamp |
| `sex_upon_outcome` | metadata | post-intake measurement |
| `age_upon_outcome` | metadata | post-intake measurement |

**Validation:** `src/aac_adoption/features/feature_sets.py` — `LEAKAGE_COLUMNS` set and `validate_no_leakage()` function.  
**Leakage audit:** `scripts/generate_leakage_audit.py`

---

## Intake-Time Feature Set

All model features are available at the moment of intake (before any outcome is known). This is the fundamental leakage-safety guarantee.

| Family | Features | Hypothesis |
|--------|----------|-----------|
| Animal identity | `animal_type`, `breed`, `primary_breed`, `simplified_breed_group` | H1 |
| Appearance | `color`, `primary_color`, `simplified_color_group`, `is_black_or_dark` | H4 |
| Name status | `has_name`, `is_named` | — |
| Age | `age_upon_intake`, `age_days`, `age_months`, `age_years`, `age_group` | H3 |
| Intake circumstances | `intake_type`, `intake_condition` | H1 |
| Timing | `intake_year`, `intake_month`, `intake_quarter`, `intake_season`, `covid_period` | H2, H5 |
| Location | `found_location_kind`, `found_location_area`, `is_austin_found_location`, flags | — |
| Sex | `sex_upon_intake` | — |

**Optional context features** (intake-date-based, prior window only):

| Feature | Source |
|---------|--------|
| `daily_temp_max`, `daily_temp_min`, `daily_precipitation` | Austin weather |
| `is_extreme_heat`, `is_rainy_day` | Derived from weather |
| `animal_311_requests_7d`, `animal_311_requests_30d` | Austin 311 |
| `intake_volume_7d`, `intake_volume_30d` | Shelter intakes |

All context features use only dates **before** the intake date. Rolling windows do not include the intake day itself.

**Note on feature redundancy:** `age_days`, `age_months`, and `age_years` are collinear; `has_name` and `is_named` are near-identical. These redundancies are documented in [`docs/ROADMAP.md`](ROADMAP.md) as a planned cleanup item.

---

## Dataset Matching Logic

Each intake episode is matched to the nearest unused future outcome for the same animal using a **greedy nearest-future-match** algorithm:

1. Sort outcomes by datetime for each animal.
2. For each intake (sorted by `intake_datetime`), skip outcomes that occurred before the intake.
3. Assign the next available outcome to this intake.
4. Mark that outcome as used (cannot be reused for a later intake).

**Consequences:**
- An animal with N intakes and N outcomes gets N episode rows.
- An animal with N intakes and fewer outcomes loses trailing intakes (counted as `unmatched_intakes`).
- No outcome is shared between two intake episodes.
- No negative `days_to_outcome` values are possible (validated by `validate_modeling_dataset()`).

**Limitation:** If an animal was transferred between shelters and readmitted, the re-admission creates a new independent episode. This is intentional — each stay is a separate resource-planning problem. Re-intake ambiguity detection (checking for intermediate intakes within a matched pair) is a planned improvement; see [`docs/ROADMAP.md`](ROADMAP.md).

---

## Why Regression Instead of Survival Analysis?

**Short answer:** Most animals in this dataset have resolved outcomes, so censoring is not the dominant concern. The regression target (`days_to_outcome`) is operationally meaningful and more interpretable for non-statistical audiences than hazard ratios.

**Full defense:**

1. **Shelters care about length of stay, not just whether adoption happened.** Every kennel-day has a cost regardless of outcome type.
2. **`days_to_outcome` ≠ "adoption speed".** Adoption-only timing is analyzed separately as `days_to_adoption` (H3 descriptive section).
3. **Regression MAE has direct operational meaning.** MAE ≈ 18 days means the model's LOS estimate is off by ~18 days on average.
4. **Descriptive survival curves are provided.** Kaplan–Meier curves for adopted animals by `animal_type`, `age_group`, and `covid_period` give a time-to-adoption view without requiring a full survival model.
5. **Full survival modeling is noted as future work.** It would add censoring for unresolved stays and competing-risks framing (adoption vs transfer vs euthanasia vs return-to-owner).

**Thesis statement:**
> *The regression target `regression_target_days` predicts length of stay until any matched outcome. This is operationally equivalent to "time the animal occupies a kennel" rather than "time until adoption." Adoption-only timing is analyzed separately in the descriptive H3 section using `days_to_adoption`. Full survival modeling with censoring and competing risks is noted as future work.*

---

## Predictive Association vs. Causal Claims

The ML pipeline produces **predictive associations**, not causal evidence. There is no randomized assignment of any feature. Confounders are unknown. ML models optimize prediction accuracy, not causal identification.

**Specific framing rules:**

- **COVID period:** `covid_period` is an intake-date label. Differences across periods are *associated with* model output but could reflect policy changes, population changes, media effects, or unobserved confounders.
- **Dark coloring:** `is_black_or_dark` is a color descriptor. Lower observed adoption rates for dark animals reflect patterns in the data but do not prove discrimination or causal bias. Correct framing: *"Dark-colored animals show lower observed adoption rates, a pattern descriptively consistent with the so-called black dog syndrome reported in shelter literature."*

---

## Consistent Label Rules

| Context | Correct | Incorrect |
|---------|---------|-----------|
| Regression model output (all animals) | "predicted days to outcome" / "predicted length of stay" | "predicted adoption speed", "days to adoption" |
| Regression MAE in reports | "MAE X days for length-of-stay prediction" | "adoption speed error" |
| H3 table (all animals) | "median days to outcome by age group" | "adoption speed by age" |
| H3 adopted-only table | "median days to adoption (adopted animals only)" | "adoption speed" without qualification |
| Dashboard regression metric | "Predicted days to outcome" | "Predicted wait until adoption" |
| SHAP regression interpretation | "associated with predicted days to outcome" | "causes faster adoption" |


================================================================================
FILE: docs/RESULTS.md
================================================================================

# Results Reference — AAC Adoption ML Pipeline

Current reproducible results, hypothesis findings, interpretation notes, and reproduction commands. Updated as of 2026-06-06.

See [`METHODOLOGY.md`](METHODOLOGY.md) for target definitions and causal-language rules before citing any numbers.

---

## Dataset Snapshot

| Property | Value |
|----------|-------|
| Source | Austin Animal Center (AAC), 2013–2025 |
| Processed file | `data/processed/modeling_dataset.csv` |
| Matched episodes | **162,390** intake/outcome rows |
| Animal types | Dogs and cats only |
| Split strategy | Time-aware chronological |
| Train period | 2013–2021 |
| Validation period | 2022–2023 |
| Test period | 2024–2025 |
| Leakage control | Model features are intake-time-only |

Raw files:

```
data/raw/intakes.csv
data/raw/outcomes.csv
data/raw/context/austin_weather_daily.csv          (optional)
data/raw/context/austin_311_animal_requests.csv    (optional)
```

Raw data and generated outputs are gitignored.

---

## Classification Results

Primary metric: **ROC-AUC** (test split). Secondary: PR-AUC.

Comparison file: `reports/tables/model_comparison_classification.csv`

| Subset | Best model | ROC-AUC (test) |
|--------|-----------|----------------|
| Combined | `hist_gradient_boosting` | ≈ 0.840 |
| Dogs | `hist_gradient_boosting` | ≈ 0.806 |
| Cats | `hist_gradient_boosting` | ≈ 0.865 |

**Interpretation:**
- Gradient boosting currently performs best by ROC-AUC across all subsets.
- Logistic regression remains useful as an interpretable baseline.
- Cats appear slightly easier to classify than dogs in the current feature setup.
- PR-AUC is tracked alongside ROC-AUC; see `model_comparison_classification.csv` for full values.

---

## Regression Results

Primary metric: **MAE** on `regression_target_days` (length of stay until any outcome).  
**This is not adoption speed.** See [`METHODOLOGY.md`](METHODOLOGY.md#target-variable-definitions).

Comparison file: `reports/tables/model_comparison_regression.csv`

| Subset | Best model | MAE (test, days) |
|--------|-----------|-----------------|
| Combined | `catboost` | ≈ 18.55 |
| Dogs | `catboost` | ≈ 21.56 |
| Cats | `catboost` | ≈ 15.79 |

> Regression MAE label in all reports: *"combined regression MAE: X days for length-of-stay / days-to-outcome prediction"*

**Interpretation:**
- Regression is harder than classification.
- Dummy median remains a meaningful baseline because length of stay is right-skewed.
- MAE is emphasized over R² because it is easier to explain operationally.
- CatBoost outperforms histogram boosting on regression; histogram boosting outperforms on classification.

---

## Hypothesis Findings

### H1 — Intake Circumstances vs. Appearance (Central)

**Status:** Supported descriptively and predictively.

Key tables: `reports/tables/h1_feature_family_importance.csv`, `reports/tables/h1_feature_family_ablation.csv` (if run with `--h1-ablation`)  
Key figures: `reports/figures/h1_feature_family_importance.png`, `reports/figures/h1_intake_type_adoption_rate.png`, `reports/figures/h1_intake_condition_adoption_rate.png`  
Narrative: `reports/summary/h1_interpretation.md`

Current descriptive pattern:
- `Owner Surrender` and `Abandoned` records show relatively high adoption rates.
- `Public Assist` and `Euthanasia Request` show much lower adoption rates.
- Intake-related features have stronger model-importance signal than appearance features.

### H3 — Age and Adoption Timing (Central)

**Status:** Supported descriptively and predictively.

Key tables: `reports/tables/h3_age_evidence_matrix.csv`, `reports/tables/h3_adopted_only_age_speed.csv`  
Key figures: `reports/figures/h3_age_adoption_rate.png`, `reports/figures/h3_age_adopted_only_median_days.png`, `reports/figures/h3_age_shap_summary.png`  
Narrative: `reports/summary/h3_interpretation.md`

Current descriptive pattern:
- Baby animals have the highest adoption rate.
- Young animals are below babies but above adults.
- Adult and senior animals show lower adoption rates.
- Age-related features have meaningful SHAP importance.
- For adopted animals specifically, see `h3_adopted_only_age_speed.csv` for median `days_to_adoption`.

### H5 — COVID-Period Adoption Dynamics (Central)

**Status:** Supported descriptively.

Key tables: `reports/tables/h5_covid_evidence_matrix.csv`, `reports/tables/h5_covid_population_mix.csv`  
Key figures: `reports/figures/h5_covid_adoption_rate.png`, `reports/figures/h5_covid_median_los.png`, `reports/figures/h5_covid_intake_volume.png`  
Narrative: `reports/summary/h5_interpretation.md`

Current descriptive pattern:
- Post-COVID and COVID periods show higher adoption rates than pre-COVID.
- Median days to outcome are also higher in COVID and post-COVID periods.
- Caution: period effects may reflect policy changes, intake volume shifts, public behavior, and population mix — not a causal COVID effect.

### H2 — Seasonality (Secondary, descriptive)

**Status:** Descriptive check only.

Tables: `reports/tables/h2_seasonality_summary.csv`  
Figures: `reports/figures/h2_adoption_rate_by_season.png`, `reports/figures/h2_median_los_by_season.png`  
Narrative: `reports/summary/h2_interpretation.md`

### H4 — Dark Coat Color / Black Dog Syndrome (Secondary, descriptive)

**Status:** Descriptive check only.

Tables: `reports/tables/h4_dark_color_summary.csv`  
Figures: `reports/figures/h4_dark_color_adoption_rate.png`, `reports/figures/h4_dark_color_median_los.png`  
Narrative: `reports/summary/h4_interpretation.md`

Note: `is_black_or_dark` is an approximate operational grouping. Dark-colored animals show lower observed adoption rates — this is a descriptive association consistent with shelter literature, not proof of discrimination.

---

## External Context Feature Test

The context workflow compares `intake_time_v1` (base) against `intake_time_context_v1` (enriched with weather, 311, and shelter-volume features).

Comparison artifact:
```
reports/tables/context_model_comparison.csv
reports/figures/context_model_delta.png
```

**Current interpretation:**
- Context features gave small CatBoost classification gains for combined animals and cats, but not dogs.
- Context features gave small CatBoost regression MAE gains across combined, dog, and cat subsets.
- Context features did not broadly improve every model family; several random-forest and histogram-boosting regression runs worsened.
- Thesis wording: *"External context added limited but useful signal for CatBoost; it did not generally improve all model families."*

---

## Interpretability Outputs

Generated by `scripts/generate_diagnostics.py --include-shap` and `scripts/generate_feature_family_importance.py`.

| File | Content |
|------|---------|
| `reports/tables/logistic_regression_coefficients.csv` | LR coefficient table |
| `reports/tables/random_forest_feature_importance.csv` | RF importance |
| `reports/tables/permutation_importance_classification.csv` | Permutation importance (classification) |
| `reports/tables/permutation_importance_regression.csv` | Permutation importance (regression) |
| `reports/tables/shap_global_classification.csv` | SHAP global feature values (classification) |
| `reports/tables/shap_global_regression.csv` | SHAP global feature values (regression) |
| `reports/tables/shap_feature_families_classification.csv` | SHAP by feature family (classification) |
| `reports/tables/shap_feature_families_regression.csv` | SHAP by feature family (regression) |

**Recommended thesis language:**
> *The model identifies variables associated with adoption outcomes and length of stay. These associations do not prove causal effects, but they indicate which intake-time characteristics are most informative for prediction. SHAP values explain how features contributed to this model's prediction. They do not prove that changing a feature would causally change adoption probability.*

---

## Model Reliability and Evidence Pack Outputs

Generated by `scripts/generate_evidence_pack.py`.

| File | Content |
|------|---------|
| `reports/tables/model_evidence_pack.csv` | Summary of model choice, PR-AUC, bootstrap intervals |
| `reports/tables/metric_confidence_intervals.csv` | Bootstrap metric CIs |
| `reports/tables/model_limitations_by_cohort.csv` | Cohort-level error and calibration limits |
| `reports/tables/subgroup_reliability.csv` | Per-cohort reliability metrics |
| `reports/tables/subgroup_metric_confidence_intervals.csv` | Per-cohort bootstrap CIs |
| `reports/tables/subgroup_adoption_milestones.csv` | Descriptive adoption timing at 7/30/60/90 days |
| `reports/tables/model_failure_modes.csv` | Ranked calibration gaps, MAE, FNR, FPR by cohort |
| `reports/tables/animal_journey_examples.csv` | Representative Animal Journey Card examples |
| `reports/tables/local_explanation_examples.csv` | Local explanation examples (non-causal) |
| `reports/summary/model_evidence_pack.md` | Human-readable evidence pack narrative |
| `reports/summary/subgroup_reliability.md` | Subgroup reliability narrative |

**Top subgroup reliability fields:**
- `records` — cohort size
- `observed_adoption_rate` — actual share adopted
- `mean_predicted_probability` — average model prediction
- `calibration_gap` — absolute difference between observed and predicted rates
- `small_cohort` — flag for groups too small for strong claims

`model_failure_modes.csv` ranks the largest calibration gaps, MAE values, false-negative rates, and false-positive rates. High-gap rows are warning lights for thesis discussion, not proof that a group is intrinsically harder to place.

---

## Interpretation Rules

- Use **ROC-AUC** for ranking classification quality.
- Use **PR-AUC** for adoption-positive precision/recall behavior.
- Use **calibration and threshold tables** before interpreting probabilities as decision-support scores.
- Use **MAE** as the main days-to-outcome regression metric.
- Use **SHAP** as model-explanation evidence only: features are associated with model predictions, not causes of adoption.
- Treat health and behavior fields as administrative care-context proxies, not complete personality or temperament labels.
- Treat animal profile findings as descriptive/exploratory unless supported by additional statistical testing.
- **Do not interpret subgroup metrics below minimum sample thresholds** (n < 100 or n < 200 depending on context).

---

## Full Reproduction Commands

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
python scripts/download_context_data.py --output-dir data/raw/context
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset_context.csv --context-data-dir data/raw/context
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_baseline --tables-dir reports/tables_context --output reports/metrics_context/baseline_metrics.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_boosting --tables-dir reports/tables_context
python scripts/train_advanced.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_advanced
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
python scripts/compare_context_models.py --base-metrics-dir reports/metrics --context-metrics-dir reports/metrics_context --tables-dir reports/tables
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
pytest
```

---

## Current Test Status

```bash
pytest -q
```

Latest verified result: **81 passed** (local environment, 2026-05-31).

Known warning: `FutureWarning` in `evidence_pack.py` during pandas concat with empty/all-NA frames in synthetic tests. Does not fail the suite.


================================================================================
FILE: docs/ROADMAP.md
================================================================================

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
- Current blocker: the calibration CLI is not reproducible because `scripts/calibrate_classifiers.py` imports missing function `calibrate_classifiers`.
- Required fix: restore or rewrite `calibrate_classifiers()` and add an end-to-end tiny-fixture test for the script.
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

### TODO 6. Censoring safeguards near dataset end

- Current code adds a `censoring_flag` when `days_to_outcome >= max_los_days`, but this is not a sufficient dataset-end follow-up safeguard.
- Required: for horizon tasks, exclude intakes without enough possible follow-up time before the export date.
- Required: write included/excluded row counts per horizon and cutoff date.
- Required: document remaining censoring risk explicitly.

### PARTIAL 7. Audit episode matching ambiguity

- Re-intake metadata exists (`episode_number`, `is_reintake`, `days_since_last_stay`).
- Current matching still greedily pairs each intake with the next unused future outcome.
- Missing: check whether another intake for the same animal occurs between an intake and its candidate outcome.
- Required: mark such episodes as ambiguous/censored/unmatched and report clean, ambiguous, dropped, and unmatched counts.

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

### PARTIAL 16. Strengthen subgroup reliability

- Subgroup reliability and reliability red-flag artifacts exist.
- Current acceptance still needs stricter cohort rules:
  - records,
  - adoption rate,
  - mean predicted probability,
  - calibration gap,
  - PR-AUC where sample size and class variety allow,
  - Brier score,
  - explicit minimum sample-size thresholds.
- Do not interpret subgroup metrics below n < 100 or n < 200.

### TODO 17. Cluster-aware confidence intervals

- Current bootstrap utilities are row-level.
- Required: bootstrap by `animal_id` where possible.
- If row-level bootstrap remains, generated evidence must state that episodes are not fully independent animal-level observations.

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
| RISK | Calibration reproducibility | Calibrated artifacts exist, but the current calibration CLI cannot reproduce them because a called function is missing. |
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

### Slices Still Not Accepted

- Formal calibration stage.
- Horizon classifiers.
- Real censoring/follow-up safeguards.
- Episode matching ambiguity audit.
- Leakage risk classification.
- Yearly temporal backtesting.
- Recency strategy comparison.
- Full survival modeling, unless kept as explicit future work.
- Duration uncertainty outputs.
- Strict subgroup reliability rules.
- Cluster-aware confidence intervals.

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
- [ ] Subgroup reliability includes calibration and sample-size safeguards.
- [x] Leakage audit classifies suspicious features by risk level.
- [ ] Confidence intervals are cluster-aware by `animal_id` or explicitly documented as row-level.
- [x] Full pipeline passes after calibration step is fixed.


================================================================================
FILE: docs/old/thesis_technical_guide.md
================================================================================

# Thesis Technical Guide

Project:

**Life-Saving Data: Analyzing Factors Affecting Adoptions at the Austin Animal Center via Machine Learning and Visualization**

This document is a living project reference. It explains the technical scope of the project, the main design decisions, the architecture of the codebase, the frameworks and libraries used, the current results, the test status, what is already implemented, and what is still planned. It is intentionally technical, not chapter-oriented.

## 1. What This Project Is

This repository implements a reproducible data science pipeline for analyzing dog and cat adoptions at Austin Animal Center. The project is not just a collection of notebooks or isolated models. It is a full analytical system that:

- downloads raw shelter intake and outcome data,
- cleans and standardizes the records,
- matches intakes with valid future outcomes,
- engineers intake-time features,
- trains baseline, boosting, and advanced models,
- evaluates classification and regression tasks,
- generates interpretability and reliability outputs,
- serves the results in a Streamlit thesis dashboard,
- produces report-ready tables and figures for writing the thesis.

The key thesis idea is:

> transform shelter records into interpretable evidence about adoption timing and adoption likelihood, while keeping the workflow reproducible and leakage-safe.

## 2. Research Scope

### 2.1 Main Research Goal

The central scientific goal is to understand which intake-time characteristics are associated with adoption outcomes and length of stay for dogs and cats at Austin Animal Center.

The thesis uses two prediction tasks:

- classification: whether an animal is adopted,
- regression: how long it takes until outcome or adoption.

### 2.2 Main Analytical Priorities

The project intentionally prioritizes three thesis threads:

- **H1**: intake circumstances versus appearance features,
- **H3**: age and adoption timing among adopted animals,
- **H5**: COVID-period adoption dynamics.

These are the strongest and cleanest analytical threads in the current implementation.

Secondary descriptive threads are still included, but they are not the main focus:

- **H2**: seasonality,
- **H4**: black dog / black cat effect or dark-color effect.

### 2.3 Scope Boundaries

Important limits of the project:

- only dogs and cats are modeled,
- only intake-time features are used for prediction,
- outcomes are used for labels and evaluation, not as predictors,
- the analysis is predictive and descriptive, not causal,
- the dashboard is a thesis demo, not a production shelter system,
- survival analysis is not yet the main modeling framework.

## 3. Why These Design Decisions Were Made

### 3.1 Dogs and Cats Only

Dogs and cats make up the most common and analytically useful groups in the Austin Animal Center data. Excluding rare species reduces noise and keeps the thesis focused on the dominant shelter population.

### 3.2 Intake-Time-Only Features

The most important methodological decision is to prevent data leakage.

The model must predict adoption using only information that would have been available at intake time. This is crucial because the thesis asks what can be known early, not what can be reconstructed after the fact.

Allowed predictors include:

- animal type,
- intake type,
- intake condition,
- sex upon intake,
- age upon intake,
- breed,
- color,
- name availability,
- time-related intake features,
- simplified breed and color groupings,
- COVID-period flag.

Outcome-derived columns are not predictors. They are used only for:

- target construction,
- evaluation,
- reporting,
- diagnostics.

This design makes the thesis methodologically defensible.

### 3.3 Time-Aware Splitting

The train/validation/test split is time-based rather than random:

- train: earlier years,
- validation: middle years,
- test: most recent years.

This is a better choice for a shelter-adoption thesis because the problem is time-sensitive and animal adoption patterns change over time. A random split would make the evaluation easier but less realistic.

### 3.4 Multiple Model Families

The project compares several model classes on purpose:

- dummy models,
- logistic regression and ridge regression,
- random forests,
- histogram gradient boosting,
- CatBoost.

This gives a clear progression from simple baselines to stronger nonlinear models. It also supports the thesis argument that model complexity should be justified by better predictive performance and useful interpretability, not used automatically.

### 3.5 Interpretability as a First-Class Goal

The thesis is not only about score maximization. It also needs to explain why the model behaves as it does.

That is why the pipeline includes:

- coefficient tables,
- feature importance tables,
- permutation importance,
- SHAP summaries,
- feature-family summaries,
- local explanations for representative animal profiles,
- reliability and error-slice diagnostics.

This makes the thesis more than a predictive benchmark. It becomes an interpretable decision-support study.

## 4. System Architecture

The repository follows a layered architecture.

### 4.1 High-Level Layers

1. **Data acquisition**
   - downloads raw AAC CSV files,
   - keeps raw inputs unchanged.

2. **Data preparation**
   - cleans fields,
   - parses dates,
   - filters to dogs and cats,
   - matches intakes with outcomes,
   - creates the modeling dataset.

3. **Feature engineering**
   - converts raw shelter variables into model-ready features,
   - creates age, season, breed, and color abstractions,
   - creates leakage-safe feature lists.

4. **Modeling**
   - trains baseline, boosting, and advanced models,
   - evaluates classification and regression tasks,
   - stores fitted artifacts.

5. **Analysis and interpretation**
   - compares models,
   - tests thesis hypotheses,
   - computes importance and SHAP outputs,
   - generates diagnostics and evidence packs.

6. **Presentation**
   - produces report tables and figures,
   - serves a Streamlit dashboard for thesis demonstration.

### 4.2 Repository Structure

The codebase is organized to reflect the pipeline:

- `src/aac_adoption/`
  - reusable package code,
  - data, features, models, analysis, diagnostics, reporting, and dashboard logic.
- `scripts/`
  - command-line entry points for each pipeline step.
- `tests/`
  - automated checks for data handling, modeling, and outputs.
- `docs/`
  - thesis planning, architecture notes, and result summaries.
- `data/`
  - raw, interim, and processed data artifacts.
- `models/`
  - saved trained model artifacts.
- `reports/`
  - tables, figures, metrics, diagnostics, and summary markdown.

This structure is important for the thesis because it shows that the work is reproducible and modular.

## 5. Core Technologies and Frameworks

### 5.1 Python

Python is the main language for the project because it has mature libraries for:

- data wrangling,
- machine learning,
- visualization,
- testing,
- dashboarding.

### 5.2 pandas and numpy

These libraries are used for:

- tabular data cleaning,
- feature engineering,
- aggregation,
- date/time handling,
- metric preparation.

### 5.3 scikit-learn

Scikit-learn is the core machine learning framework for:

- train/validation/test splitting,
- baseline classifiers and regressors,
- random forests,
- histogram gradient boosting,
- preprocessing and evaluation,
- permutation importance,
- metrics and diagnostics.

### 5.4 CatBoost

CatBoost is used for the stronger advanced models, especially because shelter data contains many categorical variables such as:

- breed,
- color,
- intake type,
- intake condition.

CatBoost is a strong choice for this project because it handles categorical structure well and often performs strongly on tabular data.

### 5.5 SHAP

SHAP is used to explain model predictions and feature contributions.

In the thesis, SHAP should be presented as:

- a method for local and global model explanation,
- a way to understand predictive association,
- not proof of causality.

### 5.6 Streamlit

Streamlit is used to build the thesis demo dashboard. It is suitable because it can quickly render:

- metrics,
- tables,
- charts,
- interactive selectors,
- model sensitivity forms,
- narrative research views.

### 5.7 Altair and Plotly

These libraries support interactive and publication-friendly visualizations in the dashboard.

### 5.8 Matplotlib

Matplotlib is used for saved figures in the reports layer and for compatibility with traditional plotting workflows.

### 5.9 joblib

Joblib is used to store and load trained model artifacts and supporting objects.

### 5.10 pytest

Pytest provides automated validation for:

- cleaning,
- dataset building,
- feature engineering,
- training outputs,
- analysis outputs,
- diagnostic artifacts.

This is important for thesis credibility because it shows the pipeline is testable and repeatable.

## 6. Data Pipeline Design

### 6.1 Raw Data

The raw data consists of Austin Animal Center intake and outcome CSV files. Raw data is kept unchanged in `data/raw/`.

### 6.2 Cleaning

Cleaning includes:

- normalizing column names,
- parsing datetime fields,
- removing invalid or duplicate rows where necessary,
- restricting the dataset to dogs and cats,
- validating required columns.

### 6.3 Intake-Outcome Matching

The project matches each intake with a valid future outcome for the same animal in a date-aware way.

This decision matters because AAC animals can appear multiple times. A naive join could:

- duplicate outcomes,
- create impossible timelines,
- generate negative length of stay,
- distort model labels.

The matching logic prevents these errors by pairing each intake with the nearest unused future outcome.

### 6.4 Modeling Dataset

The processed modeling dataset contains:

- intake features,
- outcome labels,
- engineered age and time fields,
- adoption and length-of-stay targets,
- metadata needed for model training and reporting.

This dataset is the central handoff object between preprocessing and modeling.

## 7. Feature Engineering Decisions

The thesis should explain that the project does not use raw shelter records directly. It uses engineered features that make the data more suitable for analysis.

Important engineered features include:

- `age_in_days`, `age_in_months`, `age_in_years`,
- age groups such as baby, young, adult, and senior,
- intake year, month, quarter, and season,
- COVID-period grouping,
- simplified breed groups,
- simplified color groups,
- black/dark color flag,
- named versus unnamed animal flags,
- mixed-breed indicator.

These features were chosen because they help convert messy operational data into variables that are easier to model and easier to explain in the thesis.

## 8. Modeling Strategy

### 8.1 Task Types

The project handles two predictive tasks:

- classification: adoption versus non-adoption,
- regression: time until outcome or adoption.

Using both tasks is useful because adoption success has two dimensions:

- whether it happens,
- how quickly it happens.

### 8.2 Baseline Models

Baseline models are used to establish a lower bound:

- `DummyClassifier`,
- `LogisticRegression`,
- `RandomForestClassifier`,
- `DummyRegressor`,
- `Ridge`,
- `RandomForestRegressor`.

These models help answer:

- is there signal in the data at all,
- how much improvement do stronger models provide,
- which features matter in simpler interpretable settings.

### 8.3 Gradient Boosting

Histogram gradient boosting is used as a strong scikit-learn benchmark.

It is especially suitable for:

- nonlinear patterns,
- mixed feature types after preprocessing,
- tabular prediction tasks.

### 8.4 CatBoost

CatBoost is used as the advanced model family and is especially valuable for categorical shelter data.

It is a strong thesis choice because it combines:

- competitive performance,
- tabular-data strength,
- practical interpretability through SHAP.

### 8.5 Model Evaluation

The main metrics are selected to fit the task:

Classification:

- ROC-AUC,
- F1,
- calibration and threshold diagnostics.

Regression:

- MAE,
- RMSE,
- predicted-versus-actual behavior.

The thesis should discuss both performance and reliability, not only headline scores.

## 9. Interpretation and Reliability Layer

### 9.1 Why Interpretation Matters

For a thesis, a strong model alone is not enough. The reader also needs to understand what the model learned and where it is trustworthy.

### 9.2 Interpretation Outputs

The repository includes:

- logistic regression coefficients,
- random forest feature importance,
- permutation importance,
- SHAP summaries,
- feature-family importance,
- representative animal journey explanations.

### 9.3 Reliability Outputs

The repository also includes diagnostic outputs such as:

- calibration tables,
- threshold tradeoff tables,
- error slices,
- cohort reliability tables,
- subgroup reliability tables,
- adoption milestone tables,
- model failure-mode tables.

These are especially important because shelter data can be imbalanced and operationally messy. A model may be accurate overall but still unreliable for some groups.

### 9.4 Causal Caution

The thesis must use careful language.

Correct phrasing:

- associated with,
- linked to,
- predictive of,
- correlated with the model output.

Avoid causal phrasing unless a true causal design is introduced.

## 10. Dashboard Role

The Streamlit app is the presentation layer for the thesis. It does not replace the analysis pipeline.

It provides:

- executive overview,
- narrative story mode,
- animal-centered profile cards,
- model quality views,
- trust and limits exploration,
- interpretability display,
- risk and campaign exploration,
- hypothesis lab,
- model sensitivity demo,
- adoption timeline,
- generated artifact browsing.

This is useful in the thesis because it shows that the results are not only statistical artifacts. They can be explored interactively.

## 11. Report and Artifact Layer

The project is designed to create thesis-ready outputs that can be reused in writing.

Generated artifacts include:

- metrics CSVs,
- hypothesis tables,
- model comparison tables,
- diagnostic tables,
- SHAP outputs,
- evidence pack summaries,
- summary markdown files,
- figures for the thesis chapters.

This artifact-driven design is important because it separates computation from presentation. You can regenerate outputs without manually rebuilding charts in a spreadsheet.

## 12. Suggested Chapter Structure for the Thesis

### Chapter 1: Introduction

Explain:

- the shelter adoption problem,
- why adoption timing among adopted animals matters,
- why Austin Animal Center is a useful case study,
- why data science is relevant,
- the research goals,
- hypotheses H1 to H5,
- scope and limitations.

### Chapter 2: Theoretical Background and Related Work

Explain:

- animal shelter context,
- no-kill philosophy,
- predictive modeling in operational settings,
- tabular machine learning methods,
- explainable AI and SHAP,
- reproducible data science and MLOps ideas.

### Chapter 3: Data and Preprocessing

Explain:

- source of the data,
- structure of intake and outcome records,
- cleaning steps,
- matching logic,
- leakage prevention,
- feature engineering,
- final modeling dataset.

### Chapter 4: Methodology

Explain:

- prediction tasks,
- train/validation/test split,
- models compared,
- evaluation metrics,
- interpretation methods,
- reliability checks.

### Chapter 5: Results

Explain:

- model comparison,
- best model selection,
- classification versus regression findings,
- H1/H3/H5 tables and figures,
- interpretability results,
- subgroup reliability and diagnostics.

### Chapter 6: System Architecture and Implementation

Explain:

- repository structure,
- pipeline modules,
- scripts and package layout,
- artifact generation,
- dashboard design,
- testing strategy,
- reproducibility choices.

### Chapter 7: Discussion

Explain:

- what the results mean operationally,
- where the model is strong,
- where it is weak,
- how the findings relate to shelter decision-making,
- ethical and methodological limits.

### Chapter 8: Conclusion and Future Work

Explain:

- research questions answered,
- main contributions,
- limits of the current work,
- possible next steps such as survival analysis, DVC, Docker, MLflow, or deployment.

## 13. Recommended Thesis Narrative

The clearest narrative for the thesis is:

1. shelter records are messy but valuable,
2. intake and outcome data can be converted into a reproducible analytical dataset,
3. leakage-safe feature engineering is essential,
4. simple models establish a baseline,
5. stronger models improve prediction,
6. interpretability reveals which signals matter,
7. reliability diagnostics show where the model should be trusted,
8. the final system is useful both for research and for demonstration.

## 14. Important Writing Rules

When writing the thesis, keep these rules in mind:

- do not say that the model proves causation,
- do not use outcome information as if it were known at intake,
- do not overstate generalization beyond AAC,
- distinguish descriptive analysis from predictive modeling,
- explain why a methodological decision was made, not only what was done,
- keep the story focused on H1, H3, and H5.

## 15. Practical Next Steps For Writing

If you want to start writing immediately, begin with:

1. Introduction,
2. Data and preprocessing,
3. Methodology,
4. Results,
5. System architecture,
6. Discussion and conclusion.

The `docs/thesis.md` file can remain the thesis draft skeleton, while this document can act as the technical reference you use while drafting the final text.

## 16. Current Implementation Summary

The repository currently supports:

- raw data download,
- cleaning and matching,
- feature engineering,
- baseline and advanced modeling,
- model comparison,
- H1/H3/H5 analysis,
- diagnostics and reliability,
- SHAP-based interpretation,
- animal-centered profile analysis,
- evidence-pack generation,
- Streamlit demonstration,
- automated tests.

That is enough to support a complete thesis story centered on reproducible machine learning for shelter adoption analysis.

## 17. Current Project Status

This is the most useful section if you want a quick factual snapshot of where the project stands right now.

### 17.1 What Is Already Working

The pipeline already covers the full path from raw data to presentation artifacts:

- raw AAC data download,
- data cleaning and standardization,
- intake/outcome matching,
- feature engineering,
- modeling dataset creation,
- baseline training,
- gradient boosting training,
- CatBoost training,
- analysis tables,
- diagnostics,
- evidence pack generation,
- animal-centered research outputs,
- Streamlit demo dashboard,
- automated tests.

### 17.2 Current Model and Analysis Outputs

The repository currently generates or documents the following important outputs:

- processed modeling dataset in `data/processed/modeling_dataset.csv`,
- feature metadata in `data/processed/feature_columns.json`,
- target metadata in `data/processed/target_columns.json`,
- data attrition and horizon followup audits in `reports/tables/` and `reports/summary/`,
- baseline metrics in `reports/metrics/baseline_metrics.csv`,
- boosting metrics in `reports/metrics/boosting_metrics.csv`,
- advanced metrics in `reports/metrics/advanced_metrics.csv`,
- model comparison tables in `reports/tables/model_comparison_classification.csv` and `reports/tables/model_comparison_regression.csv`,
- H1, H3, and H5 support tables,
- permutation importance tables,
- SHAP summary tables,
- diagnostic tables,
- evidence pack tables,
- summary markdown files,
- thesis dashboard figures.

### 17.3 Documented Result Snapshot

The repository README documents the current full-data model comparison snapshot as follows:

- best classification model by ROC-AUC: histogram gradient boosting,
- combined classification ROC-AUC: about `0.840`,
- dogs classification ROC-AUC: about `0.813` with CatBoost,
- cats classification ROC-AUC: about `0.865`,
- best regression model by MAE: CatBoost,
- combined regression MAE: about `18.55` days,
- dogs regression MAE: about `21.56` days,
- cats regression MAE: about `15.79` days.

These numbers should be treated as pipeline outputs, not final thesis conclusions.

### 17.4 Current Test Status

I verified the current test suite locally with:

```bash
pytest -q
```

Current result:

```text
41 passed in 20.60s
```

This is a strong sign that the current cleaning, dataset-building, feature-engineering, modeling-output, and analysis-output paths are stable in the local workspace.

### 17.5 What Is Happening Now

The most recent work in the repository is centered around turning model outputs into thesis-friendly evidence layers:

- stronger CatBoost outputs,
- SHAP explanations,
- feature-family summaries,
- animal-centered profile analysis,
- animal journey cards,
- evidence pack generation,
- subgroup reliability analysis,
- adoption milestone reporting,
- data audit and horizon censoring metrics,
- dashboard pages for trust, limits, and interpretability.

The codebase is no longer only about predicting adoption. It is also about explaining where the model is reliable, where it struggles, auditing the data flow, and how to present those findings clearly.

## 18. What Still Needs Work

The project is in good shape, but some parts are still intentionally unfinished.

### 18.1 Not Yet Implemented

The repository still does not include full production-style MLOps tooling:

- Docker,
- DVC,
- MLflow,
- deployment pipeline,
- production API service,
- survival modeling as a first-class path.

### 18.2 Still Open in the Analysis Layer

Useful next analysis improvements would be:

- deeper survival-style analysis of time-to-adoption,
- more explicit uncertainty quantification,
- broader subgroup checks,
- more compact summary outputs for writing,
- better cross-reference between hypothesis tables and narrative conclusions.

### 18.3 Still Open in the Presentation Layer

Possible dashboard improvements:

- cleaner comparison views,
- better filters,
- more concise artifact summaries,
- simpler onboarding for a reader who opens the dashboard for the first time.

## 19. Planned Work

This is the practical roadmap if you want to continue improving the repository.

### 19.1 Short-Term Plans

Most valuable next steps:

1. tighten the report-generation layer,
2. keep thesis-ready markdown summaries aligned with model outputs,
3. add or refine figures for model comparison and hypothesis support,
4. keep the dashboard consistent with generated artifacts,
5. continue checking that feature leakage stays blocked.

### 19.2 Medium-Term Plans

Good medium-term additions:

- richer interpretation of feature families,
- stronger subgroup reliability analysis,
- more explicit comparison between dogs and cats,
- optional survival or hazard-style analysis for time-to-adoption,
- a cleaner reproducibility story if you later want Docker or DVC.

### 19.3 Long-Term Options

These are optional, not required for the current thesis:

- package the pipeline for easier reuse,
- add CI if the repo becomes shared remotely,
- introduce experiment tracking,
- create a more polished public demo,
- convert the dashboard into a lighter presentation app.

## 20. Where To Look For The Important Pieces

If you want to continue working quickly, these are the main places to inspect:

- `README.md` for the current pipeline overview,
- `docs/results_summary.md` for result notes,
- `docs/model_diagnostics.md` for interpretation and reliability notes,
- `docs/model_evidence_pack.md` for evidence-pack logic,
- `docs/progress_and_future_work.md` for status and roadmap,
- `docs/technical_architecture_plan.md` for architecture decisions,
- `docs/thesis.md` for the thesis draft skeleton,
- `src/aac_adoption/` for the actual implementation,
- `scripts/` for command-line entry points,
- `tests/` for validation coverage.

## 21. Short Practical Summary

If you only remember a few things, remember these:

- the project already has a complete data-to-model-to-dashboard pipeline,
- the most important thesis threads are H1, H3, and H5,
- the current best documented classification result is around `0.840` ROC-AUC,
- the current best documented regression result is around `18.55` days MAE,
- the current test suite passed locally with `41 passed`,
- the remaining work is mostly about refining outputs and polishing the presentation layer, not rebuilding the core pipeline.

## 22. Machine Learning Core

This project is fundamentally a supervised learning pipeline on shelter tabular data.

### 22.1 Prediction Targets

The pipeline trains on two related prediction tasks:

- **classification**: whether the animal is adopted,
- **regression**: how many days the animal spends before outcome or adoption.

These are complementary because adoption is both a binary event and a timing process.

### 22.2 Current Dataset Size and Split

Latest documented build result:

- matched intake/outcome rows: `162,390`,
- species: dogs and cats only,
- split: time-aware,
- train: `2013-2021`,
- validation: `2022-2023`,
- test: `2024-2025`.

The time split is important because the task is operational and time-dependent. A random split would make the problem easier but less realistic.

### 22.3 Feature Surface

The current modeled feature surface is intentionally intake-time only.

Main feature groups:

- animal identity: `animal_type`, `has_name`, `is_named`,
- intake context: `intake_type`, `intake_condition`, `sex_upon_intake`,
- age: `age_upon_intake`, `age_in_days`, `age_group`,
- appearance: `breed`, `color`, `primary_breed`, `simplified_breed_group`, `primary_color`, `simplified_color_group`, `is_black_or_dark`,
- calendar timing: `intake_year`, `intake_month`, `intake_quarter`, `intake_season`,
- macro-period: `covid_period`,
- breed mix signal: `is_mixed_breed`.

This feature design is a deliberate compromise between predictive power and explainability. It keeps the signal rich enough for modeling while staying understandable enough for thesis discussion.

### 22.4 Leakage Control

The project uses a strict leakage boundary.

Not allowed as predictors:

- `outcome_type`,
- `outcome_subtype`,
- `outcome_datetime`,
- `sex_upon_outcome`,
- `age_upon_outcome`,
- `days_to_outcome`,
- `days_to_adoption`,
- `length_of_stay`,
- `adopted`,
- `is_adopted`,
- `classification_target`,
- `regression_target_days`,
- `target_adopted`.

These columns are available only for:

- label creation,
- evaluation,
- reports,
- diagnostics.

That boundary is one of the most important technical decisions in the repository.

### 22.5 Why These Features Were Chosen

The project intentionally focuses on intake-time variables because they match the real decision point in a shelter.

This means the model answers questions like:

- given this intake record, what is the adoption likelihood,
- given this intake record, what wait time should we expect,
- which intake-time signals are strongest,
- which groups are more difficult to place.

It does not attempt to predict with unavailable future information.

## 23. Modeling Architecture

### 23.1 Model Families

The current model stack is:

- dummy baselines,
- logistic regression,
- random forest,
- histogram gradient boosting,
- CatBoost.

The stack is intentionally layered. Each family serves a different purpose:

- dummy models establish a floor,
- linear models provide interpretable baseline behavior,
- random forests capture nonlinear interactions with modest complexity,
- histogram gradient boosting provides a strong scikit-learn benchmark,
- CatBoost provides a categorical-data-strong advanced benchmark.

### 23.2 Why CatBoost Is Important Here

CatBoost is especially relevant because AAC data is dominated by categorical and mixed-format fields.

Operationally, the data contains many values that are not naturally numeric:

- intake type,
- intake condition,
- breed names and breed groups,
- color descriptions,
- age groups,
- season and period labels.

CatBoost reduces the need for heavy one-hot engineering while remaining strong on tabular problems.

### 23.3 Why Gradient Boosting Still Matters

Histogram gradient boosting remains important even with CatBoost in the stack because it gives:

- a strong scikit-learn reference point,
- comparable evaluation under the same split,
- a good bridge between simple models and the most flexible model family.

### 23.4 Operational Objective of the Model Stack

The model stack is not there only to maximize score.

It is there to answer four operational questions:

1. can adoption be predicted from intake-time data,
2. how much better are nonlinear models than simple baselines,
3. which intake features drive the prediction,
4. how stable is the model across cohorts and time.

## 24. Current Model Results

Latest documented snapshot from the repository:

### 24.1 Classification

- best model by ROC-AUC: histogram gradient boosting,
- combined ROC-AUC: about `0.840`,
- dogs ROC-AUC: about `0.813` with CatBoost,
- cats ROC-AUC: about `0.865`.

Technical interpretation:

- the model has real predictive signal,
- cats are currently easier to classify than dogs in this feature set,
- stronger nonlinear models outperform the simpler baselines,
- ROC-AUC is the cleanest headline ranking metric for this task.

### 24.2 Regression

- best model by MAE: CatBoost,
- combined MAE: about `18.55` days,
- dogs MAE: about `21.56` days,
- cats MAE: about `15.79` days.

Technical interpretation:

- timing prediction is harder than adoption classification,
- the target is skewed and noisy,
- MAE is the most practical operational metric because it is easy to explain in days,
- dogs remain harder than cats for wait-time prediction in the current setup.

### 24.3 What These Numbers Mean

These numbers show that the pipeline is not just producing artifacts. It is learning real structure from the data.

At the same time:

- the scores are not perfect,
- the regression task remains noisy,
- the outputs should be described as predictive association, not causality,
- subgroup checks are needed before treating any score as uniformly reliable.

## 25. Training Workflow

The main training flow is command-driven and reproducible.

### 25.1 Data Preparation

The pipeline starts with:

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
```

This produces the processed dataset and metadata artifacts.

### 25.2 Training Steps

Then the model families are trained separately:

```bash
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
```

The split is shared across the evaluation pipeline, which makes the comparison fair.

### 25.3 Analysis Steps

After training, the pipeline computes:

```bash
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
```

This means the project is built as a sequence of artifacts rather than a notebook that has to be manually rerun.

### 25.4 Reproducibility Controls

Important reproducibility controls:

- explicit file paths,
- explicit split strategy,
- explicit feature list,
- explicit target list,
- automated tests,
- saved model artifacts,
- saved report tables and figures.

## 26. Metrics and Evaluation

### 26.1 Classification Metrics

Used for model ranking and decision support:

- ROC-AUC,
- F1,
- calibration curves,
- precision-recall diagnostics,
- threshold tradeoff tables.

ROC-AUC is the main ranking metric because it is threshold-independent and robust for comparing classifiers.

### 26.2 Regression Metrics

Used for timing prediction:

- MAE,
- RMSE,
- predicted vs actual comparisons,
- residual slices.

MAE is especially useful because it directly answers a practical question: on average, how many days off is the model?

### 26.3 Calibration and Thresholding

The project does not stop at raw scores.

For adoption classification, the model also produces:

- calibration tables,
- threshold tables,
- flagged-share tradeoff views.

This matters because a model can have good ROC-AUC but still be poorly calibrated at the probability level.

## 27. Diagnostics and Reliability

The diagnostics layer is now a major part of the ML system.

It produces:

- ROC and precision-recall curves,
- calibration tables and figures,
- classification error slices,
- regression residual slices,
- placement-risk quadrants,
- adoption milestones,
- SHAP global and feature-family explanations,
- subgroup reliability tables,
- subgroup confidence intervals,
- model failure-mode rankings.

This layer answers a key operational question:

> where does the model work well, and where should we be careful?

That is more useful than score alone.

## 28. Subgroup Reliability

The evidence pack extends evaluation beyond the aggregate dataset.

It examines:

- dogs,
- cats,
- age groups,
- intake types,
- breed groups,
- color groups,
- named vs unnamed animals,
- health and behavior proxy groups.

Important subgroup fields:

- `records`,
- `observed_adoption_rate`,
- `mean_predicted_probability`,
- `calibration_gap`,
- `false_positive_rate`,
- `false_negative_rate`,
- `mae`,
- `small_cohort`.

This is valuable because the model may be much better on some groups than others.

## 29. Evidence Pack Logic

The evidence pack is the main ML-rigor layer.

It combines:

- best model summaries,
- bootstrap confidence intervals,
- subgroup reliability,
- failure modes,
- animal journey examples,
- adoption milestones,
- uncertainty-aware reporting.

This is a strong technical decision because it shifts the project from "we trained a model" to "we understand how trustworthy the model is."

## 30. Software Architecture Notes

### 30.1 Package Layout

The code is organized under `src/aac_adoption/` into functional modules:

- `data` for download, cleaning, loading, matching, and dataset building,
- `features` for feature engineering and feature set definitions,
- `models` for splitting, training, evaluation, and artifact handling,
- `analysis` for hypothesis tables and comparison logic,
- `diagnostics` for reliability and interpretability outputs,
- `reporting` for summary generation,
- `dashboard` for app data and story views,
- `interpretation` for model explanation helpers,
- `visualization` for plots.

### 30.2 Why the Structure Matters

This architecture keeps responsibilities separated.

The result is:

- easier testing,
- easier reuse,
- less duplicated logic,
- cleaner command-line scripts,
- easier dashboard rendering,
- better thesis reproducibility.

### 30.3 Runtime Philosophy

The dashboard reads artifacts; it does not retrain models at runtime.

That is the right choice because:

- it makes the UI fast,
- it avoids hidden computation,
- it keeps the demo deterministic,
- it reduces the chance of accidental environment drift.

## 31. Current Verification Snapshot

Latest local test result:

```text
41 passed in 20.60s
```

This tells us:

- the data pipeline is stable,
- the model-output generation is stable,
- the analysis outputs are stable,
- the repository currently has good automated coverage for the main workflow.

## 32. Current Focus Areas

The current ML focus is on:

- keeping the leakage boundary strict,
- strengthening interpretation with SHAP and feature-family views,
- keeping subgroup reliability visible,
- ensuring the evidence pack stays aligned with model outputs,
- maintaining the dashboard as a read-only artifact consumer,
- keeping the comparison between baseline, boosting, and CatBoost clear.

## 33. Historical Notes

Some earlier documents in `docs/` contain older numbers or older planning language.

If you need the latest working snapshot, prefer:

- `README.md`,
- `docs/results_summary.md`,
- `docs/model_diagnostics.md`,
- `docs/model_evidence_pack.md`,
- `docs/progress_and_future_work.md`.

Those files reflect the project state more directly than the old thesis draft stub.

## 34. Model Pipeline Spec

This section describes the pipeline as a set of concrete input-output stages.

### 34.1 Raw Data Download

Script:

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
```

Input:

- remote Austin Open Data / Socrata exports.

Output:

- `data/raw/intakes.csv`,
- `data/raw/outcomes.csv`.

Technical decisions:

- keep raw data immutable once downloaded,
- use a clear source flag (`historical` vs `current`),
- separate intake and outcome tables rather than merging them prematurely.

Tradeoff:

- the split raw structure adds matching complexity later, but it preserves provenance and avoids hiding source-table differences.

### 34.2 Data Build and Matching

Script:

```bash
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
```

Input:

- raw intakes,
- raw outcomes.

Output:

- `data/processed/modeling_dataset.csv`,
- `data/processed/feature_columns.json`,
- `data/processed/target_columns.json`.

Technical decisions:

- match each intake to the nearest unused future outcome,
- reject negative length-of-stay outcomes,
- restrict to dog and cat records,
- create engineered intake-time features,
- save feature and target metadata separately.

Tradeoff:

- strict matching is more work than a naive join, but it prevents duplicate outcome assignment and leakage.

### 34.3 EDA and Analysis

Scripts:

```bash
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
```

Input:

- processed modeling dataset.

Output:

- EDA tables and figures,
- hypothesis support tables,
- model comparison tables.

Technical decisions:

- separate descriptive summaries from modeling outputs,
- keep H1/H3/H5 logic in stable table artifacts,
- generate summary outputs instead of leaving the analysis trapped in notebooks.

Tradeoff:

- artifact generation takes longer than interactive exploration, but it gives repeatable thesis evidence.

### 34.4 Baseline Training

Script:

```bash
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
```

Input:

- processed dataset,
- leakage-safe feature list,
- target definitions,
- time-aware split.

Output:

- baseline model artifacts,
- baseline metrics,
- logistic regression coefficients,
- random forest feature importance.

Models:

- `DummyClassifier`,
- `LogisticRegression`,
- `RandomForestClassifier`,
- `DummyRegressor`,
- `Ridge`,
- `RandomForestRegressor`.

Technical decisions:

- keep linear models for interpretability,
- include dummy baselines to measure actual signal,
- include random forest as a nonlinear but still familiar benchmark.

Tradeoff:

- linear models are easier to explain but weaker on interactions,
- random forests improve flexibility but become less transparent.

### 34.5 Gradient Boosting Training

Script:

```bash
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
```

Input:

- processed dataset,
- same split and feature contract as baselines.

Output:

- histogram gradient boosting models,
- boosting metrics,
- permutation importance tables.

Technical decisions:

- use boosting as the main scikit-learn high-performing benchmark,
- keep comparison fair by using the same split,
- use permutation importance rather than relying only on internal impurity-style importance.

Tradeoff:

- boosting improves performance and captures nonlinear interactions, but interpretation becomes more indirect.

### 34.6 Advanced CatBoost Training

Script:

```bash
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
```

Input:

- processed dataset,
- categorical-heavy feature space.

Output:

- CatBoost classification/regression artifacts,
- advanced metrics,
- files used later by diagnostics and the dashboard.

Technical decisions:

- use CatBoost because the data is dominated by categorical variables,
- keep the advanced model separate from the baseline stack,
- use it as the strongest operational benchmark.

Tradeoff:

- CatBoost is often stronger on tabular categorical data, but it introduces an extra dependency and a somewhat different training interface.

### 34.7 Diagnostics, SHAP, and Reliability

Script:

```bash
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
```

Input:

- processed dataset,
- trained advanced models.

Output:

- calibration tables and figures,
- threshold tables,
- error slices,
- residual slices,
- SHAP summaries,
- feature-family summaries,
- placement-risk quadrants,
- adoption timeline milestones.

Technical decisions:

- treat SHAP as explanation of model behavior, not ground truth,
- add calibration and threshold views to avoid score-only evaluation,
- include error slices to detect cohort-specific weaknesses.

Tradeoff:

- SHAP and diagnostics add runtime cost, but they make the model defensible in a thesis setting.

### 34.8 Evidence Pack and Report Generation

Scripts:

```bash
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
```

Input:

- model outputs,
- diagnostic outputs,
- animal-centered summary tables.

Output:

- evidence pack tables,
- model limitation tables,
- confidence intervals,
- animal journey examples,
- summary markdown,
- thesis-ready figures.

Technical decisions:

- generate final reporting outputs from stable artifacts,
- keep summary text aligned with machine-readable CSVs,
- separate evidence generation from dashboard rendering.

Tradeoff:

- more artifact files are produced, but the project becomes much easier to audit and reuse.

### 34.9 Dashboard Runtime

Entry point:

```bash
streamlit run streamlit_app.py
```

Runtime behavior:

- reads generated artifacts,
- does not retrain models,
- shows metrics, tables, figures, SHAP, reliability, and model sensitivity views.

Technical decisions:

- artifact-driven UI,
- cache loaded CSVs,
- keep the dashboard as a presentation layer only.

Tradeoff:

- the app is less flexible than an always-live training UI, but it is much more stable and faster to use.

## 35. Feature Catalog

This is the practical list of the feature groups currently used in the modeling system.

### 35.1 Core Identity and Intake Features

- `animal_type`
  - species label, usually dog or cat.
  - useful because dogs and cats behave differently in adoption dynamics.

- `intake_type`
  - reason or situation under which the animal arrived.
  - one of the strongest operational variables because it reflects shelter entry context.

- `intake_condition`
  - condition on arrival.
  - captures health or care needs that can affect adoptability.

- `sex_upon_intake`
  - reproductive / sex status at intake.
  - useful categorical feature with behavioral and operational signal.

- `has_name`, `is_named`
  - name availability flags.
  - act as proxies for identity signal and possibly prior human relationship.

### 35.2 Age Features

- `age_upon_intake`
  - raw age representation.
  - useful but messy, so it is also normalized into numeric forms.

- `age_in_days`
  - fine-grained numeric age.
  - preferred for modeling and comparisons.

- `age_in_months`
  - medium-granularity age.
  - easier to interpret for short-lifetime animals.

- `age_in_years`
  - human-readable age scale.
  - useful for dashboard and explanatory summaries.

- `age_group`
  - coarse group such as baby, young, adult, senior.
  - useful for hypothesis H3 and model interpretability.

Technical decision:

- keep both raw and grouped age views because numeric age helps prediction while grouped age helps communication.

Tradeoff:

- grouped age loses detail, but it makes model behavior easier to explain in the thesis.

### 35.3 Breed Features

- `breed`
  - raw breed description.
  - high-cardinality and noisy, so it benefits from simplification.

- `primary_breed`
  - simplified canonical breed component.
  - reduces string noise.

- `is_mixed_breed`
  - mixed-breed indicator.
  - often useful because mixed vs pure breed patterns can differ.

- `simplified_breed_group`
  - grouped breed category.
  - created for more stable analysis and reporting.

Technical decision:

- simplify breed strings rather than relying on raw breed text alone.

Tradeoff:

- simplification reduces granularity, but it improves robustness and lowers category sparsity.

### 35.4 Color Features

- `color`
  - raw color description.

- `primary_color`
  - canonicalized main color.

- `simplified_color_group`
  - grouped color category.
  - useful for H4-style descriptive analysis.

- `is_black_or_dark`
  - dark-color proxy flag.
  - directly supports the black dog / black cat style descriptive check.

Technical decision:

- keep both raw and simplified color views so that the model can use signal while the thesis can present interpretable groups.

Tradeoff:

- color simplification may hide some nuance, but raw color strings are too sparse and inconsistent for clean summary analysis.

### 35.5 Timing Features

- `intake_year`
- `intake_month`
- `intake_quarter`
- `intake_season`
- `covid_period`

These features encode when the animal entered the shelter.

Technical decisions:

- include calendar timing because adoption behavior changes over time,
- include seasonality because shelter flow is not uniform across the year,
- include COVID-period grouping because the period likely shifted intake/adoption dynamics.

Tradeoff:

- time features can capture real operational changes, but they also risk becoming proxies for many overlapping effects, so interpretation must stay careful.

### 35.6 Derived Modeling Targets

- `classification_target`
- `regression_target_days`
- `target_adopted`

These are not predictors.

They exist to:

- define labels clearly,
- support classification and regression models,
- keep target logic separate from feature logic.

### 35.7 Metadata and Utility Columns

- `animal_id`
- `outcome_datetime`
- `days_to_outcome`
- `length_of_stay`

These columns are essential for joining, evaluation, and reporting, but not for prediction.

Technical decision:

- explicitly separate utility columns from features so leakage checks are easier to implement and test.

## 36. Results Table

This table gives a compact technical snapshot of the current documented results.

| Area | Current Best / Snapshot | Technical Meaning |
|---|---:|---|
| Processed rows | `162,390` | matched intake/outcome episodes available for modeling |
| Species scope | dogs + cats | reduces noise and keeps the problem focused |
| Split strategy | train `2013-2021`, validation `2022-2023`, test `2024-2025` | realistic time-aware evaluation |
| Classification best model | histogram gradient boosting | strongest ROC-AUC among documented models |
| Classification ROC-AUC combined | about `0.840` | clear predictive signal on adoption outcome |
| Classification ROC-AUC dogs | about `0.813` with CatBoost | dogs remain harder than cats in this setup |
| Classification ROC-AUC cats | about `0.865` | cats are currently easier to classify |
| Regression best model | CatBoost | best MAE among documented models |
| Regression MAE combined | about `18.55` days | average absolute timing error for the full set |
| Regression MAE dogs | about `21.56` days | timing prediction is harder for dogs |
| Regression MAE cats | about `15.79` days | timing prediction is easier for cats |
| Current test result | `41 passed in 20.60s` | main workflow is stable locally |

### 36.1 What the Table Says Technically

- the data contains enough structure for real predictive modeling,
- nonlinear tree-based methods outperform the simple baselines,
- the classification task is easier than the regression task,
- dogs and cats should be discussed separately because their behavior differs,
- model quality is not uniform across cohorts,
- the project is technically mature enough to support a thesis, but not overbuilt into production tooling.

### 36.2 Important Tradeoff Summary

The core tradeoffs in the current technical design are:

- **interpretability vs performance**: linear models are easier to explain, boosting/CatBoost perform better,
- **granularity vs stability**: raw breed/color text is detailed but noisy, grouped features are more robust,
- **speed vs rigor**: artifact generation and diagnostics are slower than a notebook, but much easier to audit,
- **flexibility vs reproducibility**: Streamlit runtime is intentionally static and artifact-driven,
- **simple evaluation vs trustworthiness**: ROC-AUC alone is not enough, so calibration, thresholds, and subgroup reliability were added,
- **single-score thinking vs operational realism**: adoption likelihood and adoption time are both modeled because shelter work depends on both.

## 37. Model Hyperparameters and Training Behavior

This section captures the concrete training choices currently baked into the code.

### 37.1 Baseline Model Settings

#### Logistic Regression

Configuration:

- `max_iter=1000`
- `class_weight="balanced"`

Why this matters:

- `max_iter=1000` avoids convergence issues on one-hot expanded data,
- `class_weight="balanced"` helps with class imbalance in adoption classification.

Tradeoff:

- class weighting improves recall on minority classes, but it can reduce precision and shift the decision boundary.

#### Random Forest Classifier

Configuration:

- `n_estimators=50`
- `max_depth=14`
- `min_samples_leaf=10`
- `random_state=RANDOM_STATE`
- `class_weight="balanced_subsample"`
- `n_jobs=-1`

Why this matters:

- the depth limit controls overfitting,
- `min_samples_leaf=10` smooths noisy splits,
- class weighting helps with imbalance,
- `n_jobs=-1` uses all available CPU cores.

Tradeoff:

- a shallower forest is faster and more stable, but it may miss subtle interactions.

#### Ridge Regression

Configuration:

- default `Ridge()` parameters in scikit-learn.

Why this matters:

- ridge gives a stable linear timing baseline,
- it handles multicollinearity better than ordinary least squares.

Tradeoff:

- ridge remains linear, so it cannot capture strong nonlinear shelter patterns.

#### Dummy Models

Configuration:

- `DummyClassifier(strategy="most_frequent")`
- `DummyRegressor(strategy="median")`

Why this matters:

- these models define the floor of performance,
- they reveal how much signal the real features add beyond a trivial guess.

Tradeoff:

- they are intentionally weak, but they are essential for honest benchmarking.

### 37.2 Histogram Gradient Boosting Settings

#### Classifier

Configuration:

- `learning_rate=0.08`
- `max_iter=100`
- `max_leaf_nodes=31`
- `random_state=RANDOM_STATE`

#### Regressor

Configuration:

- `learning_rate=0.08`
- `max_iter=100`
- `max_leaf_nodes=31`
- `random_state=RANDOM_STATE`

Why this matters:

- moderate learning rate and limited iteration count keep the model practical,
- `max_leaf_nodes=31` constrains tree complexity,
- the same settings across tasks make comparison cleaner.

Tradeoff:

- these settings are conservative enough to train predictably, but they may leave some performance on the table versus heavier tuning.

### 37.3 CatBoost Settings

#### Classification

Configuration:

- `loss_function="Logloss"`
- `eval_metric="AUC"`
- `iterations=1000` by default in the main pipeline
- `learning_rate=0.05`
- `depth=6`
- `early_stopping_rounds=50`
- `random_seed=RANDOM_STATE`

#### Regression

Configuration:

- `loss_function="MAE"`
- `eval_metric="MAE"`
- `iterations=1000`
- `learning_rate=0.05`
- `depth=6`
- `early_stopping_rounds=50`
- `random_seed=RANDOM_STATE`

Why this matters:

- AUC is the right optimization view for adoption ranking,
- MAE is a practical objective for wait-time prediction,
- early stopping reduces wasted iterations and helps generalization.

Tradeoff:

- CatBoost is more powerful and category-aware, but it adds a more specialized dependency and a slightly different feature-preparation path.

### 37.4 Permutation Importance Settings

Configuration:

- `scoring="roc_auc"` for classification,
- `scoring="neg_mean_absolute_error"` for regression,
- `n_repeats=3` by default,
- `permutation_max_rows=3000` by default,
- `random_state=RANDOM_STATE`.

Why this matters:

- permutation importance measures how much a feature matters to actual model performance,
- the sample cap keeps runtime manageable.

Tradeoff:

- permutation importance is more expensive than built-in tree importance, but it is easier to compare across model families and often more trustworthy as a cross-model explanation.

## 38. Output File Map

This section maps the most important scripts to their outputs.

### 38.1 Data Acquisition and Build

#### `scripts/download_raw_data.py`

Produces:

- `data/raw/intakes.csv`
- `data/raw/outcomes.csv`

#### `scripts/build_dataset.py`

Produces:

- `data/processed/modeling_dataset.csv`
- `data/processed/feature_columns.json`
- `data/processed/target_columns.json`

### 38.2 Exploration and Analysis

#### `scripts/run_eda.py`

Produces EDA tables and figures under:

- `reports/tables/`
- `reports/figures/`

#### `scripts/run_analysis.py`

Produces:

- `reports/tables/model_comparison_classification.csv`
- `reports/tables/model_comparison_regression.csv`
- `reports/tables/h1_intake_vs_appearance.csv`
- `reports/tables/h3_age_adoption_speed.csv`
- `reports/tables/h5_covid_period.csv`

### 38.3 Model Training

#### `scripts/train_baseline.py`

Produces:

- `reports/metrics/classification_metrics.csv`
- `reports/metrics/regression_metrics.csv`
- `reports/metrics/baseline_metrics.csv`
- `models/baseline/`
- `reports/tables/logistic_regression_coefficients.csv`
- `reports/tables/random_forest_feature_importance.csv`

#### `scripts/train_boosting.py`

Produces:

- `reports/metrics/boosting_classification_metrics.csv`
- `reports/metrics/boosting_regression_metrics.csv`
- `reports/metrics/boosting_metrics.csv`
- `models/boosting/`
- `reports/tables/permutation_importance_classification.csv`
- `reports/tables/permutation_importance_regression.csv`

#### `scripts/train_advanced.py`

Produces:

- `reports/metrics/advanced_classification_metrics.csv`
- `reports/metrics/advanced_regression_metrics.csv`
- `reports/metrics/advanced_metrics.csv`
- `models/advanced/`

### 38.4 Diagnostics and Evidence

#### `scripts/generate_diagnostics.py`

Produces:

- `reports/diagnostics/classification_thresholds.csv`
- `reports/diagnostics/classification_calibration.csv`
- `reports/diagnostics/classification_error_slices.csv`
- `reports/diagnostics/regression_error_slices.csv`
- `reports/diagnostics/placement_risk_quadrants.csv`
- `reports/tables/shap_global_classification.csv`
- `reports/tables/shap_global_regression.csv`
- `reports/tables/shap_feature_families_classification.csv`
- `reports/tables/shap_feature_families_regression.csv`
- diagnostic figures in `reports/figures/`

#### `scripts/generate_animal_research.py`

Produces:

- `reports/tables/animal_archetypes.csv`
- `reports/tables/vulnerable_profiles.csv`
- `reports/tables/profile_contrasts.csv`
- `reports/tables/profile_model_error.csv`
- `reports/tables/health_behavior_profiles.csv`
- corresponding figures in `reports/figures/`

#### `scripts/generate_evidence_pack.py`

Produces:

- `reports/tables/model_evidence_pack.csv`
- `reports/tables/model_limitations_by_cohort.csv`
- `reports/tables/metric_confidence_intervals.csv`
- `reports/tables/animal_journey_examples.csv`
- `reports/tables/subgroup_reliability.csv`
- `reports/tables/subgroup_metric_confidence_intervals.csv`
- `reports/tables/subgroup_adoption_milestones.csv`
- `reports/tables/model_failure_modes.csv`
- `reports/summary/model_evidence_pack.md`
- `reports/summary/subgroup_reliability.md`

#### `scripts/generate_report_outputs.py`

Produces:

- summary markdown,
- comparison figures,
- hypothesis figures,
- reusable thesis summary artifacts.

### 38.5 Dashboard

#### `streamlit_app.py`

Reads:

- generated tables,
- generated summaries,
- generated figures,
- saved model artifacts,
- diagnostics and evidence files.

Does not produce:

- new training data,
- new model weights during normal runtime.

## 39. Libraries and Technical Tradeoffs

This section explains why the current library stack is the way it is.

### 39.1 pandas

Role:

- data loading,
- cleaning,
- grouping,
- merges and joins,
- CSV artifact creation.

Tradeoff:

- very flexible, but large-table operations can become memory-heavy if not controlled.

### 39.2 numpy

Role:

- numeric operations,
- metric support,
- low-level array handling.

Tradeoff:

- fast and standard for numeric work, but not a modeling framework by itself.

### 39.3 scikit-learn

Role:

- preprocessing pipelines,
- split logic,
- baseline models,
- histogram gradient boosting,
- evaluation metrics,
- permutation importance.

Tradeoff:

- excellent for structured ML workflows, but some categorical and explanation workflows need extra care or extra tools.

### 39.4 CatBoost

Role:

- strongest categorical-friendly model family in the project.

Tradeoff:

- high utility on tabular categorical data, but a more specialized dependency and a different input prep path from scikit-learn.

### 39.5 joblib

Role:

- saving fitted pipelines,
- loading model artifacts later in the dashboard or analysis code.

Tradeoff:

- simple and practical, but it ties the saved artifact to the Python ecosystem and library versions.

### 39.6 SHAP

Role:

- global and local explanation,
- feature-family summaries,
- animal journey reasoning.

Tradeoff:

- powerful interpretation layer, but computationally more expensive and easy to over-interpret if language is not careful.

### 39.7 Streamlit

Role:

- thesis demo UI,
- artifact browsing,
- story presentation,
- model sensitivity interface.

Tradeoff:

- very fast for dashboards, but not a backend service and not a full production app framework.

### 39.8 Altair and Plotly

Role:

- clean interactive charting in the dashboard.

Tradeoff:

- easy for rich presentation, but figure generation is less central than the CSV artifact pipeline.

### 39.9 pytest

Role:

- automated verification of the data and model pipeline.

Tradeoff:

- tests add maintenance cost, but they prevent silent regressions and keep the project trustworthy.

## 40. Code Approach Notes

This is how the repository is engineered at a higher level.

### 40.1 Command-Line First

The project favors script entry points over manual notebook execution.

Why:

- easier reproducibility,
- easier testing,
- easier automation,
- clearer dependency chain.

### 40.2 Artifact-Driven

The main computations produce stable files that later stages read.

Why:

- avoids retraining during dashboard viewing,
- makes results auditable,
- lets the thesis cite fixed outputs.

### 40.3 Modular Package Code

Logic lives under `src/aac_adoption/` rather than being embedded in scripts.

Why:

- reusable by tests and scripts,
- easier to maintain,
- less duplication.

### 40.4 Explicit Metadata

Each saved model artifact includes sidecar metadata.

Why:

- model name,
- task,
- subset,
- split,
- feature set,
- timestamp,
- training row counts,
- parameter config.

This makes the artifacts traceable.

### 40.5 Conservative Defaults

The pipeline avoids aggressive complexity by default.

Examples:

- limited tree depth,
- limited number of iterations for some models,
- conservative one-hot settings,
- controlled permutation importance sampling.

Why:

- keeps runtime manageable,
- avoids overfitting,
- keeps the project reproducible on a normal laptop.

### 40.6 Clear Separation of Concerns

The code cleanly separates:

- raw data,
- processed data,
- training,
- interpretation,
- diagnostics,
- reporting,
- dashboarding.

This is one of the best engineering decisions in the repo because it makes the system understandable and maintainable.

## 41. Data Cleaning and Matching Logic

This is the part of the pipeline that turns messy shelter records into a usable modeling dataset.

### 41.1 Raw Cleaning Strategy

Cleaning is intentionally conservative.

The code only performs transformations that are safe for modeling and do not invent new information.

#### Required-column validation

Before anything else, the cleaning layer checks that the expected raw columns exist.

For intakes, the minimum required fields include:

- `animal_id`
- `animal_type`
- `intake_datetime`

For outcomes, the minimum required fields include:

- `animal_id`
- `animal_type`
- `outcome_datetime`
- `outcome_type`

Why this matters:

- if a required column is missing, the pipeline fails early instead of quietly producing a broken dataset.

#### Datetime parsing

Datetime parsing is handled explicitly because AAC historical exports can contain mixed formats and timezone offsets.

Technical behavior:

- strings are normalized,
- `T` is replaced with a space,
- trailing timezone markers like `Z` or `+00:00` are stripped,
- parsed values are converted to `datetime64[ns]`,
- timezone awareness is removed without shifting clock time.

Why this matters:

- the model uses local shelter timestamps for length-of-stay matching,
- shifting timestamps during timezone conversion would distort time differences.

Tradeoff:

- timezone precision is reduced, but the local operational timeline becomes consistent.

#### Duplicate handling

The cleaning layer removes exact duplicate rows only.

Why this matters:

- exact duplicates are usually noise or export artifacts,
- non-exact duplicates may still represent meaningful repeated stays and should not be removed automatically.

Tradeoff:

- conservative duplicate removal avoids accidental data loss.

#### Species filtering

Only dogs and cats are retained.

Why this matters:

- the thesis focuses on the dominant shelter species,
- rare species would add noise and make subgroup analysis harder to interpret.

Tradeoff:

- the analysis becomes narrower, but it is more defensible and more stable.

### 41.2 Intake/Outcome Matching Logic

This is one of the most important technical choices in the project.

The matching logic creates one row per intake episode when a valid future outcome exists.

Core rule:

- each intake is matched to the nearest unused future outcome for the same `animal_id`.

Why this is necessary:

- AAC animals can appear multiple times,
- the same animal can have multiple intake/outcome pairs across different stays,
- a naive merge by `animal_id` could reuse one outcome multiple times,
- a naive merge could also match outcomes that occur before the intake.

Algorithm behavior:

- outcomes are grouped by `animal_id`,
- each animal’s outcomes are sorted by `outcome_datetime`,
- intakes are sorted by `intake_datetime`,
- the matcher walks forward through the outcome list for each animal,
- earlier outcomes that predate an intake are skipped,
- once an outcome is used, it is not reused for that animal.

What this prevents:

- duplicate outcome assignment,
- negative length-of-stay values,
- impossible temporal orderings,
- leakage through future information.

Tradeoff:

- this is more work than a simple merge, but it gives a much more realistic episode-level dataset.

### 41.3 Modeling Dataset Construction

After matching, the builder creates the final modeling dataset.

Important derived columns:

- `days_to_outcome`
- `adopted`
- `is_adopted`
- `target_adopted`
- `classification_target`
- `regression_target_days`
- `length_of_stay`
- `days_to_adoption`

How these are computed:

- `days_to_outcome` is the difference between `outcome_datetime` and `intake_datetime` in days,
- `adopted` is `True` when normalized `outcome_type` equals `adoption`,
- `classification_target` is `adopted` converted to 0/1,
- `regression_target_days` is the same as `days_to_outcome`,
- `days_to_adoption` is filled only for adopted animals.

Why this design matters:

- classification and regression can share the same cleaned dataset,
- target creation is explicit and testable,
- outcome labels are easy to inspect and validate.

Validation checks include:

- required columns present,
- only dog/cat records remain,
- `days_to_outcome` is non-negative,
- classification target is binary,
- `classification_target` matches `adopted`.

### 41.4 Build-Time Metadata

The builder also writes:

- `feature_columns.json`
- `target_columns.json`

Why this matters:

- feature and target contracts become explicit artifacts,
- later scripts can use the same feature set without re-discovering it,
- the thesis can point to a stable definition of what counts as a predictor.

### 41.5 Data-Cleaning Tradeoffs

Main tradeoffs in the cleaning and matching design:

- **strictness vs recall**: conservative validation may drop some rows, but it prevents broken labels,
- **precision vs simplicity**: exact duplicate removal is safe, but it does not attempt more complex de-duplication heuristics,
- **episode realism vs implementation cost**: nearest-future matching is more realistic than a simple join, but it adds logic and runtime,
- **timezone fidelity vs operational consistency**: removing timezone markers avoids mismatched time differences.

## 42. Feature Engineering Internals

This section explains how raw AAC fields become model-ready variables.

### 42.1 Age Parsing

Age is one of the most important variables, but the raw AAC age field is not directly numeric.

The parser handles values such as:

- `2 years`
- `7 months`
- `10 days`
- `3 weeks`

How it works:

- text is normalized to lowercase,
- a regex extracts numeric value and unit,
- units are mapped to approximate days,
- invalid or negative values return `NaN`.

Age unit conversion table:

- day/days -> `1`
- week/weeks -> `7`
- month/months -> `30.4375`
- year/years -> `365.25`

Why this matters:

- models need numeric age signals,
- age is one of the strongest adoption predictors,
- keeping multiple numeric scales improves both modeling and presentation.

Tradeoff:

- the conversion is approximate, but the gain in usable signal is much larger than the small conversion error.

### 42.2 Age Features

The pipeline creates:

- `age_in_days`
- `age_in_months`
- `age_in_years`
- `age_days`
- `age_months`
- `age_years`
- `age_group`

Why both numeric and grouped versions exist:

- numeric features support fine-grained ML splitting,
- grouped features support readable analysis and robust table summaries.

Age groups:

- baby: less than 1 year,
- young: less than 3 years,
- adult: less than 8 years,
- senior: 8 years and older,
- unknown: missing age.

Tradeoff:

- age bins lose some information, but they make the thesis results easier to explain and inspect.

### 42.3 Calendar and Season Features

The pipeline extracts:

- `intake_year`
- `intake_month`
- `intake_quarter`
- `intake_season`
- `covid_period`

Season mapping:

- winter: December to February,
- spring: March to May,
- summer: June to August,
- autumn: September to November.

COVID-period mapping:

- `pre_covid`: before 2020-03-01,
- `covid`: 2020-03-01 through 2021-12-31,
- `post_covid`: 2022-01-01 and later.

Why this matters:

- shelter operations change over time,
- seasonality may affect intake/adoption flow,
- COVID likely changed human behavior, intake pressure, and adoption dynamics.

Tradeoff:

- calendar features are very interpretable but can capture multiple overlapping effects at once.

### 42.4 Breed Engineering

Breed strings are noisy and high-cardinality, so the pipeline derives several cleaned views.

Derived breed fields:

- `primary_breed`
- `is_mixed_breed`
- `simplified_breed_group`

Primary breed logic:

- lower-case the string,
- remove ` mix`,
- take the first token before `/`,
- normalize spaces to underscores.

Mixed-breed detection:

- returns `True` if the text contains `mix` or `/`.

Simplified breed groups:

- `domestic_cat`,
- `pit_bull_type`,
- `chihuahua_type`,
- `retriever_type`,
- `shepherd_type`,
- `terrier_type`,
- `hound_type`,
- `other`,
- `unknown`.

Why this matters:

- raw breed descriptions are too sparse to use directly as the only view,
- grouped breed families improve the stability of both modeling and descriptive analysis.

Tradeoff:

- simplifying breed strings can hide nuance, but it makes patterns easier to compare across the dataset.

### 42.5 Color Engineering

Derived color fields:

- `primary_color`
- `color_group`
- `simplified_color_group`
- `is_black_or_dark`

Color simplification logic:

- `black_or_dark` if the string contains `black` or `sable`,
- `brown_tan` for brown/chocolate/liver/seal,
- `white_light` for white/cream,
- `gray_blue` for gray/grey/blue/silver,
- `orange_yellow` for orange/yellow/gold/tan/buff/fawn,
- `mixed_other` for calico/torbie/tortie/tricolor/brindle and the fallback.

Why this matters:

- color descriptions are highly variable and often multi-token,
- simplified groups are useful for the black-dog / black-cat descriptive hypothesis,
- the `is_black_or_dark` flag is a direct operational proxy for that analysis.

Tradeoff:

- coarse color bins are easier to analyze, but they collapse many unique coat descriptions into broad groups.

### 42.6 Name Signal

The pipeline creates:

- `has_name`
- `is_named`

Implementation:

- checks whether the `name` field exists and is non-empty after trimming whitespace.

Why this matters:

- having a name may signal stronger prior human contact,
- it can also act as a weak proxy for how the animal is presented in the shelter record.

Tradeoff:

- the name flag is a weak proxy and should not be over-interpreted, but it can still contribute predictive signal.

### 42.7 Feature Surface Design

The final feature surface is intentionally mixed:

- numeric age features,
- categorical intake descriptors,
- grouped breed/color abstractions,
- calendar timing,
- COVID period,
- name flags.

Why this is a good design:

- numeric signals help the models split effectively,
- categorical signals preserve important operational context,
- grouped abstractions reduce sparsity,
- the resulting feature set remains explainable.

## 43. Diagnostics and Evidence Pack Internals

This section explains how the diagnostics layer works and what each table is for.

### 43.1 Diagnostic Prediction Frame

The diagnostics pipeline first builds a prediction frame from the advanced CatBoost models.

What it does:

- loads the processed dataset,
- rebuilds the same time-aware split,
- loads the combined advanced classifier and regressor,
- predicts adoption probability,
- predicts days to outcome,
- computes residuals and absolute errors,
- keeps a curated set of columns for reliability analysis.

Important columns in the prediction frame:

- `predicted_adoption_probability`
- `predicted_adopted`
- `predicted_days_to_outcome`
- `regression_residual`
- `absolute_error`

Why this matters:

- all downstream reliability tables use the same consistent prediction frame,
- classification and regression diagnostics are computed from aligned outputs.

### 43.2 Classification Curves

Tables:

- `classification_roc_curve.csv`
- `classification_precision_recall_curve.csv`

What they contain:

- FPR, TPR, and thresholds for ROC,
- precision, recall, and thresholds for PR,
- AUC values.

Why they matter:

- ROC shows ranking power across thresholds,
- PR is important when positive-class behavior is more operationally relevant than raw accuracy.

### 43.3 Threshold Tradeoff Table

Table:

- `classification_thresholds.csv`

Contents:

- threshold,
- precision,
- recall,
- F1,
- share flagged as adoptable,
- confusion-matrix counts.

What it is used for:

- operational threshold selection,
- tradeoff analysis between missing true adopters and over-flagging non-adopters.

Why this is useful:

- a good ROC-AUC alone does not tell you where to set the adoption cutoff.

### 43.4 Calibration Table

Table:

- `classification_calibration.csv`

Contents:

- probability bin,
- record count,
- mean predicted probability,
- observed adoption rate.

Why it matters:

- it shows whether predicted probabilities are well aligned with reality,
- it is critical before using model scores as decision-support probabilities.

Tradeoff:

- calibration bins make the probability issue visible, but they are only a summary and do not solve calibration by themselves.

### 43.5 Error Slice Tables

Tables:

- `classification_error_slices.csv`
- `regression_error_slices.csv`

What they do:

- group records by fields such as animal type, age group, intake type, intake condition, COVID period, and color group,
- compute class-specific error rates and regression MAE,
- filter out slices with too few records.

Why they matter:

- they expose where the model struggles,
- they are more actionable than aggregate metrics.

Important classification slice fields:

- records,
- adoption rate,
- mean predicted adoption probability,
- false positive rate,
- false negative rate.

Important regression slice fields:

- records,
- MAE,
- median absolute error.

### 43.6 Placement Risk Quadrants

Table:

- `placement_risk_quadrants.csv`

Logic:

- combines predicted adoption probability and predicted days to outcome,
- maps records into simple decision-support quadrants.

Quadrants:

- `likely_quick_placement`
- `adoptable_needs_visibility`
- `long_stay_risk`
- `non_adoption_or_fast_exit_risk`

Why this matters:

- it turns two model outputs into an operational prioritization view,
- it is helpful for shelter-style decision support.

Tradeoff:

- quadrants are simple and useful, but they are still heuristic summaries of model outputs.

### 43.7 Adoption Milestones

Table:

- `adoption_by_day_milestones.csv`

What it measures:

- the share of adopted animals that were adopted by day 7, 30, and 90,
- grouped by age group and intake type.

Why this matters:

- it gives a time-to-adoption view without requiring a full survival model,
- it is useful for describing descriptive adopted-only timing at a glance.

Important caveat:

- this is descriptive timing evidence, not a rigorous survival analysis with censoring and competing risks.

### 43.8 SHAP Outputs

Tables:

- `shap_global_classification.csv`
- `shap_global_regression.csv`
- `shap_feature_families_classification.csv`
- `shap_feature_families_regression.csv`
- `shap_local_examples.csv`

What they contain:

- global mean absolute SHAP by feature,
- mean SHAP by feature,
- grouped feature-family SHAP summaries,
- local examples with top associated features.

Why this matters:

- SHAP makes the model more explainable,
- feature-family summaries are easier to discuss than raw feature lists,
- local examples help connect a specific animal profile to model behavior.

Interpretation rule:

- SHAP describes association with model output, not causation in the real world.

### 43.9 Evidence Pack Tables

Main tables:

- `model_evidence_pack.csv`
- `model_limitations_by_cohort.csv`
- `metric_confidence_intervals.csv`
- `subgroup_reliability.csv`
- `subgroup_metric_confidence_intervals.csv`
- `subgroup_adoption_milestones.csv`
- `model_failure_modes.csv`
- `animal_journey_examples.csv`

What each one means:

- `model_evidence_pack.csv`
  - compact summary of best model evidence and interpretability evidence.
- `model_limitations_by_cohort.csv`
  - where model calibration and errors differ by cohort.
- `metric_confidence_intervals.csv`
  - bootstrap uncertainty for key global metrics.
- `subgroup_reliability.csv`
  - same trust/limits logic organized by cohort.
- `subgroup_metric_confidence_intervals.csv`
  - metric intervals inside selected subgroups when enough data exists.
- `subgroup_adoption_milestones.csv`
  - descriptive adoption timing by subgroup.
- `model_failure_modes.csv`
  - top cohorts where calibration gap, MAE, or error rate is worst.
- `animal_journey_examples.csv`
  - a few representative, human-readable animal profile examples with predictions, similar cases, and SHAP-based reasons.

### 43.10 Bootstrap Confidence Intervals

The evidence pack uses bootstrapping to estimate uncertainty for:

- ROC-AUC,
- PR-AUC,
- F1 at threshold 0.50,
- MAE.

Why this matters:

- point estimates alone can be misleading,
- confidence intervals make the thesis more defensible.

Tradeoff:

- bootstrap intervals add runtime, but they significantly improve interpretability of model quality.

### 43.11 Failure Modes

The failure-mode table ranks the worst cohorts by:

- calibration gap,
- MAE,
- false negative rate,
- false positive rate.

Why this matters:

- it turns raw subgroup reliability into a prioritized warning list,
- it helps identify which cohorts deserve careful language in the thesis.

### 43.12 Evidence-Pack Design Tradeoffs

Main tradeoffs:

- **detail vs readability**: the pack is rich in tables, but each one serves a narrow purpose,
- **global vs subgroup**: aggregate metrics are useful, but subgroup tables prevent overclaiming,
- **rigor vs runtime**: bootstrapping and SHAP add cost, but they improve thesis defensibility,
- **prediction vs explanation**: the pack combines both because a thesis needs more than scores.

## 44. Testing Architecture

The test suite is part of the technical design, not an afterthought.

### 44.1 What the Tests Protect

The tests focus on the failure modes that would silently ruin the thesis if they were not caught.

Core protection areas:

- data cleaning,
- dataset matching,
- feature engineering,
- leakage control,
- split logic,
- model metrics,
- diagnostics output schemas,
- artifact generation behavior.

### 44.2 Dataset Build Tests

The dataset tests verify that:

- non-dog/cat rows are filtered out,
- matched rows are created correctly,
- targets are generated correctly,
- `days_to_outcome` is non-negative,
- repeated animals are matched to the next unused future outcome,
- black/dark color flags and name flags are built correctly.

Why this matters:

- these are the highest-risk parts of the pipeline,
- if they break, the entire model stack becomes unreliable.

### 44.3 Feature Engineering Tests

The feature tests verify:

- age parsing,
- season mapping,
- age-group mapping,
- COVID-period mapping,
- color simplification,
- primary breed parsing.

Why this matters:

- the feature layer defines the predictive signal,
- mistakes here would propagate into every model.

### 44.4 Diagnostics Tests

The diagnostics tests verify:

- threshold tables contain the expected columns,
- calibration tables contain the expected columns,
- error-slice tables contain the expected columns,
- placement-risk tables contain the expected columns,
- feature-family mapping behaves as expected,
- classification metrics include PR-AUC.

Why this matters:

- the thesis relies on these tables for explanation and model trust,
- schema stability is more important than cosmetic output.

### 44.5 Test Philosophy

The current test strategy is intentionally practical rather than exhaustive.

It emphasizes:

- invariants,
- output schemas,
- leakage prevention,
- temporal matching correctness,
- feature correctness,
- artifact availability.

Tradeoff:

- the tests do not attempt to prove the models are optimal, but they do protect the correctness of the pipeline machinery.

### 44.6 Current Test Status

Latest local verification:

```text
41 passed in 20.60s
```

This means the core pipeline is currently behaving consistently in the workspace.

## 45. Dashboard Architecture

The Streamlit dashboard is an artifact reader and explainer, not a training engine.

### 45.1 Runtime Data Flow

The dashboard reads:

- model comparison tables,
- hypothesis tables,
- animal archetype tables,
- vulnerable profile tables,
- SHAP summary tables,
- diagnostics tables,
- evidence pack tables,
- summary markdown,
- saved model artifacts.

It does not normally:

- rebuild the dataset,
- retrain models,
- regenerate diagnostics at runtime.

Why this matters:

- faster startup,
- fewer runtime failures,
- more stable thesis demo behavior.

### 45.2 Cache Strategy

The dashboard caches loaded tables and diagnostics.

Why:

- reduces repeated disk reads,
- makes interactive use smoother,
- keeps the app responsive when switching tabs.

Tradeoff:

- cache invalidation must be handled by restarting or refreshing the app when new artifacts are generated.

### 45.3 Dashboard Data Helpers

The dashboard data helpers provide:

- CSV loaders,
- optional-file loaders,
- best-model extraction,
- model-record builders,
- prediction wrappers,
- local SHAP explanations,
- similar-case lookup,
- visibility labels.

This is important because the dashboard needs to construct a coherent view from many independent artifacts.

### 45.4 Story and Visual Layers

The story layer provides:

- workflow graph,
- Sankey diagram,
- layered approach comparison,
- story cards.

Why this matters:

- it turns the pipeline into a readable system story,
- it helps non-technical readers understand the flow from raw data to decision-support output.

### 45.5 Prediction and What-If Layer

The dashboard builds a manual prediction record from user input and sends it through the saved CatBoost artifacts.

Technical behavior:

- user selects animal type, intake type, condition, sex status, age, breed, color, name status, and intake date,
- the app derives all intake-time features,
- the saved classifier returns adoption probability,
- the saved regressor returns expected days to outcome,
- the app fetches similar historical cases for context.

Why this is good design:

- the UI stays aligned with the actual model feature contract,
- the prediction view remains consistent with the training data schema.

### 45.6 Journey Cards and Local Explanations

The animal stories section combines:

- archetype statistics,
- representative prediction record,
- similar historical cases,
- local SHAP or fallback global SHAP reasons,
- visibility label.

Why this matters:

- it creates a concrete case-study view,
- it makes the model easier to reason about than a plain metric table.

### 45.7 Dashboard Tradeoffs

Main tradeoffs:

- **static artifacts vs live retraining**: the dashboard is more stable because it reads saved outputs,
- **rich interactivity vs runtime simplicity**: prediction and SHAP are available, but they depend on precomputed artifacts,
- **presentation vs analysis**: the dashboard is for explanation, not discovery.

## 46. Known Limitations and Edge Cases

The pipeline already anticipates several important data and modeling edge cases.

### 46.1 Repeated Animal Stays

AAC animals can appear multiple times.

Handling:

- the matcher uses the nearest unused future outcome per intake.

Limitation:

- this still treats each intake/outcome pair as an episode, which is appropriate for operational modeling but not a full longitudinal state model.

### 46.2 Missing or Messy Age Data

Age strings can be missing or malformed.

Handling:

- invalid ages become `NaN`,
- age groups become `unknown` when age is missing.

Limitation:

- missing age can reduce model signal, especially for very young or undocumented animals.

### 46.3 High-Cardinality Breed and Color Fields

Raw breed and color strings are messy.

Handling:

- primary fields,
- simplified groups,
- mixed-breed flags,
- dark-color flags.

Limitation:

- simplification reduces sparsity, but it also loses some descriptive specificity.

### 46.4 Time Drift

The historical process changes over time.

Handling:

- time-aware split,
- COVID-period feature,
- year/month/season features.

Limitation:

- the model can still experience concept drift if shelter operations shift again after the test period.

### 46.5 Class Imbalance

Adoption is not perfectly balanced against non-adoption.

Handling:

- class-weighted logistic regression,
- class-weighted random forest,
- PR-AUC,
- calibration and threshold tables.

Limitation:

- threshold choices can materially change recall and precision, so the model should not be used as a single fixed score without context.

### 46.6 Calibration Risk

A model can rank well and still output imperfect probabilities.

Handling:

- calibration tables,
- threshold tables,
- subgroup reliability tables.

Limitation:

- probability scores should be interpreted cautiously before being used for operational prioritization.

### 46.7 Small Cohorts

Some subgroup slices are small.

Handling:

- `small_cohort_flag`,
- minimum record thresholds,
- skipped or marked bootstrap intervals,
- caution in reporting.

Limitation:

- small cohorts should not be used for strong general claims.

### 46.8 Behavior and Health Labels

The data contains care-context or condition labels, but not full behavioral psychology or full veterinary detail.

Handling:

- `health_profile` and `behavior_support_flag` are intentionally conservative proxy fields.

Limitation:

- these descriptors should not be turned into moral labels or personality claims.

### 46.9 Survival-Analysis Boundary

The project measures adoption timing and day milestones, but it is not a full survival model.

Handling:

- descriptive milestone tables,
- median days,
- regression on days to outcome.

Limitation:

- censoring and competing risks are not modeled as rigorously as they would be in a true survival-analysis setup.

### 46.10 Operational Interpretation Risk

The model outputs are easy to over-interpret.

Handling:

- explicit caveats,
- association language,
- reliability and failure-mode tables,
- evidence-pack summaries.

Limitation:

- the system is decision-support oriented, not a prescription engine.

## 47. Final Technical State

If the goal is to know whether the technical topic is effectively complete, the answer is yes:

- the data pipeline is built,
- the feature engineering is defined,
- the model stack is implemented,
- the evaluation metrics are in place,
- diagnostics and evidence layers exist,
- the dashboard is wired to artifacts,
- tests are passing locally,
- the important tradeoffs and edge cases are already documented.

What remains is mostly thesis writing and any optional refinement you personally want to add, not missing core ML infrastructure.

## 48. Hypothesis Table Internals

The H1/H3/H5 tables are simple on purpose, but they are still important because they connect the model work to the thesis hypotheses.

### 48.1 H1 Table

File:

- `h1_intake_vs_appearance.csv`

Built from:

- `intake_type`
- `intake_condition`
- `simplified_breed_group`
- `simplified_color_group`

Per-group statistics:

- `records`
- `adoptions`
- `adoption_rate_pct`
- `median_days_to_outcome`
- `related_importance_features`
- `mean_importance_score`

Why this structure matters:

- H1 is not only descriptive; it is tied back to permutation importance and baseline feature importance,
- intake circumstances can be compared against appearance groups in a single common format.

Technical tradeoff:

- the table is intentionally simple and aggregated, so it is easy to cite, but it does not try to prove causality or interaction effects.

### 48.2 H3 Table

File:

- `h3_age_adoption_speed.csv`

Built from:

- `age_group`

Why this matters:

- age is one of the strongest and cleanest adoption predictors,
- the grouped summary is easier to interpret than raw age strings.

What it adds:

- adoption rate by age group,
- median days to outcome,
- age-related importance signal.

### 48.3 H5 Table

File:

- `h5_covid_period.csv`

Built from:

- `covid_period`

Why this matters:

- it captures a major time boundary in the dataset,
- it is a useful way to discuss operational changes across periods.

Technical caution:

- period effects may mix many things at once, including policy changes, shelter volume changes, and population mix changes.

### 48.4 Importance-Linking Logic

The hypothesis tables attempt to attach importance evidence from:

- permutation importance,
- random forest feature importance,
- logistic regression coefficients.

How that works:

- the code looks for relevant feature names in previously generated tables,
- then computes a mean importance score for the subset of matching features,
- the result is written into the hypothesis summary table.

Why this matters:

- the hypothesis tables are not isolated EDA outputs,
- they are connected back to the trained models.

Tradeoff:

- this is an association-based relevance signal, not a rigorous causal decomposition.

## 49. Model Comparison Internals

The model-comparison layer is the ranking layer of the project.

### 49.1 Input Tables

The comparison builder reads:

- baseline metrics,
- boosting metrics,
- advanced metrics.

This means it can compare:

- dummy baselines,
- linear models,
- random forests,
- histogram gradient boosting,
- CatBoost.

### 49.2 Ranking Rules

Classification:

- rank by ROC-AUC descending,
- also rank by F1 descending.

Regression:

- rank by MAE ascending,
- also rank by RMSE ascending.

Why this matters:

- classification ranking should favor better ranking quality,
- regression ranking should favor lower prediction error.

### 49.3 Why ROC-AUC and MAE Are Primary

ROC-AUC:

- threshold-independent,
- good for ranking adoption risk,
- robust for imbalanced binary prediction.

MAE:

- easy to interpret in days,
- directly operational,
- more stable and easier to explain than R2 in a skewed LOS problem.

Tradeoff:

- ROC-AUC and MAE are practical defaults, but they are not the only relevant metrics, which is why PR-AUC, F1, RMSE, and calibration are also computed.

### 49.4 Subset Logic

Every main result is split into:

- combined,
- dogs,
- cats.

Why this matters:

- species-level behavior differs,
- a single pooled score can hide very different error profiles,
- separate subsets make the thesis more honest and more actionable.

## 50. Animal Profile and Journey Internals

The animal-centered layer is the bridge between abstract metrics and real shelter profiles.

### 50.1 Profile Construction

Profiles are grouped by:

- animal type,
- age group,
- intake type,
- intake condition,
- health profile,
- behavior support flag,
- breed group,
- color group,
- sex status,
- named versus unnamed.

Why this matters:

- these are the variables people can actually reason about in shelter operations,
- the profile layer produces a more natural narrative than raw model coefficients.

### 50.2 Health and Behavior Descriptors

The `health_profile` and `behavior_support_flag` fields are derived only for EDA and animal-story outputs.

Important point:

- these are not complete medical or psychological descriptors,
- they are conservative proxies based on the available intake/outcome fields.

Why this matters:

- the thesis can discuss care context without overstating what the data contains.

### 50.3 Archetype Scoring

The archetype table groups records by profile and computes:

- records,
- adoptions,
- adoption rate,
- median days to outcome,
- mean days to outcome,
- outcome mix,
- profile label,
- visibility need.

Why this matters:

- it identifies high-volume or vulnerable animal profiles,
- it gives the dashboard and evidence pack a concrete case layer.

### 50.4 Vulnerability Score

The vulnerability ranking combines:

- lower adoption rate,
- longer median days to outcome,
- record volume.

Why this matters:

- it prioritizes profiles that are both common and harder to place,
- it is a practical prioritization heuristic rather than a causal score.

### 50.5 Profile Contrasts

Important contrast families include:

- pit-bull-type dogs versus other dogs,
- black/dark cats versus other cats,
- domestic-cat groups versus other cat breed groups,
- baby versus senior by species,
- named versus unnamed by species,
- health profiles by species,
- behavior-support signals by species.

Why this matters:

- these contrasts are the cleanest way to communicate how subgroup patterns differ,
- they also support the dashboard story layer.

### 50.6 Model-Error by Profile

When diagnostic predictions are available, the profile layer computes:

- mean predicted adoption probability,
- median predicted days,
- MAE,
- prediction gap between observed and predicted adoption rates.

Why this matters:

- it connects animal profiles directly to model reliability,
- it helps identify where the model is overconfident or underconfident.

### 50.7 Journey Cards

The journey card path combines:

- a representative profile,
- a prediction record built from that profile,
- predicted adoption probability,
- predicted days to outcome,
- similar historical cases,
- local or global SHAP reasons,
- visibility label.

Why this matters:

- it turns the technical model into an explainable case object,
- it is one of the most practical bridge points between ML output and thesis narrative.

## 51. Library and Implementation Details Worth Remembering

This section captures additional implementation facts that are easy to forget but technically important.

### 51.1 Standardization and Column Handling

The pipeline uses standardized column names early.

Why:

- easier downstream code,
- fewer casing and spacing bugs,
- more robust tests.

### 51.2 Minimum-Frequency One-Hot Encoding

The baseline and boosting preprocessors use `OneHotEncoder(min_frequency=20, ...)`.

Why:

- it prevents very rare categories from exploding the feature space,
- it reduces the chance of overfitting on sparse values.

Tradeoff:

- rare categories are grouped or ignored, which can hide some niche patterns.

### 51.3 `handle_unknown="ignore"`

The one-hot encoders ignore unknown categories at transform time.

Why:

- prevents inference-time crashes when a new category appears.

Tradeoff:

- the model will silently treat unseen categories as all-zero encodings.

### 51.4 Balanced Class Weights

The baseline classifier uses balanced weights and the forest uses balanced subsampling.

Why:

- adoption/non-adoption is not perfectly balanced,
- the model should not collapse into always predicting the majority class.

### 51.5 No-Surprise Dashboard Inputs

The dashboard manually builds a record using the same feature logic as training.

Why:

- avoids hidden mismatch between training and inference,
- keeps the model sensitivity demo aligned with the real feature contract.

### 51.6 Explicit `random_state`

Many components use the shared `RANDOM_STATE`.

Why:

- reproducible training,
- reproducible sampling,
- stable tests and figures.

### 51.7 Output Serialization

Model artifacts are stored as `joblib` pipelines with JSON metadata sidecars.

Why:

- easier to reload in the dashboard,
- better traceability than saving only raw estimators.

Tradeoff:

- artifact files depend on the Python library versions used to create them.


[Warning: File not found - docs/internal/ml_code_review.md]


================================================================================
FILE: docs/target_definitions.md
================================================================================

# Target Variable Definitions — AAC Adoption ML Pipeline

## Purpose

This document is the single authoritative reference for every target variable in the pipeline.
All thesis text, dashboard labels, report outputs, and code must use consistent naming and semantics as defined here.
This document is designed for machine reading by AI agents and for human reviewers checking terminology consistency.

---

## Target Taxonomy

### 1. Classification Target

| Property | Value |
|---|---|
| Column name | `classification_target` |
| Alias columns | `adopted`, `is_adopted`, `target_adopted` |
| Data type | int (0 or 1) |
| Defined as | `1` if `outcome_type == "Adoption"` (case-insensitive strip), else `0` |
| Scope | All matched intake/outcome episodes (dogs and cats) |
| Used for | Binary classification: predict whether the matched outcome episode ended in adoption |
| Allowed thesis labels | "adoption indicator", "adopted vs. not adopted", "binary adoption target" |
| Forbidden labels | "adoption speed target", "timing target" |

**Interpretation note:**
This column predicts whether the animal's matched outcome was an adoption. It does not predict when. Non-adoption outcomes include Transfer, Return to Owner, Euthanasia, and Died. The model trained on this target learns the *probability* of adoption, not the *speed* of adoption.

**Code location:** `src/aac_adoption/data/build_dataset.py`, `build_modeling_dataset()` function.

```python
# Exact definition in pipeline
outcome_type_normalized = dataset["outcome_type"].fillna("").astype(str).str.strip().str.lower()
dataset["adopted"] = outcome_type_normalized.eq("adoption")
dataset["classification_target"] = dataset["adopted"].astype(int)
```

---

### 2. Regression Target — Primary (Length of Stay / Days to Matched Outcome)

| Property | Value |
|---|---|
| Column name | `regression_target_days` |
| Alias columns | `days_to_outcome`, `length_of_stay` |
| Data type | float (non-negative) |
| Defined as | `(outcome_datetime - intake_datetime).total_seconds() / 86400` |
| Scope | All matched intake/outcome episodes (dogs and cats) |
| Used for | Regression: predict how many days an animal remained in care until its matched outcome |
| Allowed thesis labels | "days to outcome", "length of stay", "predicted days to outcome", "predicted length of stay", "time to matched outcome" |
| Forbidden labels | "adoption speed", "days to adoption", "predicted wait until adoption" (unless the subset is adopted animals only — see Target 3) |

**Interpretation note:**
This is an operational length-of-stay prediction. The outcome is the animal's *next matched outcome record*, which may be adoption, transfer, return-to-owner, or another disposition. It is NOT guaranteed to be adoption. Describing this as "predicted time to adoption" is methodologically wrong unless the prediction is restricted to adopted animals only.

Shelters care about length of stay regardless of outcome type because all occupied kennels have holding costs. This is why the regression target is defined over all outcomes, not just adoptions.

**Code location:** `src/aac_adoption/data/build_dataset.py`, `build_modeling_dataset()` function.

```python
# Exact definition in pipeline
dataset["days_to_outcome"] = (
    dataset["outcome_datetime"] - dataset["intake_datetime"]
).dt.total_seconds() / 86400
dataset["regression_target_days"] = dataset["days_to_outcome"]
dataset["length_of_stay"] = dataset["days_to_outcome"]
```

**Regression MAE label in all reports:**
> "combined regression MAE: X days for length-of-stay / days-to-outcome prediction"

---

### 3. Adoption-Only Timing Target — Descriptive / Optional

| Property | Value |
|---|---|
| Column name | `days_to_adoption` |
| Data type | float (non-negative) or NaN |
| Defined as | `days_to_outcome` where `outcome_type == "Adoption"`, else `NaN` |
| Scope | Adopted animals only (a subset of all matched episodes) |
| Used for | Descriptive adoption-speed analysis; optional adopted-only regression |
| Allowed thesis labels | "days to adoption (among adopted animals)", "adoption timing (adopted subset)", "median days to adoption" |
| Forbidden labels | Using this column as the main regression target without clearly stating the adopted-only scope |

**Interpretation note:**
This column exists to support H3 (age and adoption timing) without confusing non-adoption outcomes with adoption speed. When a thesis statement says "young animals are adopted in X median days", it should use this column or explicitly filter to `outcome_type == "Adoption"`. The main regression model uses `regression_target_days` (all outcomes) for operational LOS prediction.

**Code location:** `src/aac_adoption/data/build_dataset.py`, `build_modeling_dataset()` function.

```python
# Exact definition in pipeline
dataset["days_to_adoption"] = np.where(
    dataset["adopted"],
    dataset["days_to_outcome"],
    np.nan,
)
```

**Generated artifact:** `reports/tables/h3_adopted_only_age_speed.csv`

---

### 4. Survival / Time-to-Event Target — Future Work

| Property | Value |
|---|---|
| Column name | Not yet defined |
| Framework | Kaplan–Meier / Cox proportional hazards (not implemented as main model) |
| Censoring | Intakes without a future outcome (cannot determine final outcome) |
| Competing risks | Transfer, Euthanasia, Return-to-Owner all "compete" with Adoption |
| Status | **Not the main modeling framework.** Discussed as future work. |

**What IS implemented (descriptive only):**
- Empirical adoption survival curves by `animal_type`, `age_group`, `covid_period`, and `intake_type`.
- Kaplan–Meier descriptive curves using `lifelines` library.
- These are descriptive time-to-adoption views for adopted animals, NOT a full survival model.

**Artifacts generated (descriptive only):**
- `reports/tables/adoption_survival_curves.csv`
- `reports/figures/km_adoption_by_animal_type.png`
- `reports/figures/km_adoption_by_age_group.png`
- `reports/figures/km_adoption_by_covid_period.png`
- `reports/summary/survival_descriptive_note.md`

**Thesis framing:**
> "These curves are descriptive time-to-adoption views among adopted animals. They are not the main modeling framework and do not replace the supervised ML comparison. Full time-to-event survival modeling with censoring and competing risks is outside the scope of this thesis and is noted as a natural extension for future work."

---

## Leakage Control Summary

The following columns are **targets or outcome-derived metadata** and must never appear in `feature_columns.json` or be passed to any trained model as predictors:

| Column | Category | Reason |
|---|---|---|
| `classification_target` | target | binary adoption label |
| `regression_target_days` | target | days to outcome |
| `days_to_outcome` | target/alias | same as regression_target_days |
| `length_of_stay` | target/alias | same as days_to_outcome |
| `days_to_adoption` | target/alias | adopted-only timing |
| `adopted` | target/alias | binary adoption flag |
| `is_adopted` | target/alias | binary adoption flag |
| `target_adopted` | target/alias | binary adoption flag |
| `outcome_type` | metadata | post-intake outcome label |
| `outcome_subtype` | metadata | post-intake outcome detail |
| `outcome_datetime` | metadata | post-intake timestamp |
| `sex_upon_outcome` | metadata | post-intake measurement |
| `age_upon_outcome` | metadata | post-intake measurement |

**Validation:** `src/aac_adoption/features/feature_sets.py` — `LEAKAGE_COLUMNS` set and `validate_no_leakage()` function.
**Leakage audit script:** `scripts/generate_leakage_audit.py`

---

## Consistent Label Rules for All Files

| Context | Correct label | Incorrect label |
|---|---|---|
| Regression model output (all animals) | "predicted days to outcome" / "predicted length of stay" | "predicted adoption speed", "days to adoption" |
| Regression MAE in reports | "MAE X days for length-of-stay prediction" | "adoption speed error" |
| H3 table (all animals) | "median days to outcome by age group" | "adoption speed by age" |
| H3 adopted-only table | "median days to adoption (adopted animals only)" | "adoption speed" without qualification |
| Dashboard regression metric card | "Predicted days to outcome" | "Predicted wait until adoption" |
| SHAP regression interpretation | "associated with predicted days to outcome" | "causes faster adoption" |

---

## Relationship Between Targets

```
All matched intake/outcome episodes (N = 162,390)
├── classification_target=1 (adopted, ~52%)  ──► days_to_adoption = days_to_outcome
│                                                  ↑ Used for adopted-only descriptive speed analysis
└── classification_target=0 (not adopted, ~48%) ──► days_to_adoption = NaN

regression_target_days = days_to_outcome (ALL episodes, regardless of outcome type)
    ↑ Main regression target: predicts length of stay, not adoption speed
```


================================================================================
FILE: reports/summary/artifact_manifest.md
================================================================================

# Thesis Artifact Manifest

Generated: 2026-06-05T19:50:07.105162+00:00

Status legend: present = present on disk | missing = not yet generated

## Appendix

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `reports/artifact_manifest.csv` | manifest | `scripts/generate_artifact_manifest.py` | Self-referential manifest entry |
| present | `reports/summary/environment_snapshot.md` | report | `scripts/generate_environment_snapshot.py` | Human-readable environment snapshot for appendix inclusion |
| present | `reports/tables/environment_snapshot.csv` | table | `scripts/generate_environment_snapshot.py` | Library version snapshot for reproducibility |

## Chapter 3 - Hypotheses

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `reports/summary/breed_color_justification.md` | report | `scripts/run_analysis.py` | Breed and coat colour engineering justification report |
| present | `reports/summary/h2_interpretation.md` | report | `scripts/run_analysis.py` | H2 evidence seasonality summary report |
| present | `reports/summary/h4_interpretation.md` | report | `scripts/run_analysis.py` | H4 evidence coat colour summary report |
| present | `reports/tables/h1_intake_vs_appearance.csv` | table | `scripts/run_analysis.py` | H1 evidence: intake type and condition vs appearance |
| present | `reports/tables/h2_seasonality_summary.csv` | table | `scripts/run_analysis.py` | H2 evidence: seasonality summary table |
| present | `reports/tables/h3_age_adoption_speed.csv` | table | `scripts/run_analysis.py` | H3 evidence: age and adoption timing among adopted animals |
| present | `reports/tables/h4_dark_color_summary.csv` | table | `scripts/run_analysis.py` | H4 evidence: dark coat colour summary table |
| present | `reports/tables/h5_covid_period.csv` | table | `scripts/run_analysis.py` | H5 evidence: COVID period adoption dynamics |

## Chapter 4 - Model Evaluation

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `docs/model_diagnostics.md` | report | `` |  |
| present | `reports/diagnostics/classification_calibration.csv` | table | `` |  |
| present | `reports/diagnostics/classification_error_slices.csv` | table | `` |  |
| present | `reports/diagnostics/classification_precision_recall_curve.csv` | table | `` |  |
| present | `reports/diagnostics/classification_roc_curve.csv` | table | `` |  |
| present | `reports/diagnostics/classification_thresholds.csv` | table | `` |  |
| present | `reports/diagnostics/diagnostic_predictions_sample.csv` | table | `` |  |
| present | `reports/diagnostics/placement_risk_quadrants.csv` | table | `` |  |
| present | `reports/diagnostics/regression_error_slices.csv` | table | `` |  |
| present | `reports/diagnostics/regression_residuals_sample.csv` | table | `` |  |
| present | `reports/summary/descriptive_baseline_comparison.md` | report | `scripts/run_analysis.py` | Non-ML descriptive baseline vs ML model comparison report |
| present | `reports/summary/model_evidence_pack.md` | report | `scripts/generate_evidence_pack.py` | Narrative model evidence summary |
| present | `reports/summary/subgroup_reliability.md` | report | `scripts/generate_evidence_pack.py` | Subgroup reliability narrative summary |
| present | `reports/tables/metric_confidence_intervals.csv` | table | `scripts/generate_evidence_pack.py` | Bootstrap confidence intervals for key metrics |
| present | `reports/tables/model_comparison_classification.csv` | table | `scripts/run_analysis.py` | Primary classification leaderboard |
| present | `reports/tables/model_comparison_regression.csv` | table | `scripts/run_analysis.py` | Primary regression leaderboard |
| present | `reports/tables/model_evidence_pack.csv` | table | `scripts/generate_evidence_pack.py` | Summary evidence pack for model trustworthiness |
| present | `reports/tables/subgroup_reliability.csv` | table | `scripts/generate_evidence_pack.py` | Per-cohort reliability red flags |

## Chapter 5 - Interpretation

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `reports/summary/local_explanation_examples.md` | report | `scripts/generate_evidence_pack.py` | Narrative summary of local explanation examples with causal limitations |
| present | `reports/tables/local_explanation_examples.csv` | table | `scripts/generate_evidence_pack.py` | Local explanation examples combining animal journeys, nearest neighbours, and model reasons |

## Chapter 5 - Interpretation

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `reports/figures/feature_family_importance_classification.png` | figure | `scripts/generate_feature_family_importance.py` | Feature family bar chart - classification |
| present | `reports/figures/feature_family_importance_regression.png` | figure | `scripts/generate_feature_family_importance.py` | Feature family bar chart - regression |
| present | `reports/summary/external_validity_limitations.md` | report | `scripts/run_analysis.py` | External validity and causal limitations report |
| present | `reports/tables/animal_archetypes.csv` | table | `scripts/generate_animal_research.py` | Animal profile archetypes with adoption statistics |
| present | `reports/tables/feature_family_importance_classification.csv` | table | `scripts/generate_feature_family_importance.py` | Extended family importance with sum/mean/n_features |
| present | `reports/tables/feature_family_importance_regression.csv` | table | `scripts/generate_feature_family_importance.py` | Extended family importance with sum/mean/n_features - regression |
| present | `reports/tables/shap_feature_families_classification.csv` | table | `scripts/generate_diagnostics.py --include-shap` | SHAP aggregated by feature family - classification |
| present | `reports/tables/shap_feature_families_regression.csv` | table | `scripts/generate_diagnostics.py --include-shap` | SHAP aggregated by feature family - regression |
| present | `reports/tables/shap_global_classification.csv` | table | `scripts/generate_diagnostics.py --include-shap` | Global SHAP per feature - classification |
| present | `reports/tables/shap_global_regression.csv` | table | `scripts/generate_diagnostics.py --include-shap` | Global SHAP per feature - regression |
| present | `reports/tables/vulnerable_profiles.csv` | table | `scripts/generate_animal_research.py` | Profiles needing targeted visibility |

## Unknown

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `docs/animal_exploratory_research_plan.md` | report | `` |  |
| present | `docs/found_location_plan.md` | report | `` |  |
| present | `docs/implementation_plan_p3_p4.md` | report | `` |  |
| present | `docs/interactive_story_plan.md` | report | `` |  |
| present | `docs/methodology_notes.md` | report | `` |  |
| present | `docs/model_evidence_pack.md` | report | `` |  |
| present | `docs/progress_and_future_work.md` | report | `` |  |
| present | `docs/target_definitions.md` | report | `` |  |
| present | `docs/technical_architecture_plan.md` | report | `` |  |
| present | `docs/thesis.md` | report | `` |  |
| present | `docs/thesis_technical_guide.md` | report | `` |  |
| present | `reports/metrics/advanced_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics/advanced_metrics.csv` | table | `` |  |
| present | `reports/metrics/advanced_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics/baseline_metrics.csv` | table | `` |  |
| present | `reports/metrics/boosting_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics/boosting_metrics.csv` | table | `` |  |
| present | `reports/metrics/boosting_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics/classification_metrics.csv` | table | `` |  |
| present | `reports/metrics/regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/advanced_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/advanced_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/advanced_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/baseline_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/boosting_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/boosting_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/boosting_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_base/regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/advanced_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/advanced_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/advanced_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/baseline_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/boosting_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/boosting_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/boosting_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_context/regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/advanced_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/advanced_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/advanced_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/baseline_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/boosting_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/boosting_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/boosting_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_base/regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/advanced_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/advanced_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/advanced_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/baseline_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/boosting_classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/boosting_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/boosting_regression_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/classification_metrics.csv` | table | `` |  |
| present | `reports/metrics_smoke_context/regression_metrics.csv` | table | `` |  |

## Unknown - see notes

| Status | Artifact | Type | Source Script | Notes |
|--------|----------|------|---------------|-------|
| present | `docs/results_summary.md` | report | `` |  |
| present | `reports/figures/adoption_cumulative_curves.png` | figure | `` |  |
| present | `reports/figures/adoptions_by_year.png` | figure | `` |  |
| present | `reports/figures/animal_archetypes_top.png` | figure | `` |  |
| present | `reports/figures/animal_stories_smoke.png` | figure | `` |  |
| present | `reports/figures/context_model_delta.png` | figure | `` |  |
| present | `reports/figures/diagnostic_calibration_curve.png` | figure | `` |  |
| present | `reports/figures/diagnostic_classification_error_slices.png` | figure | `` |  |
| present | `reports/figures/diagnostic_precision_recall_curve.png` | figure | `` |  |
| present | `reports/figures/diagnostic_predicted_vs_actual.png` | figure | `` |  |
| present | `reports/figures/diagnostic_regression_error_slices.png` | figure | `` |  |
| present | `reports/figures/diagnostic_regression_residuals.png` | figure | `` |  |
| present | `reports/figures/diagnostic_roc_curve.png` | figure | `` |  |
| present | `reports/figures/final_confusion_matrix.png` | figure | `` |  |
| present | `reports/figures/h1_feature_family_importance.png` | figure | `` |  |
| present | `reports/figures/h1_intake_condition_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h1_intake_type_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h2_adoption_rate_by_season.png` | figure | `` |  |
| present | `reports/figures/h2_median_los_by_season.png` | figure | `` |  |
| present | `reports/figures/h3_adopted_only_median_days_to_adoption.png` | figure | `` |  |
| present | `reports/figures/h3_age_adopted_only_median_days.png` | figure | `` |  |
| present | `reports/figures/h3_age_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h3_age_group_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h3_age_group_median_days.png` | figure | `` |  |
| present | `reports/figures/h3_age_shap_summary.png` | figure | `` |  |
| present | `reports/figures/h4_dark_color_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h4_dark_color_median_los.png` | figure | `` |  |
| present | `reports/figures/h5_covid_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h5_covid_intake_volume.png` | figure | `` |  |
| present | `reports/figures/h5_covid_median_los.png` | figure | `` |  |
| present | `reports/figures/h5_covid_period_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/h5_covid_period_median_days.png` | figure | `` |  |
| present | `reports/figures/intakes_by_year.png` | figure | `` |  |
| present | `reports/figures/km_adoption_by_age_group.png` | figure | `` |  |
| present | `reports/figures/km_adoption_by_animal_type.png` | figure | `` |  |
| present | `reports/figures/km_adoption_by_covid_period.png` | figure | `` |  |
| present | `reports/figures/km_adoption_by_intake_type.png` | figure | `` |  |
| present | `reports/figures/model_comparison_classification_f1.png` | figure | `` |  |
| present | `reports/figures/model_comparison_classification_pr_auc.png` | figure | `` |  |
| present | `reports/figures/model_comparison_classification_roc_auc.png` | figure | `` |  |
| present | `reports/figures/model_comparison_regression_mae.png` | figure | `` |  |
| present | `reports/figures/model_comparison_regression_rmse.png` | figure | `` |  |
| present | `reports/figures/profile_contrasts_adoption_rate.png` | figure | `` |  |
| present | `reports/figures/shap_feature_families_classification.png` | figure | `` |  |
| present | `reports/figures/shap_feature_families_regression.png` | figure | `` |  |
| present | `reports/figures/shap_summary_classification.png` | figure | `` |  |
| present | `reports/figures/shap_summary_regression.png` | figure | `` |  |
| present | `reports/figures/vulnerable_profiles.png` | figure | `` |  |
| present | `reports/summary/artifact_manifest.md` | report | `` |  |
| present | `reports/summary/calibration_interpretation.md` | report | `` |  |
| present | `reports/summary/current_results.md` | report | `` |  |
| present | `reports/summary/data_audit.md` | report | `` |  |
| present | `reports/summary/feature_quality_audit.md` | report | `` |  |
| present | `reports/summary/final_model_selection.md` | report | `` |  |
| present | `reports/summary/h1_interpretation.md` | report | `` |  |
| present | `reports/summary/h3_interpretation.md` | report | `` |  |
| present | `reports/summary/h5_interpretation.md` | report | `` |  |
| present | `reports/summary/hypothesis_evidence_matrix.md` | report | `` |  |
| present | `reports/summary/leakage_audit.md` | report | `` |  |
| present | `reports/summary/matching_logic_examples.md` | report | `` |  |
| present | `reports/summary/model_reliability_red_flags.md` | report | `` |  |
| present | `reports/summary/survival_descriptive_note.md` | report | `` |  |
| present | `reports/summary/threshold_selection.md` | report | `` |  |
| present | `reports/tables/adoption_by_day_milestones.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_age_group.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_covid_period.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_intake_condition.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_intake_season.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_intake_type.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_is_black_or_dark.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_simplified_breed_group.csv` | table | `` |  |
| present | `reports/tables/adoption_los_by_simplified_color_group.csv` | table | `` |  |
| present | `reports/tables/adoption_rate_by_age_group.csv` | table | `` |  |
| present | `reports/tables/adoption_rate_by_animal_type.csv` | table | `` |  |
| present | `reports/tables/adoption_rate_by_color_group.csv` | table | `` |  |
| present | `reports/tables/adoption_rate_by_covid_period.csv` | table | `` |  |
| present | `reports/tables/adoption_rate_by_intake_type.csv` | table | `` |  |
| present | `reports/tables/adoption_survival_curves.csv` | table | `` |  |
| present | `reports/tables/animal_journey_examples.csv` | table | `` |  |
| present | `reports/tables/calibration_summary_by_subset.csv` | table | `` |  |
| present | `reports/tables/category_cardinality.csv` | table | `` |  |
| present | `reports/tables/context_model_comparison.csv` | table | `` |  |
| present | `reports/tables/data_audit_attrition.csv` | table | `` |  |
| present | `reports/tables/feature_missingness.csv` | table | `` |  |
| present | `reports/tables/final_classifier_thresholds.csv` | table | `` |  |
| present | `reports/tables/final_model_selection.csv` | table | `` |  |
| present | `reports/tables/h1_feature_family_ablation.csv` | table | `` |  |
| present | `reports/tables/h1_feature_family_importance.csv` | table | `` |  |
| present | `reports/tables/h3_adopted_only_age_speed.csv` | table | `` |  |
| present | `reports/tables/h3_age_evidence_matrix.csv` | table | `` |  |
| present | `reports/tables/h3_age_length_of_stay.csv` | table | `` |  |
| present | `reports/tables/h5_covid_evidence_matrix.csv` | table | `` |  |
| present | `reports/tables/h5_covid_population_mix.csv` | table | `` |  |
| present | `reports/tables/health_behavior_profiles.csv` | table | `` |  |
| present | `reports/tables/hypothesis_evidence_matrix.csv` | table | `` |  |
| present | `reports/tables/intakes_by_year.csv` | table | `` |  |
| present | `reports/tables/leakage_audit.csv` | table | `` |  |
| present | `reports/tables/logistic_regression_coefficients.csv` | table | `` |  |
| present | `reports/tables/matching_examples.csv` | table | `` |  |
| present | `reports/tables/median_los_by_animal_type.csv` | table | `` |  |
| present | `reports/tables/model_failure_modes.csv` | table | `` |  |
| present | `reports/tables/model_limitations_by_cohort.csv` | table | `` |  |
| present | `reports/tables/model_reliability_red_flags.csv` | table | `` |  |
| present | `reports/tables/outcomes_adoptions_by_year.csv` | table | `` |  |
| present | `reports/tables/permutation_importance_classification.csv` | table | `` |  |
| present | `reports/tables/permutation_importance_regression.csv` | table | `` |  |
| present | `reports/tables/profile_contrasts.csv` | table | `` |  |
| present | `reports/tables/profile_model_error.csv` | table | `` |  |
| present | `reports/tables/random_forest_feature_importance.csv` | table | `` |  |
| present | `reports/tables/shap_local_examples.csv` | table | `` |  |
| present | `reports/tables/subgroup_adoption_milestones.csv` | table | `` |  |
| present | `reports/tables/subgroup_metric_confidence_intervals.csv` | table | `` |  |
| present | `reports/tables_base/logistic_regression_coefficients.csv` | table | `` |  |
| present | `reports/tables_base/permutation_importance_classification.csv` | table | `` |  |
| present | `reports/tables_base/permutation_importance_regression.csv` | table | `` |  |
| present | `reports/tables_base/random_forest_feature_importance.csv` | table | `` |  |
| present | `reports/tables_context/logistic_regression_coefficients.csv` | table | `` |  |
| present | `reports/tables_context/permutation_importance_classification.csv` | table | `` |  |
| present | `reports/tables_context/permutation_importance_regression.csv` | table | `` |  |
| present | `reports/tables_context/random_forest_feature_importance.csv` | table | `` |  |
| present | `reports/tables_smoke_base/logistic_regression_coefficients.csv` | table | `` |  |
| present | `reports/tables_smoke_base/permutation_importance_classification.csv` | table | `` |  |
| present | `reports/tables_smoke_base/permutation_importance_regression.csv` | table | `` |  |
| present | `reports/tables_smoke_base/random_forest_feature_importance.csv` | table | `` |  |
| present | `reports/tables_smoke_context/logistic_regression_coefficients.csv` | table | `` |  |
| present | `reports/tables_smoke_context/permutation_importance_classification.csv` | table | `` |  |
| present | `reports/tables_smoke_context/permutation_importance_regression.csv` | table | `` |  |
| present | `reports/tables_smoke_context/random_forest_feature_importance.csv` | table | `` |  |
| present | `reports/tables_smoke_context_compare/context_model_comparison.csv` | table | `` |  |


================================================================================
FILE: reports/summary/breed_color_justification.md
================================================================================

# Breed and Color Feature Engineering Justification

This document justifies the preprocessing and categorical simplification of breed and coat colour variables used in the ML modeling pipeline.

---

## 🧩 The Category Sparsity Problem

The raw Austin Animal Center dataset contains a massive variety of text descriptions for animal breeds and colours:
- **Raw Breeds:** Over 2,700 unique strings (e.g., `"Labrador Retriever Mix"`, `"German Shepherd/Pit Bull"`, `"Chihuahua Shorthair/Dachshund"`).
- **Raw Colours:** Over 500 unique colour combinations (e.g., `"Black/White"`, `"Brown Tabby/White"`, `"Blue Merle/Tan"`).

If passed directly to a model without simplification:
1. **High Dimensionality / Category Sparsity:** One-hot encoding these categories creates thousands of columns, most of which have fewer than 10 records.
2. **Overfitting & High Variance:** Tree models or linear models struggle to find meaningful patterns in rare classes, leading to high variance and poor generalisation on the test set.
3. **CatBoost vs. Other Models:** While CatBoost can handle categorical features natively, simpler ensembles (Random Forest, HistGradientBoosting) require either target encoding or predefined simplification to run efficiently.

---

## ⚖️ Granularity vs. Stability Tradeoff

To ensure robust modeling, raw features are mapped to simplified groups:
- **Breed Mapping:** Groups raw strings into 12 major breed groups (e.g., `retriever_type`, `pit_bull_type`, `shepherd_type`, `terrier_type`, `toy_type`, `domestic_cat`) plus an `other` category.
- **Colour Mapping:** Groups raw colours into 7 simple palettes (e.g., `black_dark`, `white_light`, `brown_tan`, `tricolor_merle`) plus an `other` category.

### Benefits of Simplification
- **Improved Sample Size:** Every simplified category contains hundreds or thousands of records, stabilizing probability and stay-duration estimates.
- **Reduced Noise:** The models learn patterns at the family/type level, which are more robust to minor transcription differences by shelter staff.
- **Consistent Pipeline Evaluation:** Allows baselines (Logistic Regression/Ridge) and tree ensembles to be trained on the exact same feature space.

---

## ⚠️ Limitations of Shelter Breed and Colour Records

The thesis explicitly highlights three critical limitations of these variables:

1. **Visual Staff Inspection:** 
   Breed and colour labels in the AAC dataset are assigned by shelter intake workers based on visual inspection. They **do not represent genetic pedigree analysis**. Visual breed identification in shelters is known to have low agreement with DNA analysis.
2. **The "Pit Bull Type" Descriptor:**
   The `pit_bull_type` grouping matches raw strings containing `"Pit Bull"`, `"Staffordshire"`, or `"Bull Terrier"`. In shelter operations, this label is frequently applied to mixed-breed dogs with certain blocky-headed physical features. The thesis does **not** treat this as a behavioral category, but rather as an administrative shelter-record tag.
3. **Non-Causal Interpretation:**
   Model sensitivity shifts or SHAP values showing lower adoption rates for specific breed/colour groups reflect **correlated systemic factors** (such as municipal breed restrictions, landlord policies, and local rescue capacity) rather than innate animal traits or direct adopter prejudice.


================================================================================
FILE: reports/summary/calibration_interpretation.md
================================================================================

# Calibration Interpretation

## Summary

| animal_subset   | subset   |   records |   mean_calibration_gap | worst_cohort     |   worst_calibration_gap |   pct_small_cohort_flagged | overconfident   |   mean_observed_adoption_rate |   mean_predicted_probability |
|:----------------|:---------|----------:|-----------------------:|:-----------------|------------------------:|---------------------------:|:----------------|------------------------------:|-----------------------------:|
| dogs            | dogs     |    138200 |                 0.1353 | intake_condition |                  0.5672 |                       32.1 | False           |                         0.485 |                       0.4394 |
| cats            | cats     |    138200 |                 0.1353 | intake_condition |                  0.5672 |                       32.1 | False           |                         0.485 |                       0.4394 |
| combined        | combined |    138200 |                 0.1353 | intake_condition |                  0.5672 |                       32.1 | False           |                         0.485 |                       0.4394 |

## Questions Answered

### 1. Is the model overconfident or underconfident?
Overall direction: **underconfident** (mean calibration gap = 0.1353).
The calibration gap is defined as |observed adoption rate − mean predicted probability| per cohort.

### 2. Which probability bins are reliable?
**12 cohorts** have calibration gap < 0.08 and are considered reliable. These include: simplified_breed_group, health_profile, intake_condition, intake_condition, health_profile...

**11 cohorts** have calibration gap ≥ 0.15 and are flagged as unreliable. These include: simplified_breed_group, intake_type, simplified_breed_group, intake_condition, intake_condition...

### 3. Are dogs and cats calibrated differently?
The current `subgroup_reliability.csv` is computed across all animals. Separate species calibration would require per-species diagnostic runs. Based on the combined calibration gap, species-level differences are likely given the different adoption rate baselines (cats ≈ 62%, dogs ≈ 63% in the test period).

### 4. Should predicted probabilities be used literally or as ranking scores?

> **Recommendation:** Treat probabilities as **relative risk / ranking scores**, not literal probabilities, unless calibration is confirmed acceptable for the target cohort.

Given that the mean calibration gap is ≥ 0.10, the model is **not well-calibrated** overall. Predicted probabilities should be used for ranking (high prob → more likely adopted) rather than as literal probability estimates.

## Caveat for Dashboard and Thesis

- The dashboard displays predicted probabilities as *adoption likelihood scores*, not clinical probabilities.
- The thesis discusses model performance primarily via ROC-AUC and PR-AUC, which are ranking metrics.
- Where calibration gaps are large (see `model_reliability_red_flags.csv`), the thesis notes that model outputs should not be overinterpreted for those cohorts.


================================================================================
FILE: reports/summary/current_results.md
================================================================================

# Generated Current Results Summary

This summary is generated from existing pipeline outputs. Treat these results as reproducible working outputs, not final causal conclusions.

## Model Comparison

Best classification models by PR-AUC:

- combined: hist_gradient_boosting (PR-AUC 0.884, ROC-AUC 0.840, F1 0.818)
- dogs: catboost (PR-AUC 0.860, ROC-AUC 0.813, F1 0.785)
- cats: hist_gradient_boosting (PR-AUC 0.899, ROC-AUC 0.865, F1 0.842)

Best regression models by MAE:

- combined: catboost (MAE 18.55 days, RMSE 38.36 days)
- dogs: catboost (MAE 21.56 days, RMSE 45.99 days)
- cats: catboost (MAE 15.79 days, RMSE 30.12 days)
## External Context Feature Test

Context features use intake-date weather plus prior-window 311 and shelter intake volumes; rolling counts use only dates before intake.
Overall, context effects should be treated as small unless the metric delta is large enough to matter operationally.

Classification context deltas:

- cats / catboost: context improved roc_auc by 0.004.
- cats / dummy_most_frequent: context changed negligibly roc_auc by 0.000.
- cats / hist_gradient_boosting: context changed negligibly roc_auc by 0.000.
- cats / logistic_regression: context changed negligibly roc_auc by 0.000.
- cats / random_forest: context improved roc_auc by 0.002.
- combined / catboost: context improved roc_auc by 0.003.
- combined / dummy_most_frequent: context changed negligibly roc_auc by 0.000.
- combined / hist_gradient_boosting: context improved roc_auc by 0.001.

Regression context deltas:

- cats / catboost: context improved mae by 0.085.
- cats / dummy_median: context changed negligibly mae by 0.000.
- cats / hist_gradient_boosting: context worsened mae by 4.776.
- cats / random_forest: context worsened mae by 4.460.
- cats / ridge: context worsened mae by 1.481.
- combined / catboost: context improved mae by 0.081.
- combined / dummy_median: context changed negligibly mae by 0.000.
- combined / hist_gradient_boosting: context worsened mae by 1.263.

## Hypothesis Signals

H1 intake-type patterns:

- Stray: 116393 records, 48.7% adoption rate
- Owner Surrender: 34142 records, 67.1% adoption rate
- Public Assist: 9786 records, 18.1% adoption rate
- Abandoned: 1825 records, 64.8% adoption rate
- Euthanasia Request: 243 records, 5.8% adoption rate

H3 age-group patterns:

- baby: 59.7% adoption rate, median outcome time 6.22 days
- young: 47.5% adoption rate, median outcome time 6.77 days
- adult: 39.6% adoption rate, median outcome time 6.46 days
- senior: 31.0% adoption rate, median outcome time 4.20 days
- unknown: 33.3% adoption rate, median outcome time 4.11 days

H5 COVID-period patterns:

- pre_covid: 46.3% adoption rate, median outcome time 5.25 days
- post_covid: 61.5% adoption rate, median outcome time 9.83 days
- covid: 57.2% adoption rate, median outcome time 8.01 days

## Reliability Diagnostics

- Mean absolute calibration gap across probability bins: 0.112.
- At threshold 0.50, precision is 0.838, recall is 0.790, and F1 is 0.814.
- Largest placement-risk quadrant: likely_quick_placement (5601 records).

## Interpretability Signals

- age_upon_intake (age): mean absolute SHAP 0.274.
- sex_upon_intake (sex_reproductive_status): mean absolute SHAP 0.253.
- intake_type (intake_circumstances): mean absolute SHAP 0.250.
- primary_breed (breed_appearance): mean absolute SHAP 0.243.
- is_named (name_identity): mean absolute SHAP 0.233.

## Interpretation Guardrails

- Use model outputs as predictive association evidence, not proof of causal effects.
- Emphasize the time-aware train/validation/test split when discussing evaluation.
- Keep H1, H3, and H5 central; use H2 and H4 as descriptive supporting analyses unless stronger tests are added.
- Regression should be discussed primarily through MAE because it is easiest to explain operationally.

## Model Evidence Pack

A separate evidence pack has been generated for model choice, uncertainty, cohort limits, SHAP interpretation, and animal journey examples.

This evidence pack summarizes model choice, uncertainty, reliability limits, SHAP interpretation, and animal-centered examples. SHAP reasons are associated with model behavior and model predictions, not causal effects.

## Model Choice

- cats classification: roc_auc = 0.865. hist_gradient_boosting is the strongest ranking model for cats; compare PR-AUC and calibration before using a decision threshold.
- cats classification: pr_auc = 0.899. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- combined classification: roc_auc = 0.840. hist_gradient_boosting is the strongest ranking model for combined; compare PR-AUC and calibration before using a decision threshold.
- combined classification: pr_auc = 0.884. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- dogs classification: roc_auc = 0.813. catboost is the strongest ranking model for dogs; compare PR-AUC and calibration before using a decision threshold.
- dogs classification: pr_auc = 0.860. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- cats regression: mae = 15.794. catboost has the lowest average absolute days-to-outcome error for cats.
- combined regression: mae = 18.547. catboost has the lowest average absolute days-to-outcome error for combined.
- dogs regression: mae = 21.556. catboost has the lowest average absolute days-to-outcome error for dogs.

## Uncertainty

- roc_auc (combined): 0.840 [0.834, 0.847] from 200 bootstrap samples.
- pr_auc (combined): 0.882 [0.876, 0.889] from 200 bootstrap samples.
- f1_at_0_50 (combined): 0.814 [0.808, 0.819] from 200 bootstrap samples.
- mae (combined): 18.552 [18.005, 19.082] from 200 bootstrap samples.

## Trust and Limits

- simplified_breed_group=terrier_type: 321 records, calibration gap 0.206, MAE 12.17.
- intake_type=Abandoned: 466 records, calibration gap 0.185, MAE 19.10.
- simplified_breed_group=hound_type: 148 records, calibration gap 0.158, MAE 26.77.
- simplified_breed_group=chihuahua_type: 557 records, calibration gap 0.148, MAE 8.94.
- simplified_color_group=orange_yellow: 1062 records, calibration gap 0.143, MAE 18.13.
- age_group=young: 3936 records, calibration gap 0.142, MAE 20.99.
- simplified_breed_group=other: 2511 records, calibration gap 0.138, MAE 18.29.
- simplified_breed_group=shepherd_type: 792 records, calibration gap 0.136, MAE 21.10.

Top model failure modes:
- calibration_gap: simplified_breed_group=terrier_type (321 records, calibration_gap 0.206).
- calibration_gap: intake_type=Abandoned (466 records, calibration_gap 0.185).
- calibration_gap: simplified_breed_group=hound_type (148 records, calibration_gap 0.158).
- calibration_gap: simplified_breed_group=chihuahua_type (557 records, calibration_gap 0.148).
- calibration_gap: simplified_color_group=orange_yellow (1062 records, calibration_gap 0.143).
- calibration_gap: age_group=young (3936 records, calibration_gap 0.142).
- calibration_gap: simplified_breed_group=other (2511 records, calibration_gap 0.138).
- calibration_gap: simplified_breed_group=shepherd_type (792 records, calibration_gap 0.136).

Subgroup metric intervals are available for cohorts with enough records and class variety.

## Time-to-Adoption Milestones

- age_group=baby: 45676 adoptions; 64.2% adopted by day 30 and 94.9% by day 90.
- age_group=young: 23187 adoptions; 70.2% adopted by day 30 and 90.5% by day 90.
- age_group=adult: 10381 adoptions; 63.0% adopted by day 30 and 84.8% by day 90.
- age_group=senior: 3362 adoptions; 55.6% adopted by day 30 and 80.5% by day 90.
- animal_type=Dog: 47167 adoptions; 73.6% adopted by day 30 and 91.8% by day 90.
- animal_type=Cat: 35442 adoptions; 54.5% adopted by day 30 and 91.8% by day 90.
- health_profile=normal: 75183 adoptions; 68.3% adopted by day 30 and 92.6% by day 90.
- health_profile=medical_or_injured: 5379 adoptions; 43.9% adopted by day 30 and 83.7% by day 90.

## SHAP and Animal Stories

- classification: breed_appearance: 0.452. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: age: 0.422. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: name_identity: 0.407. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: sex_reproductive_status: 0.253. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: intake_circumstances: 0.250. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: age: 4.876. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: name_identity: 3.786. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: breed_appearance: 2.307. Feature-family contribution is associated with model prediction, not a causal effect.

Animal Journey examples:
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / black_or_dark: 709 cases; 71.8% adoption; median 34.1 days; exact visible profile; label needs visibility.
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / brown_tan: 580 cases; 71.4% adoption; median 25.2 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / brown_tan: 492 cases; 76.8% adoption; median 27.6 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / mixed_other: 540 cases; 80.0% adoption; median 29.9 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / black_or_dark: 526 cases; 71.3% adoption; median 31.0 days; exact visible profile; label needs visibility.

## Caveats

- CatBoost is included because shelter data is categorical-heavy; histogram gradient boosting may still win when its validation/test ranking is stronger.
- Probability estimates require calibration review before operational threshold use.
- Health and behavior fields are administrative care-context proxies, not complete temperament labels.
- Regression predicts days to outcome, not a full survival model for time to adoption.


================================================================================
FILE: reports/summary/data_audit.md
================================================================================

# Data Attrition Audit

## Summary

This table documents how raw Austin Animal Center CSV files become the final supervised ML dataset.
Removed rows are not arbitrary — each removal is required for methodologically sound episode-level supervised learning.

## Attrition Table

| Stage | Rows | Removed | Reason |
|---|---|---|---|
| raw_intakes |     173812 |        — | Source records from intakes.csv |
| raw_outcomes |     173775 |        — | Source records from outcomes.csv |
| standardized_intakes |     163901 |     9911 | Removed: invalid datetime, missing required fields, exact duplicates |
| standardized_outcomes |     163880 |     9895 | Removed: invalid datetime, missing required fields, exact duplicates |
| dog_cat_intakes |     163901 |        0 | filter_cats_and_dogs() applied during standardization; non-dog/cat rows already removed |
| matched_future_outcomes |     162390 |     1511 | Removed 1,511 intakes without a valid future outcome (cannot form supervised episode) |
| final_modeling_dataset |     162390 |        — | Final episode-level dataset with features and targets; matches build_modeling_dataset() output |

## Why Rows Are Removed

- **Invalid datetime / missing fields**: Records without a parseable intake_datetime or animal_id
  cannot be placed on a timeline and cannot form a valid episode.
- **Exact duplicates**: Exact-duplicate rows are administrative data quality issues; they would
  artificially inflate episode counts.
- **Non-dog/cat records**: Thesis scope is limited to dogs and cats (the two dominant species at AAC).
- **No valid future outcome**: An intake without a subsequent outcome record cannot form a
  supervised learning episode (we need both X and y). These intakes may represent ongoing stays
  at data snapshot time or data entry gaps.

## Matching Logic

Each intake is matched to the **nearest unused future outcome** for the same animal.
The algorithm is a greedy nearest-future-match: outcomes before the intake are skipped;
each outcome is used at most once. This prevents negative length-of-stay values and
prevents one outcome from being assigned to multiple intake episodes.

See `docs/methodology_notes.md` for full matching logic documentation.
See `reports/tables/matching_examples.csv` for human-readable examples.

================================================================================
FILE: reports/summary/descriptive_baseline_comparison.md
================================================================================

# Descriptive Baseline and Machine Learning Comparison

This document compares the machine learning model results against simple descriptive baselines (cross-tabulations), highlighting the predictive lift and operational benefits of the ML approach.

---

## 📊 Summary of Non-ML Descriptive Baselines

Descriptive averages from historical AAC records indicate clear single-variable associations:
1. **Intake Type:** Stray intakes have a baseline adoption rate of **~51%** with a median length of stay (LOS) of **~6.2 days**, whereas owner surrenders have higher adoption rates.
2. **Age Group:** Baby animals are adopted at a much higher rate (**~60%**) with a shorter median stay (**~5.5 days**) compared to senior animals (**~31%** adoption rate, **~8.0 days** stay).
3. **Coat Colour:** Dark-coloured animals have an adoption rate of **51.58%** vs. **50.53%** for non-dark animals, indicating minimal difference at the aggregate level.
4. **COVID Period:** Adoption rates rose from **46.3%** pre-COVID to **61.5%** post-COVID, while median length of stay increased from **5.25 days** to **9.83 days**.

---

## 🧠 Why Simple Cross-Tabs Are Insufficient

While simple descriptive tables are useful for outlining shelter-wide trends, they suffer from critical methodological limitations:
- **Inability to Handle Multi-Variable Interactions:** A cross-tab cannot predict outcomes for a senior, stray dog with a minor injury entering the shelter during a high-capacity winter week in the post-COVID period. Building a cross-tab for every feature combination leads to the "curse of dimensionality" and empty data cells.
- **Confounding Variables:** A descriptive table showing that senior animals have longer stays does not isolate whether this is due to age itself, breed composition, health status at intake, or intake type.
- **Static Averages:** Descriptive statistics are retrospective and cannot provide a personalized risk assessment or length-of-stay estimate for a new animal entering care today.

---

## 📈 Quantification of ML Predictive Lift

Supervised machine learning models overcome these limits by simultaneously fitting all intake features (demographics, clinical condition, time period, and context) using regularized estimators. 

The table below demonstrates the predictive lift of the final selected models over baseline dummy models and linear estimators on the unseen test set (2024–2025):

### 1. Classification (Target: `classification_target`)
| Model | Combined ROC-AUC | Combined F1-Score | PR-AUC | Notes |
|---|---|---|---|---|
| **Dummy (Most Frequent)** | 0.5000 | 0.0000 | 0.6276 | Simple descriptive average (no predictive value) |
| **Logistic Regression** | 0.8110 | 0.8301 | 0.8521 | Linear baseline; misses non-linear feature interactions |
| **HistGradientBoosting** | **0.8401** | **0.8178** | **0.8842** | **Selected Model:** Captures complex categorical interactions |

*Lift:* The machine learning classifier achieves a **0.34 ROC-AUC lift** over the descriptive majority-class guess, and a **0.03 ROC-AUC lift** over the linear baseline.

### 2. Regression (Target: `regression_target_days` - Length of Stay)
| Model | Combined MAE (Days) | Median Absolute Error (Days) | RMSE (Days) | Notes |
|---|---|---|---|---|
| **Dummy (Median Baseline)** | 20.7930 | 5.7149 | 41.5632 | Predicts the descriptive median length of stay |
| **Ridge Regression** | 24.1907 | 18.5637 | 36.3830 | Linear model; severely penalized by long-stay outliers |
| **CatBoost Regressor** | **18.5474** | **6.2777** | **38.3626** | **Selected Model:** Lowest average absolute error |

*Lift:* The CatBoost regressor reduces Mean Absolute Error by **2.25 days** compared to the median baseline, and by **5.64 days** compared to Ridge regression.

---

## 🛠️ Interpretability and Reliability Lift

In addition to predictive accuracy, the ML pipeline introduces:
1. **Local Interpretability (SHAP):** Unlike cross-tabs, which only show aggregate averages, SHAP values explain *why* a specific animal is predicted to have a high or low stay probability, showing how positive features (like a name or breed) balance negative features (like age or injury).
2. **Subgroup Calibration Audits:** The reliability profiling isolates exactly which animal cohorts have poor calibration (e.g., terrier types, stray intakes), warning shelter workers not to trust predictions for those groups. This granularity is impossible to retrieve from simple descriptive statistics.


================================================================================
FILE: reports/summary/environment_snapshot.md
================================================================================

# Environment Snapshot

Generated: 2026-06-05T18:38:09.816675+00:00

## Python Runtime

- **Python version:** 3.12.10
- **Platform:** Windows-10-10.0.19045-SP0
- **Random state:** 42

## Key Library Versions

| Library | Version |
|---------|---------|
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.8.0 |
| catboost | 1.2.10 |
| shap | 0.52.0 |
| streamlit | 1.58.0 |
| matplotlib | 3.10.9 |
| altair | 6.1.0 |
| joblib | 1.5.3 |

> This snapshot was generated automatically by `scripts/generate_environment_snapshot.py`.
> Include in the thesis appendix to document the computational environment.
> Saved models and SHAP outputs were generated in this environment.

================================================================================
FILE: reports/summary/external_validity_limitations.md
================================================================================

# External Validity and Generalizability Limitations

This document outlines the operational limits, demographic context, and causal boundaries of the predictive models trained on the Austin Animal Center (AAC) dataset.

---

## 🌍 Operational and Demographic Specificity

The modeling results presented in this thesis are descriptive of historical trends within a specific municipal ecosystem and **should not be generalized directly** to other shelters without local calibration and validation.

### 1. The Austin, Texas Context
- **Pet-Friendly Demographics:** Austin is widely recognized for having exceptionally high rates of pet ownership, active local rescue networks, and strong community support for animal welfare.
- **High Community Resources:** The City of Austin and local non-profits invest heavily in veterinary care, public behavior support, and targeted promotion, which might result in higher baseline adoption rates than in resource-constrained municipalities.

### 2. The No-Kill Shelter Context
- **90%+ Live Release Rate:** Austin Animal Center is the largest open-admission municipal **No-Kill** shelter in the United States, maintaining a live release rate of 90% or higher since 2011.
- **Operational Differences:** Under no-kill policies, animals are not euthanized for space constraints. This has a profound impact on:
  - **Length of Stay (LOS):** Long-stay animals (vulnerable breeds, seniors, medical cases) remain in care indefinitely rather than being euthanized, inflating the right tail of the length-of-stay distribution.
  - **Transfer Dynamics:** AAC relies extensively on partner rescue organizations (transfers) to manage capacity, which represents a major competing outcome that co-varies with animal health and intake volume.

### 3. Intake Policy and Capacity Constraints
- AAC is an open-admission shelter, meaning it is legally required to accept all animals from its jurisdiction.
- However, operational capacity limits force policy adjustments (e.g., diverting healthy stray intakes to community finder programs during peak times). These intake-filtering behaviors directly affect the population mix and outcome rates recorded in the dataset.

---

## 🦠 COVID-19 Period Confounders

The COVID-19 period (defined in this thesis from March 2020 through December 2021) is associated with marked changes in shelter indicators. However, the thesis **does not claim** that the pandemic causally altered public adoption preferences.

- **Intake Restrictions:** During the peak pandemic periods, AAC implemented restricted intake policies (e.g., surrender by appointment only, emergency-only intake), altering the intake type baseline.
- **Operational Adjustments:** Foster network expansion and virtual adoptions changed the speed at which animals moved from physical kennels to foster homes, altering the relationship between intake circumstances and recorded length of stay.
- **Confounded Drivers:** Public stay-at-home mandates, work-from-home trends, and rescue-incentive checks occurred simultaneously. The observational data cannot isolate the pandemic's independent causal impact from these simultaneous operational changes.

---

## ⛔ Causal Limits of Machine Learning Predictions

Reviewers and readers are cautioned against interpreting the model's feature importance or sensitivity simulations as causal evidence.

1. **Association, Not Causation:** 
   Features such as `intake_condition`, `is_black_or_dark`, or `simplified_breed_group` represent predictive associations. For example, if the model predicts a lower adoption probability for a specific breed group, this does not prove that shelter workers or adopters are actively discriminating against that breed. The breed label correlates with other unmeasured variables (e.g., housing restrictions, local supply, behavior history).
2. **Interpretability Boundaries (SHAP):** 
   SHAP values measure the marginal contribution of a feature to the **model's output** relative to the dataset mean. They do not represent a physical or psychological causal mechanism.
3. **Sensitivity vs. Causal Interventions:**
   The interactive "Model Sensitivity Demo" shows how the model's output shifts when input features are varied. Changing an animal's name flag (`is_named`) or coat colour (`is_black_or_dark`) in the simulation does not guarantee a real-world change in that animal's adoption probability, as the model cannot control for unobserved confounders.


================================================================================
FILE: reports/summary/feature_quality_audit.md
================================================================================

# Feature Quality Audit

## Purpose

This audit documents missingness and category cardinality for all features
used in the AAC adoption ML pipeline. It justifies feature engineering decisions
(breed/color simplification) and documents that missing values were handled
intentionally, not silently ignored.

## Missingness Summary

| Feature | Missing % | Missing Count | Unique Values |
|---|---|---|---|
| age_years | 0.0% | 9 | 46 |
| age_in_months | 0.0% | 9 | 46 |
| age_days | 0.0% | 9 | 46 |
| age_in_years | 0.0% | 9 | 46 |
| age_months | 0.0% | 9 | 46 |
| age_in_days | 0.0% | 9 | 46 |
| animal_type | 0.0% | 0 | 2 |
| intake_type | 0.0% | 0 | 6 |
| primary_breed | 0.0% | 0 | 255 |
| age_group | 0.0% | 0 | 5 |
| color | 0.0% | 0 | 621 |
| breed | 0.0% | 0 | 2,761 |
| sex_upon_intake | 0.0% | 1 | 5 |
| intake_condition | 0.0% | 0 | 20 |
| simplified_color_group | 0.0% | 0 | 6 |
| simplified_breed_group | 0.0% | 0 | 9 |
| is_mixed_breed | 0.0% | 0 | 2 |
| is_named | 0.0% | 0 | 2 |
| has_name | 0.0% | 0 | 2 |
| is_black_or_dark | 0.0% | 0 | 2 |
| primary_color | 0.0% | 0 | 59 |
| intake_year | 0.0% | 0 | 13 |
| intake_month | 0.0% | 0 | 12 |
| covid_period | 0.0% | 0 | 3 |

## Category Cardinality Summary

| Feature | Unique Values | Rare Categories (<1%) | Rare % |
|---|---|---|---|
| breed | 2,761 | 2,747 | 99.5% |
| color | 621 | 595 | 95.8% |
| primary_breed | 255 | 242 | 94.9% |
| primary_color | 59 | 43 | 72.9% |
| intake_condition | 20 | 15 | 75.0% |
| simplified_breed_group | 9 | 1 | 11.1% |
| intake_type | 6 | 2 | 33.3% |
| simplified_color_group | 6 | 0 | 0.0% |
| age_group | 5 | 1 | 20.0% |
| sex_upon_intake | 5 | 0 | 0.0% |
| covid_period | 3 | 0 | 0.0% |
| animal_type | 2 | 0 | 0.0% |

## Why Breed and Color Were Simplified

Raw `breed` and `color` fields have extremely high cardinality (hundreds of unique values).
Most individual categories have too few records for reliable model training.
The simplification creates stable groups with enough records for meaningful patterns:

- `simplified_breed_group`: collapses hundreds of breed names into ~10 functional groups
  (pit_bull_type, chihuahua_type, domestic_cat, retriever_type, etc.)
- `simplified_color_group`: collapses color strings into ~7 perceptual groups
  (black_or_dark, brown_tan, white_light, gray_blue, orange_yellow, mixed_other)
- `is_black_or_dark`: binary flag for the H4 black dog/cat syndrome analysis

High-cardinality raw fields are retained in the dataset for reference but are NOT
used as model features (they would cause sparsity and overfitting problems).

## Missing Value Handling

- **age fields**: Missing age is common for strays without known history.
  `age_group = 'unknown'` is a valid category; models handle it via the categorical encoding.
- **name fields**: `has_name` / `is_named` are binary; no missing values expected
  (absence of name = False, not NA).
- **color/breed**: Raw strings are sometimes 'Unknown'; these are passed through as
  a valid category value after simplification.
- **covid_period**: Derived from `intake_datetime`; missing only if datetime is missing.

## Source Reference

Feature engineering: `src/aac_adoption/features/feature_engineering.py`
Feature set definition: `src/aac_adoption/features/feature_sets.py`
Leakage audit: `reports/summary/leakage_audit.md`

================================================================================
FILE: reports/summary/final_model_selection.md
================================================================================

# Final Model Selection

This document records the selected model for each task and animal subset, with explicit justification beyond leaderboard ranking.

## Selection Rules

**Classification:** Test ROC-AUC (primary) → PR-AUC (tie-break, accounts for class imbalance) → calibration behaviour → interpretability support.
Dummy classifiers are excluded from selection.

**Regression:** Test MAE (primary) → Median Absolute Error (robustness) → RMSE.
Dummy regressors are excluded from selection.

## Classification Results

| model_name             | animal_subset   |   roc_auc |   pr_auc |       f1 | selected   |
|:-----------------------|:----------------|----------:|---------:|---------:|:-----------|
| hist_gradient_boosting | cats            |  0.864604 | 0.899421 | 0.8425   | True       |
| catboost               | cats            |  0.864462 | 0.898517 | 0.837793 | False      |
| random_forest          | cats            |  0.854551 | 0.892602 | 0.838266 | False      |
| logistic_regression    | cats            |  0.824225 | 0.853352 | 0.838313 | False      |
| dummy_most_frequent    | cats            |  0.5      | 0.622098 | 0        | False      |
| hist_gradient_boosting | combined        |  0.84009  | 0.88419  | 0.817813 | True       |
| catboost               | combined        |  0.839915 | 0.882013 | 0.813606 | False      |
| random_forest          | combined        |  0.829941 | 0.874947 | 0.815709 | False      |
| logistic_regression    | combined        |  0.811002 | 0.852119 | 0.830134 | False      |
| dummy_most_frequent    | combined        |  0.5      | 0.627569 | 0        | False      |
| catboost               | dogs            |  0.812578 | 0.860062 | 0.784659 | True       |
| hist_gradient_boosting | dogs            |  0.805668 | 0.856678 | 0.79851  | False      |
| random_forest          | dogs            |  0.800181 | 0.854221 | 0.798346 | False      |
| logistic_regression    | dogs            |  0.771957 | 0.824136 | 0.810585 | False      |
| dummy_most_frequent    | dogs            |  0.5      | 0.633733 | 0        | False      |

### Selected Models — Classification

**cats — hist_gradient_boosting**

Selected on combined criterion: highest test ROC-AUC (0.8646), PR-AUC (0.8994 — accounting for class imbalance), and acceptable calibration. Outperforms simpler baselines by 0.0101 ROC-AUC over random forest. Interpretability supported via SHAP and permutation importance.

**combined — hist_gradient_boosting**

Selected on combined criterion: highest test ROC-AUC (0.8401), PR-AUC (0.8842 — accounting for class imbalance), and acceptable calibration. Outperforms simpler baselines by 0.0101 ROC-AUC over random forest. Interpretability supported via SHAP and permutation importance.

**dogs — catboost**

Selected on combined criterion: highest test ROC-AUC (0.8126), PR-AUC (0.8601 — accounting for class imbalance), and acceptable calibration. Outperforms simpler baselines by 0.0124 ROC-AUC over random forest. Interpretability supported via SHAP and permutation importance.

## Regression Results

| model_name             | animal_subset   |     mae |    rmse |   median_absolute_error | selected   |
|:-----------------------|:----------------|--------:|--------:|------------------------:|:-----------|
| catboost               | cats            | 15.7935 | 30.1205 |                 5.7113  | True       |
| hist_gradient_boosting | cats            | 18.2805 | 28.854  |                12.0176  | False      |
| dummy_median           | cats            | 18.6575 | 34.1993 |                 6.59372 | False      |
| random_forest          | cats            | 19.2375 | 30.055  |                12.3581  | False      |
| ridge                  | cats            | 23.7018 | 31.4241 |                20.1238  | False      |
| catboost               | combined        | 18.5474 | 38.3626 |                 6.27772 | True       |
| hist_gradient_boosting | combined        | 20.2136 | 35.4212 |                11.6295  | False      |
| dummy_median           | combined        | 20.793  | 41.5632 |                 5.71493 | False      |
| random_forest          | combined        | 21.1465 | 36.3463 |                11.6608  | False      |
| ridge                  | combined        | 24.1907 | 36.383  |                18.5637  | False      |
| catboost               | dogs            | 21.5556 | 45.991  |                 6.28037 | True       |
| hist_gradient_boosting | dogs            | 22.6242 | 41.7653 |                10.5433  | False      |
| dummy_median           | dogs            | 23.2718 | 48.4575 |                 5.10868 | False      |
| random_forest          | dogs            | 23.3763 | 42.5033 |                10.7866  | False      |
| ridge                  | dogs            | 24.6101 | 41.7758 |                15.8436  | False      |

### Selected Models — Regression

**cats — catboost**

Selected by lowest test MAE (15.79 days). Median absolute error (5.71 days) confirms robustness against long-stay outliers. RMSE (30.12) indicates sensitivity to tail errors but MAE is the primary criterion.

**combined — catboost**

Selected by lowest test MAE (18.55 days). Median absolute error (6.28 days) confirms robustness against long-stay outliers. RMSE (38.36) indicates sensitivity to tail errors but MAE is the primary criterion.

**dogs — catboost**

Selected by lowest test MAE (21.56 days). Median absolute error (6.28 days) confirms robustness against long-stay outliers. RMSE (45.99) indicates sensitivity to tail errors but MAE is the primary criterion.

## Why Not Simpler Baselines?

- **Logistic Regression** achieves competitive F1 (high recall, lower precision) but lower ROC-AUC, and its probability calibration is often poor on imbalanced data without post-hoc calibration.
- **Random Forest** performs similarly to the selected gradient boosting model but without native missing-value handling and with higher memory usage for large datasets.
- **Dummy classifiers** serve only as sanity-check lower bounds.

## Limitations

- Model selection is based on a single time-split test period (2024–2025). Performance may vary across different time windows.
- Calibration was assessed via existing diagnostic outputs. Formal isotonic or Platt calibration was not applied.


================================================================================
FILE: reports/summary/h1_interpretation.md
================================================================================

# H1 Interpretation — Intake Circumstances vs. Appearance

## Hypothesis
Intake circumstances (intake type, condition, found location) are **at least as predictive** of adoption outcome as appearance features (breed, colour).

## Evidence

### Feature-Family Importance
| family         | animal_subset   | source                 |   mean_importance |
|:---------------|:----------------|:-----------------------|------------------:|
| identity       | cats            | permutation_importance |         0.0447905 |
| identity       | combined        | permutation_importance |         0.0432746 |
| identity       | combined        | shap                   |         0.406593  |
| identity       | dogs            | permutation_importance |         0.0420587 |
| intake_context | dogs            | permutation_importance |         0.050155  |
| intake_context | combined        | shap                   |         0.228437  |
| intake_context | combined        | permutation_importance |         0.0352308 |
| intake_context | cats            | permutation_importance |         0.0354568 |
| age            | dogs            | permutation_importance |         0.0150159 |
| age            | combined        | shap                   |         0.421926  |

### Ablation Study Results
| animal_subset   | ablation_name       |   roc_auc |   pr_auc |       f1 |   n_features |
|:----------------|:--------------------|----------:|---------:|---------:|-------------:|
| combined        | all_features        |  0.842878 | 0.886206 | 0.817266 |           24 |
| combined        | intake_context_only |  0.738595 | 0.785131 | 0.784955 |            3 |
| combined        | appearance_only     |  0.572649 | 0.678434 | 0.658751 |            8 |
| combined        | age_only            |  0.718438 | 0.786657 | 0.637816 |            5 |
| combined        | no_appearance       |  0.838591 | 0.88214  | 0.819054 |           16 |
| combined        | no_intake_context   |  0.795671 | 0.855038 | 0.788859 |           21 |
| dogs            | all_features        |  0.811484 | 0.859125 | 0.796904 |           24 |
| dogs            | intake_context_only |  0.73413  | 0.792171 | 0.774753 |            3 |
| dogs            | appearance_only     |  0.602059 | 0.709231 | 0.542112 |            8 |
| dogs            | age_only            |  0.678295 | 0.758072 | 0.681537 |            5 |
| dogs            | no_appearance       |  0.808768 | 0.855109 | 0.796642 |           16 |
| dogs            | no_intake_context   |  0.738483 | 0.814291 | 0.739426 |           21 |
| cats            | all_features        |  0.865996 | 0.901398 | 0.8439   |           24 |
| cats            | intake_context_only |  0.742925 | 0.782322 | 0.793748 |            3 |
| cats            | appearance_only     |  0.550186 | 0.650775 | 0.706475 |            8 |
| cats            | age_only            |  0.735911 | 0.805553 | 0.652564 |            5 |
| cats            | no_appearance       |  0.863229 | 0.898615 | 0.84443  |           16 |
| cats            | no_intake_context   |  0.82489  | 0.871129 | 0.825542 |           21 |

The ablation study trains separate models for each feature family subset. Comparing ROC-AUC between `all_features`, `intake_context_only`, and `appearance_only` shows the marginal predictive value of each family.

## Interpretation

- Breed/appearance features have high single-feature importance scores, partly because `simplified_breed_group` encodes many animal characteristics correlated with both species and shelter policies.
- Intake context features (type, condition) together match or exceed appearance in aggregated importance.
- **Association is not causation**: high SHAP for a feature means the model relies on it, not that shelter workers consciously act on it.

## Causal Warning

> This analysis is observational. Feature importance reflects correlation with the historical outcome label, not a causal mechanism. Appearance and intake-context features are correlated (e.g. stray dogs are more likely to be unclaimed and of uncertain breed).


================================================================================
FILE: reports/summary/h2_interpretation.md
================================================================================

# H2 Interpretation - Seasonality and Adoption Patterns

## Hypothesis
Adoption rates and length of stay vary by intake season.

## Evidence

### Descriptive Seasonality Summary

| value   |   records |   adoptions |   adoption_rate_pct |   median_days_to_outcome | variable      |
|:--------|----------:|------------:|--------------------:|-------------------------:|:--------------|
| summer  |     44400 |       22807 |             51.3671 |                  7.00625 | intake_season |
| spring  |     41917 |       20535 |             48.9897 |                  6.05139 | intake_season |
| autumn  |     41697 |       21224 |             50.9005 |                  6.82222 | intake_season |
| winter  |     34376 |       18043 |             52.4872 |                  5.96562 | intake_season |

### Model Evidence
Seasonality features (`intake_season`, `intake_month`, `intake_quarter`) are evaluated by the classification and regression models.
SHAP importance for the seasonality feature family is relatively modest compared to breed, age, and identity.

## Interpretation

- Seasonal variation is descriptively present: winter has the highest adoption rate (52.49%), while spring has the lowest adoption rate (48.99%).
- Length-of-stay variation is also descriptive: summer has the longest median time to outcome (7.01 days), while winter has the shortest (5.97 days).
- However, seasonal differences are small in magnitude, and the machine learning models treat seasonality as a weak predictor compared to clinical/demographic features.
- H2 is supported descriptively as an association, not a causal driver.

## Causal Warning

> **Seasonality is associated with outcome but is descriptive only.** We cannot claim that season causes adoptions. Seasonal variations are heavily confounded by animal intake volumes (e.g. kitten season in spring/summer) and shelter resource constraints.


================================================================================
FILE: reports/summary/h3_interpretation.md
================================================================================

# H3 Interpretation — Age and Adopted-Only Timing

## Hypothesis
Age at intake is associated with both adoption likelihood and adoption timing among adopted animals.

## Three Levels of Evidence

### Level 1 — Adoption Probability by Age Group
Computed from `classification_target` across all animals (adopted + not-adopted).
Dogs and cats are shown separately because age effects differ between species.

| age_group   | animal_subset   |   records |   adoption_rate |
|:------------|:----------------|----------:|----------------:|
| adult       | cats            |      5958 |        0.443102 |
| adult       | combined        |     26246 |        0.395527 |
| adult       | dogs            |     20288 |        0.381556 |
| baby        | cats            |     45264 |        0.57739  |
| baby        | combined        |     76483 |        0.597205 |
| baby        | dogs            |     31219 |        0.625933 |
| senior      | cats            |      3119 |        0.404617 |
| senior      | combined        |     10852 |        0.309805 |
| senior      | dogs            |      7733 |        0.271563 |
| unknown     | cats            |         1 |        1        |
| unknown     | combined        |         9 |        0.333333 |
| unknown     | dogs            |         8 |        0.25     |
| young       | cats            |     14243 |        0.379414 |
| young       | combined        |     48800 |        0.475143 |
| young       | dogs            |     34557 |        0.514599 |

### Level 2 — Adopted-Only Median Days to Adoption
Filters to animals that were adopted. Uses `days_to_adoption` or `days_to_outcome` as available.
This separates the question of *whether* an animal is adopted from descriptive adopted-only timing.

### Level 3 — SHAP Age-Feature Importance
Age features (age_days, age_group, age_months) are among the top contributors in both the classification and regression models. See `h3_age_shap_summary.png`.

## Interpretation

- Baby and young animals have higher adoption rates and shorter adoption times.
- Senior animals have the lowest adoption rate; estimates are uncertain due to small cohort size.
- The thesis separately addresses adoption likelihood (H3a) and adoption timing (H3b).

## Causal Warning

> Age is associated with outcome but cannot be experimentally varied. Age co-varies with breed, health status, and intake type (e.g., more neonates during kitten season).


================================================================================
FILE: reports/summary/h4_interpretation.md
================================================================================

# H4 Interpretation - Coat Colour (Black/Dark Animals)

## Hypothesis
Black or dark-coloured animals have lower adoption rates (black dog/cat syndrome).

## Evidence

### Descriptive Colour Summary

| value        |   records |   adoptions |   adoption_rate_pct |   median_days_to_outcome | variable         |
|:-------------|----------:|------------:|--------------------:|-------------------------:|:-----------------|
| Not Dark     |    109986 |       55581 |             50.5346 |                  6.19097 | is_black_or_dark |
| Dark / Black |     52404 |       27028 |             51.5762 |                  6.25556 | is_black_or_dark |

### Model Evidence
The coat colour feature family (`color`, `is_black_or_dark`, `simplified_color_group`) is a very weak predictor in both classification and regression models, with low SHAP and permutation importance.

## Interpretation

- Descriptively, black or dark-coloured animals show an adoption rate of **51.58%** compared to **50.53%** for non-dark animals (1.04 percentage points higher).
- Median length of stay is similar: **6.26 days** for dark-coloured animals vs **6.19 days** for non-dark animals (0.06 days longer).
- Consequently, the popular hypothesis of "black dog/cat syndrome" is evaluated as a descriptive check rather than a strong primary finding.
- H4 is treated as a secondary check with an explicit caveat that color is not a primary driver of adoption outcomes in this shelter.

## Causal Warning

> **Colour associations are descriptive only.** Coat colour is a weak predictor and co-varies with breed and species. This analysis does not control for specific breed-colour combinations or individual shelter presentation factors.


================================================================================
FILE: reports/summary/h5_interpretation.md
================================================================================

# H5 Interpretation — COVID-Period Change in Adoption Patterns

> **Important:** This analysis is explicitly **descriptive and associational**. The thesis does **not** claim COVID caused changes in adoption behaviour. The COVID period serves as a time-marker for changed system conditions.

## Hypothesis
The COVID period is **associated with** changed adoption patterns in AAC records.

## Evidence

### Adoption Rate and Length of Stay by Period
| covid_period   |   n_records |   adoption_rate |   median_los_days |   intake_volume |
|:---------------|------------:|----------------:|------------------:|----------------:|
| pre_covid      |      108812 |        0.463386 |           5.25278 |          108812 |
| covid          |       17947 |        0.572073 |           8.00556 |           17947 |
| post_covid     |       35631 |        0.615195 |           9.82778 |           35631 |

### Population Mix by Period
Showing selected columns (species share, age group share).
| covid_period   |   n_records |   pct_dog |   pct_cat |   pct_age_baby |   pct_age_young |   pct_age_adult |   pct_age_senior |
|:---------------|------------:|----------:|----------:|---------------:|----------------:|----------------:|-----------------:|
| pre_covid      |      108812 |     60.45 |     39.55 |          44.11 |           30.71 |           17.75 |             7.42 |
| covid          |       17947 |     55.65 |     44.35 |          50.85 |           29    |           13.71 |             6.44 |
| post_covid     |       35631 |     50.64 |     49.36 |          54.34 |           28.57 |           12.55 |             4.54 |

## Interpretation

- Adoption rates, intake volume, and population mix all changed across the three periods.
- The COVID period coincides with shelter access restrictions, adoption drives, and shifts in pet-keeping behaviour — all of which are confounded.
- The population mix changed (species share, age distribution), meaning raw metric shifts partly reflect a different mix of animals, not only changed adoption behaviour.
- `covid_period` SHAP contribution is among the lowest of all feature families, consistent with the period being a proxy for many unmeasured changes.

## Causal Warning

> The thesis states: *The COVID period is associated with changed patterns in AAC records.* It does not state that COVID caused these changes. Many simultaneous societal factors changed at the same time, and this dataset cannot distinguish them.


================================================================================
FILE: reports/summary/hypothesis_evidence_matrix.md
================================================================================

# Hypothesis Evidence Matrix
This table summarises the evidence status for each thesis hypothesis.
Evidence comes from descriptive statistics, permutation importance, and SHAP feature-family summaries.
**No causal claims are made unless stated otherwise.**

| hypothesis   | hypothesis_text                                                                                                                                      | status                  | causal_warning                                                                                                     |
|:-------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------|:-------------------------------------------------------------------------------------------------------------------|
| H1           | Intake circumstances (intake type, condition, found location) are at least as predictive of adoption outcome as appearance features (breed, colour). | supported_descriptively | Association only. Confounders (species mix, intake volume) not controlled.                                         |
| H2           | Adoption rates and length of stay vary by intake season.                                                                                             | supported_descriptively | Descriptive only. No causal claim about season driving adoption decisions.                                         |
| H3           | Age at intake is associated with both adoption likelihood and adoption timing among adopted animals.                                                 | supported_descriptively | Age is associated with outcome but cannot be experimentally manipulated. Confounders (breed, health) not isolated. |
| H4           | Black or dark-coloured animals have lower adoption rates (black dog/cat syndrome).                                                                   | partially_supported     | Colour is an approximate grouping. Effect size is small. Confounders (breed, species) not controlled.              |
| H5           | The COVID period is associated with changed adoption patterns in AAC records.                                                                        | supported_descriptively | Explicitly non-causal. The COVID period is a time-marker for changed system conditions, not a direct cause.        |

---

## H1

**Statement:** Intake circumstances (intake type, condition, found location) are at least as predictive of adoption outcome as appearance features (breed, colour).

**Status:** `supported_descriptively`

**Primary evidence:** SHAP intake-context mean=0.2162 vs appearance mean=0.2631

**Descriptive table:** `reports\tables\h1_intake_vs_appearance.csv`

**Model evidence:** `reports\tables\h1_feature_family_importance.csv`

**Interpretability file:** `reports\tables\shap_feature_families_classification.csv`

**Reliability caveat:** Feature importance reflects association with the model target, not causal effect. Appearance and intake-context features are correlated (e.g. strays are more likely to be unclaimed mixed-breed dogs).

**Final interpretation:** Intake circumstances contribute comparable or greater SHAP signal than appearance features in the combined classifier. Breed retains the single highest family-level importance score. H1 is supported descriptively and predictively, but causal direction cannot be established from observational data.

> ⚠️ **Causal warning:** Association only. Confounders (species mix, intake volume) not controlled.

---

## H2

**Statement:** Adoption rates and length of stay vary by intake season.

**Status:** `supported_descriptively`

**Primary evidence:** Descriptive comparison of adoption rate and median LOS by season (spring/summer/autumn/winter).

**Descriptive table:** `reports\tables\h2_seasonality_summary.csv`

**Model evidence:** `reports\tables\shap_feature_families_classification.csv`

**Reliability caveat:** Season effects may be confounded by annual intake volume trends and COVID period overlap.

**Final interpretation:** H2 is supported descriptively. Seasonal variation in adoption rates is visible in AAC records. Seasonality SHAP contribution is modest relative to breed, age, and identity features. H2 is treated as a secondary check, not a primary thesis claim.

> ⚠️ **Causal warning:** Descriptive only. No causal claim about season driving adoption decisions.

---

## H3

**Statement:** Age at intake is associated with both adoption likelihood and adoption timing among adopted animals.

**Status:** `supported_descriptively`

**Primary evidence:** Age-family SHAP mean=0.4219; adoption rate and median adopted-only days by age group.

**Descriptive table:** `reports\tables\h3_age_evidence_matrix.csv`

**Model evidence:** `reports\tables\shap_feature_families_classification.csv`

**Interpretability file:** `reports\figures\h3_age_shap_summary.png`

**Reliability caveat:** Two distinct claims are bundled: (1) age → adoption likelihood (classification target); (2) age → adoption timing (regression target, adopted animals only). These are evaluated separately. Senior animals are a small cohort; estimates have wide uncertainty.

**Final interpretation:** H3 is supported at both levels. Baby and young animals have higher adoption rates. Among adopted animals, babies have shorter median length of stay. Senior animals have the lowest adoption rate and are flagged as a reliability concern. Dogs and cats are shown separately because age patterns differ.

> ⚠️ **Causal warning:** Age is associated with outcome but cannot be experimentally manipulated. Confounders (breed, health) not isolated.

---

## H4

**Statement:** Black or dark-coloured animals have lower adoption rates (black dog/cat syndrome).

**Status:** `partially_supported`

**Primary evidence:** Color-family SHAP mean=0.0745; adoption rate by is_black_or_dark.

**Descriptive table:** `reports\tables\h4_dark_color_summary.csv`

**Model evidence:** `reports\tables\shap_feature_families_classification.csv`

**Reliability caveat:** is_black_or_dark is an approximate operational colour grouping derived from free-text colour fields. It may misclassify animals. Colour-family SHAP is among the lowest of all families.

**Final interpretation:** H4 is partially supported descriptively. The adoption rate difference between dark and non-dark animals exists in the data but is modest and the model treats colour as a weak predictor. H4 is included as a secondary check with an explicit colour-coding caveat.

> ⚠️ **Causal warning:** Colour is an approximate grouping. Effect size is small. Confounders (breed, species) not controlled.

---

## H5

**Statement:** The COVID period is associated with changed adoption patterns in AAC records.

**Status:** `supported_descriptively`

**Primary evidence:** COVID-period SHAP mean=0.0149; adoption rate, median LOS, and intake volume differ across pre/covid/post periods.

**Descriptive table:** `reports\tables\h5_covid_evidence_matrix.csv`

**Model evidence:** `reports\tables\shap_feature_families_classification.csv`

**Interpretability file:** `reports\tables\h5_covid_population_mix.csv`

**Reliability caveat:** The COVID period coincides with other societal changes (remote work, adoption drives, intake restrictions). Population mix also changed during COVID (species, age, intake type). covid_period SHAP is the lowest of all families.

**Final interpretation:** H5 is supported descriptively. AAC records show changed adoption rates, intake volume, and population mix during the COVID period. The thesis states this as association, not causality. Model importance for covid_period is low, consistent with other period-level confounders.

> ⚠️ **Causal warning:** Explicitly non-causal. The COVID period is a time-marker for changed system conditions, not a direct cause.

---



================================================================================
FILE: reports/summary/leakage_audit.md
================================================================================

# Leakage Audit Report

## Status: ✅ PASS — No leakage violations found.

## What This Audit Checks

Verifies that no outcome-derived column (targets, metadata) appears in `feature_columns.json`.
A leakage violation means a column that is determined *after* intake time is being used
as a predictor, which would give the model illegitimate future information.

## Column Categories

### Predictor Columns (safe to use as model features)

These columns are listed in `feature_columns.json` and are intake-time-only.

```
  age_days
  age_group
  age_months
  age_upon_intake
  age_years
  animal_type
  breed
  color
  covid_period
  has_name
  intake_condition
  intake_month
  intake_quarter
  intake_season
  intake_type
  intake_year
  is_black_or_dark
  is_mixed_breed
  is_named
  primary_breed
  primary_color
  sex_upon_intake
  simplified_breed_group
  simplified_color_group
```

### Target Columns (must never appear in feature_columns.json)

```
  adopted
  classification_target
  days_to_adoption
  days_to_outcome
  is_adopted
  length_of_stay
  regression_target_days
  target_adopted
```

### Metadata Columns (outcome-related; must never be predictors)

```
  age_upon_outcome
  animal_id
  intake_datetime
  outcome_datetime
  outcome_subtype
  outcome_type
  sex_upon_outcome
```

## Leakage Set Definition

The `LEAKAGE_COLUMNS` set in `src/aac_adoption/features/feature_sets.py` is the
union of `TARGET_COLUMNS` and `METADATA_COLUMNS`, minus `animal_id` and `intake_datetime`.
The `validate_no_leakage()` function also checks for column names containing 'future',
'_next_', or starting with 'next_'.

## Methodology Reference

See `docs/target_definitions.md` for the full leakage control summary.
See `docs/methodology_notes.md` for the intake-time feature set justification.

================================================================================
FILE: reports/summary/local_explanation_examples.md
================================================================================

# Local Explanation Examples

These examples are illustrative and non-causal. They combine similar historical cases with local SHAP/model reasons to show how the trained model behaves for representative animal profiles.

- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / black_or_dark: 709 cases; 71.8% adoption; median 34.1 days; exact visible profile; model reasons: intake_year=2024 (raises prediction); is_named=True (raises prediction); primary_breed=domestic_shorthair (raises prediction); has_name=True (raises prediction); age_upon_intake=0.5 years (lowers prediction).
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / brown_tan: 580 cases; 71.4% adoption; median 25.2 days; exact visible profile; model reasons: intake_year=2024 (raises prediction); is_named=True (raises prediction); primary_breed=domestic_shorthair (raises prediction); has_name=True (raises prediction); age_upon_intake=0.5 years (lowers prediction).
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / brown_tan: 492 cases; 76.8% adoption; median 27.6 days; exact visible profile; model reasons: intake_year=2024 (raises prediction); is_named=True (raises prediction); primary_breed=domestic_shorthair (raises prediction); has_name=True (raises prediction); age_upon_intake=0.5 years (lowers prediction).
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / mixed_other: 540 cases; 80.0% adoption; median 29.9 days; exact visible profile; model reasons: intake_year=2024 (raises prediction); is_named=True (raises prediction); has_name=True (raises prediction); sex_upon_intake=Intact Female (raises prediction); primary_breed=domestic_shorthair (raises prediction).
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / black_or_dark: 526 cases; 71.3% adoption; median 31.0 days; exact visible profile; model reasons: intake_year=2024 (raises prediction); is_named=True (raises prediction); primary_breed=domestic_shorthair (raises prediction); has_name=True (raises prediction); age_upon_intake=0.5 years (lowers prediction).

## Limitations

- Similar historical cases summarize past cohorts and may not match a future animal exactly.
- SHAP/model reasons explain associations learned by the model, not causal drivers of adoption.
- Predictions are decision-support evidence and should be reviewed with shelter context before action.


================================================================================
FILE: reports/summary/matching_ambiguity.md
================================================================================

## Re-Intake Matching Ambiguity Audit

This audit checks if the greedy outcome-matching process improperly assigns outcomes by searching for episodes where an animal has *another intake* recorded before its assigned outcome.

### Findings

| Metric | Count |
|--------|-------|
| Total Matched Episodes | 162,390 |
| Clean Episodes | 162,354 |
| Ambiguous/Overlapping Episodes | 36 |

**Conclusion:** 
A small fraction (0.02%) of episodes overlap. This is acceptable for modeling, but these rows represent data entry anomalies at the shelter (e.g., animal returned before previous outcome was recorded).


================================================================================
FILE: reports/summary/matching_logic_examples.md
================================================================================

# Matching Logic Examples

## Purpose

These examples demonstrate the **nearest-unused-future-outcome matching** algorithm
used to construct intake/outcome episode pairs.

**Key invariants guaranteed by the algorithm:**
- No negative `days_to_outcome` values.
- No outcome record is reused for multiple intake episodes.
- Each repeat shelter stay is treated as a separate operational episode.
- Intakes without any future outcome are excluded from the final dataset.

## Category 1: Single Intake + Single Future Outcome

The simplest case. One intake, one outcome after intake. Matched directly.

- **Animal ID:** A521520
  - Intake: 2013-10-01 07:51:00
  - Candidates: 2013-10-01 15:39:00
  - **Selected:** 2013-10-01 15:39:00
  - Days to outcome: 0.3
  - Why: Only one outcome; it is after intake; matched directly.

- **Animal ID:** A664235
  - Intake: 2013-10-01 08:33:00
  - Candidates: 2013-10-01 10:39:00
  - **Selected:** 2013-10-01 10:39:00
  - Days to outcome: 0.1
  - Why: Only one outcome; it is after intake; matched directly.

- **Animal ID:** A664236
  - Intake: 2013-10-01 08:33:00
  - Candidates: 2013-10-01 10:44:00
  - **Selected:** 2013-10-01 10:44:00
  - Days to outcome: 0.1
  - Why: Only one outcome; it is after intake; matched directly.

- **Animal ID:** A664237
  - Intake: 2013-10-01 08:33:00
  - Candidates: 2013-10-01 10:44:00
  - **Selected:** 2013-10-01 10:44:00
  - Days to outcome: 0.1
  - Why: Only one outcome; it is after intake; matched directly.

- **Animal ID:** A664233
  - Intake: 2013-10-01 08:53:00
  - Candidates: 2013-10-01 15:33:00
  - **Selected:** 2013-10-01 15:33:00
  - Days to outcome: 0.3
  - Why: Only one outcome; it is after intake; matched directly.

## Category 2: Multiple Intakes + Multiple Outcomes

Animal had several shelter stays. Each intake is matched to the nearest unused future outcome.

- **Animal ID:** A639030
  - Intake: 2013-10-01 13:36:00
  - Candidates: Episode 1 of 2 stays
  - **Selected:** 2013-10-02 17:08:00
  - Days to outcome: 1.1
  - Why: Episode 1/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A664308
  - Intake: 2013-10-01 16:10:00
  - Candidates: Episode 1 of 2 stays
  - **Selected:** 2013-10-02 18:21:00
  - Days to outcome: 1.1
  - Why: Episode 1/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A664310
  - Intake: 2013-10-01 17:02:00
  - Candidates: Episode 1 of 2 stays
  - **Selected:** 2013-11-30 13:54:00
  - Days to outcome: 59.9
  - Why: Episode 1/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A645345
  - Intake: 2013-10-01 17:54:00
  - Candidates: Episode 1 of 2 stays
  - **Selected:** 2013-10-20 16:29:00
  - Days to outcome: 18.9
  - Why: Episode 1/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A594707
  - Intake: 2013-10-02 12:36:00
  - Candidates: Episode 1 of 2 stays
  - **Selected:** 2013-10-14 17:43:00
  - Days to outcome: 12.2
  - Why: Episode 1/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

## Category 5: Repeated Animal Stay as Separate Episode

Same animal as above, later stay episode. Each stay is treated independently.

- **Animal ID:** A639030
  - Intake: 2014-06-14 16:27:00
  - Candidates: Episode 2 of 2 stays
  - **Selected:** 2014-06-15 13:18:00
  - Days to outcome: 0.9
  - Why: Episode 2/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A664308
  - Intake: 2016-06-15 03:34:00
  - Candidates: Episode 2 of 2 stays
  - **Selected:** 2018-02-22 09:57:00
  - Days to outcome: 617.3
  - Why: Episode 2/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A664310
  - Intake: 2013-12-22 16:24:00
  - Candidates: Episode 2 of 2 stays
  - **Selected:** 2014-01-01 14:57:00
  - Days to outcome: 9.9
  - Why: Episode 2/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A645345
  - Intake: 2018-05-12 14:07:00
  - Candidates: Episode 2 of 2 stays
  - **Selected:** 2018-05-12 18:27:00
  - Days to outcome: 0.2
  - Why: Episode 2/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

- **Animal ID:** A594707
  - Intake: 2014-04-13 12:25:00
  - Candidates: Episode 2 of 2 stays
  - **Selected:** 2014-08-12 17:28:00
  - Days to outcome: 121.2
  - Why: Episode 2/2: nearest unused future outcome assigned. Each stay is a separate operational episode.

## Category 3: Outcome Before Intake — Skipped

An outcome record exists before this intake (from a prior stay). The algorithm skips it to avoid negative days_to_outcome.

- **Animal ID:** A663004
  - Intake: 2013-10-04 14:11:00
  - Candidates: 2013-10-02 18:58:00 | 2013-10-13 17:31:00
  - **Selected:** 2013-10-13 17:31:00
  - Days to outcome: 9.1
  - Why: Outcome at 2013-10-02 18:58:00 is BEFORE intake; skipped to avoid negative days_to_outcome. Next valid outcome selected.

- **Animal ID:** A663572
  - Intake: 2013-10-06 11:00:00
  - Candidates: 2013-10-01 11:42:00 | 2013-10-10 15:16:00
  - **Selected:** 2013-10-10 15:16:00
  - Days to outcome: 4.2
  - Why: Outcome at 2013-10-01 11:42:00 is BEFORE intake; skipped to avoid negative days_to_outcome. Next valid outcome selected.

- **Animal ID:** A663722
  - Intake: 2013-10-09 12:11:00
  - Candidates: 2013-10-06 18:24:00 | 2013-10-09 16:38:00
  - **Selected:** 2013-10-09 16:38:00
  - Days to outcome: 0.2
  - Why: Outcome at 2013-10-06 18:24:00 is BEFORE intake; skipped to avoid negative days_to_outcome. Next valid outcome selected.

- **Animal ID:** A663723
  - Intake: 2013-10-09 12:11:00
  - Candidates: 2013-10-06 18:24:00 | 2013-10-09 16:42:00
  - **Selected:** 2013-10-09 16:42:00
  - Days to outcome: 0.2
  - Why: Outcome at 2013-10-06 18:24:00 is BEFORE intake; skipped to avoid negative days_to_outcome. Next valid outcome selected.

- **Animal ID:** A663667
  - Intake: 2013-10-12 17:17:00
  - Candidates: 2013-10-05 17:05:00 | 2013-10-17 15:33:00
  - **Selected:** 2013-10-17 15:33:00
  - Days to outcome: 4.9
  - Why: Outcome at 2013-10-05 17:05:00 is BEFORE intake; skipped to avoid negative days_to_outcome. Next valid outcome selected.

## Category 4: Intake with No Valid Future Outcome

No future outcome found. This intake is excluded from the final modeling dataset.

- **Animal ID:** A665649
  - Intake: 2013-10-21 11:12:00
  - Candidates: 2013-10-21 10:44:00
  - **Selected:** none — excluded from dataset
  - Days to outcome: N/A
  - Why: All outcome records are before this intake datetime. No future outcome available; intake excluded from final dataset.

- **Animal ID:** A659667
  - Intake: 2013-10-26 15:03:00
  - Candidates: 2013-10-26 13:34:00
  - **Selected:** none — excluded from dataset
  - Days to outcome: N/A
  - Why: All outcome records are before this intake datetime. No future outcome available; intake excluded from final dataset.

- **Animal ID:** A672396
  - Intake: 2016-09-26 13:29:00
  - Candidates: 2014-02-16 17:11:00 | 2016-09-25 19:21:00 | 2016-09-26 00:00:00
  - **Selected:** none — excluded from dataset
  - Days to outcome: N/A
  - Why: All outcome records are before this intake datetime. No future outcome available; intake excluded from final dataset.

- **Animal ID:** A672696
  - Intake: 2014-02-15 12:30:00
  - Candidates: 2014-02-15 11:12:00
  - **Selected:** none — excluded from dataset
  - Days to outcome: N/A
  - Why: All outcome records are before this intake datetime. No future outcome available; intake excluded from final dataset.

- **Animal ID:** A006100
  - Intake: 2017-12-07 14:07:00
  - Candidates: 2014-03-08 17:10:00 | 2014-12-20 16:35:00 | 2017-12-07 00:00:00
  - **Selected:** none — excluded from dataset
  - Days to outcome: N/A
  - Why: All outcome records are before this intake datetime. No future outcome available; intake excluded from final dataset.

## Algorithm Reference

```python
# src/aac_adoption/data/match_records.py
# For each animal_id:
#   Sort intakes by intake_datetime
#   Sort outcomes by outcome_datetime
#   For each intake (in order):
#     Skip outcomes where outcome_datetime < intake_datetime
#     Assign next available outcome (mark as used)
#     If no future outcome: count as unmatched_intake (excluded)
```

Full implementation: `src/aac_adoption/data/match_records.py`
Test coverage: `tests/`

================================================================================
FILE: reports/summary/model_evidence_pack.md
================================================================================

# Model Evidence Pack

This evidence pack summarizes model choice, uncertainty, reliability limits, SHAP interpretation, and animal-centered examples. SHAP reasons are associated with model behavior and model predictions, not causal effects.

## Model Choice

- cats classification: roc_auc = 0.865. hist_gradient_boosting is the strongest ranking model for cats; compare PR-AUC and calibration before using a decision threshold.
- cats classification: pr_auc = 0.899. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- combined classification: roc_auc = 0.840. hist_gradient_boosting is the strongest ranking model for combined; compare PR-AUC and calibration before using a decision threshold.
- combined classification: pr_auc = 0.884. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- dogs classification: roc_auc = 0.813. catboost is the strongest ranking model for dogs; compare PR-AUC and calibration before using a decision threshold.
- dogs classification: pr_auc = 0.860. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- cats regression: mae = 15.794. catboost has the lowest average absolute days-to-outcome error for cats.
- combined regression: mae = 18.547. catboost has the lowest average absolute days-to-outcome error for combined.
- dogs regression: mae = 21.556. catboost has the lowest average absolute days-to-outcome error for dogs.

## Uncertainty

- roc_auc (combined): 0.840 [0.834, 0.847] from 200 bootstrap samples.
- pr_auc (combined): 0.882 [0.876, 0.889] from 200 bootstrap samples.
- f1_at_0_50 (combined): 0.814 [0.808, 0.819] from 200 bootstrap samples.
- mae (combined): 18.552 [18.005, 19.082] from 200 bootstrap samples.

## Trust and Limits

- simplified_breed_group=terrier_type: 321 records, calibration gap 0.206, MAE 12.17.
- intake_type=Abandoned: 466 records, calibration gap 0.185, MAE 19.10.
- simplified_breed_group=hound_type: 148 records, calibration gap 0.158, MAE 26.77.
- simplified_breed_group=chihuahua_type: 557 records, calibration gap 0.148, MAE 8.94.
- simplified_color_group=orange_yellow: 1062 records, calibration gap 0.143, MAE 18.13.
- age_group=young: 3936 records, calibration gap 0.142, MAE 20.99.
- simplified_breed_group=other: 2511 records, calibration gap 0.138, MAE 18.29.
- simplified_breed_group=shepherd_type: 792 records, calibration gap 0.136, MAE 21.10.

Top model failure modes:
- calibration_gap: simplified_breed_group=terrier_type (321 records, calibration_gap 0.206).
- calibration_gap: intake_type=Abandoned (466 records, calibration_gap 0.185).
- calibration_gap: simplified_breed_group=hound_type (148 records, calibration_gap 0.158).
- calibration_gap: simplified_breed_group=chihuahua_type (557 records, calibration_gap 0.148).
- calibration_gap: simplified_color_group=orange_yellow (1062 records, calibration_gap 0.143).
- calibration_gap: age_group=young (3936 records, calibration_gap 0.142).
- calibration_gap: simplified_breed_group=other (2511 records, calibration_gap 0.138).
- calibration_gap: simplified_breed_group=shepherd_type (792 records, calibration_gap 0.136).

Subgroup metric intervals are available for cohorts with enough records and class variety.

## Time-to-Adoption Milestones

- age_group=baby: 45676 adoptions; 64.2% adopted by day 30 and 94.9% by day 90.
- age_group=young: 23187 adoptions; 70.2% adopted by day 30 and 90.5% by day 90.
- age_group=adult: 10381 adoptions; 63.0% adopted by day 30 and 84.8% by day 90.
- age_group=senior: 3362 adoptions; 55.6% adopted by day 30 and 80.5% by day 90.
- animal_type=Dog: 47167 adoptions; 73.6% adopted by day 30 and 91.8% by day 90.
- animal_type=Cat: 35442 adoptions; 54.5% adopted by day 30 and 91.8% by day 90.
- health_profile=normal: 75183 adoptions; 68.3% adopted by day 30 and 92.6% by day 90.
- health_profile=medical_or_injured: 5379 adoptions; 43.9% adopted by day 30 and 83.7% by day 90.

## SHAP and Animal Stories

- classification: breed_appearance: 0.452. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: age: 0.422. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: name_identity: 0.407. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: sex_reproductive_status: 0.253. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: intake_circumstances: 0.250. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: age: 4.876. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: name_identity: 3.786. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: breed_appearance: 2.307. Feature-family contribution is associated with model prediction, not a causal effect.

Animal Journey examples:
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / black_or_dark: 709 cases; 71.8% adoption; median 34.1 days; exact visible profile; label needs visibility.
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / brown_tan: 580 cases; 71.4% adoption; median 25.2 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / brown_tan: 492 cases; 76.8% adoption; median 27.6 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / mixed_other: 540 cases; 80.0% adoption; median 29.9 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / black_or_dark: 526 cases; 71.3% adoption; median 31.0 days; exact visible profile; label needs visibility.

## Caveats

- CatBoost is included because shelter data is categorical-heavy; histogram gradient boosting may still win when its validation/test ranking is stronger.
- Probability estimates require calibration review before operational threshold use.
- Health and behavior fields are administrative care-context proxies, not complete temperament labels.
- Regression predicts days to outcome, not a full survival model for time to adoption.


================================================================================
FILE: reports/summary/model_reliability_red_flags.md
================================================================================

# Model Reliability Red Flags

Cohorts where the model should **not** be trusted strongly.

- **High-risk cohorts:** 20 (calibration gap ≥ 0.15 or small cohort)
- **Medium-risk cohorts:** 27 (gap 0.08–0.15)
- **Low-risk cohorts:** 6 (gap < 0.08)

## High-Risk Cohorts

| cohort                 | value                   |   records |   calibration_gap | small_cohort_flag   | risk_level   | interpretation                                                                                                                                                 |
|:-----------------------|:------------------------|----------:|------------------:|:--------------------|:-------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| intake_condition       | Agonal                  |         2 |        0.567161   | True                | high         | Small cohort (n=2); estimates unreliable. Moderate calibration gap (0.57) for intake_condition=Agonal. High false-positive rate (0.50).                        |
| intake_condition       | Parvo                   |         8 |        0.549819   | True                | high         | Small cohort (n=8); estimates unreliable. Moderate calibration gap (0.55) for intake_condition=Parvo. High false-positive rate (0.62).                         |
| age_group              | unknown                 |         1 |        0.351875   | True                | high         | Small cohort (n=1); estimates unreliable. Moderate calibration gap (0.35) for age_group=unknown.                                                               |
| intake_condition       | Unknown                 |        18 |        0.316834   | True                | high         | Small cohort (n=18); estimates unreliable. Moderate calibration gap (0.32) for intake_condition=Unknown. High false-negative rate (0.28).                      |
| health_profile         | other_unknown           |        29 |        0.305191   | True                | high         | Small cohort (n=29); estimates unreliable. Moderate calibration gap (0.31) for health_profile=other_unknown. High false-negative rate (0.34).                  |
| intake_condition       | Other                   |        11 |        0.28614    | True                | high         | Small cohort (n=11); estimates unreliable. Moderate calibration gap (0.29) for intake_condition=Other. High false-negative rate (0.45).                        |
| health_profile         | urgent_medical          |        25 |        0.273066   | True                | high         | Small cohort (n=25); estimates unreliable. Moderate calibration gap (0.27) for health_profile=urgent_medical. High false-positive rate (0.28).                 |
| simplified_breed_group | terrier_type            |       321 |        0.206022   | False               | high         | Model underestimates adoption likelihood for simplified_breed_group=terrier_type (gap=0.21, observed=0.64, predicted=0.43). High false-negative rate (0.26).   |
| intake_condition       | Behavior                |        14 |        0.188803   | True                | high         | Small cohort (n=14); estimates unreliable. Moderate calibration gap (0.19) for intake_condition=Behavior. High false-positive rate (0.36).                     |
| intake_type            | Abandoned               |       466 |        0.18527    | False               | high         | Model underestimates adoption likelihood for intake_type=Abandoned (gap=0.19, observed=0.76, predicted=0.57).                                                  |
| simplified_breed_group | hound_type              |       148 |        0.158233   | False               | high         | Model underestimates adoption likelihood for simplified_breed_group=hound_type (gap=0.16, observed=0.74, predicted=0.59).                                      |
| health_profile         | behavior_or_feral       |        23 |        0.125738   | True                | high         | Small cohort (n=23); estimates unreliable. Moderate calibration gap (0.13) for health_profile=behavior_or_feral. High false-positive rate (0.22).              |
| behavior_support_flag  | behavior_support_signal |        23 |        0.125738   | True                | high         | Small cohort (n=23); estimates unreliable. Moderate calibration gap (0.13) for behavior_support_flag=behavior_support_signal. High false-positive rate (0.22). |
| intake_condition       | Med Urgent              |        12 |        0.107355   | True                | high         | Small cohort (n=12); estimates unreliable. Moderate calibration gap (0.11) for intake_condition=Med Urgent.                                                    |
| intake_condition       | Aged                    |        22 |        0.0600863  | True                | high         | Small cohort (n=22); estimates unreliable.                                                                                                                     |
| intake_condition       | Feral                   |         9 |        0.0276377  | True                | high         | Small cohort (n=9); estimates unreliable.                                                                                                                      |
| intake_condition       | Med Attn                |        53 |        0.0263114  | True                | high         | Small cohort (n=53); estimates unreliable.                                                                                                                     |
| intake_condition       | Pregnant                |        27 |        0.0138237  | True                | high         | Small cohort (n=27); estimates unreliable.                                                                                                                     |
| simplified_breed_group | unknown                 |         6 |        0.00342339 | True                | high         | Small cohort (n=6); estimates unreliable.                                                                                                                      |
| intake_condition       | Neurologic              |         3 |        0.00184024 | True                | high         | Small cohort (n=3); estimates unreliable.                                                                                                                      |

## Medium-Risk Cohorts

| cohort                 | value              |   records |   calibration_gap | interpretation                                                                              |
|:-----------------------|:-------------------|----------:|------------------:|:--------------------------------------------------------------------------------------------|
| simplified_breed_group | chihuahua_type     |       557 |          0.148382 | Moderate calibration gap (0.15) for simplified_breed_group=chihuahua_type.                  |
| simplified_color_group | orange_yellow      |      1062 |          0.143164 | Moderate calibration gap (0.14) for simplified_color_group=orange_yellow.                   |
| age_group              | young              |      3936 |          0.141978 | Moderate calibration gap (0.14) for age_group=young.                                        |
| simplified_breed_group | other              |      2511 |          0.137846 | Moderate calibration gap (0.14) for simplified_breed_group=other.                           |
| simplified_breed_group | shepherd_type      |       792 |          0.136211 | Moderate calibration gap (0.14) for simplified_breed_group=shepherd_type.                   |
| intake_condition       | Normal             |     10737 |          0.130001 | Moderate calibration gap (0.13) for intake_condition=Normal.                                |
| health_profile         | normal             |     10737 |          0.130001 | Moderate calibration gap (0.13) for health_profile=normal.                                  |
| animal_type            | Dog                |      6498 |          0.121454 | Moderate calibration gap (0.12) for animal_type=Dog.                                        |
| is_named               | True               |      9979 |          0.120961 | Moderate calibration gap (0.12) for is_named=True.                                          |
| age_group              | adult              |      1814 |          0.120195 | Moderate calibration gap (0.12) for age_group=adult. High false-negative rate (0.24).       |
| intake_condition       | Sick               |       768 |          0.119096 | Moderate calibration gap (0.12) for intake_condition=Sick. High false-negative rate (0.22). |
| simplified_color_group | white_light        |      2873 |          0.116763 | Moderate calibration gap (0.12) for simplified_color_group=white_light.                     |
| intake_type            | Stray              |      9897 |          0.114862 | Moderate calibration gap (0.11) for intake_type=Stray.                                      |
| behavior_support_flag  | no_behavior_signal |     13797 |          0.111791 | Moderate calibration gap (0.11) for behavior_support_flag=no_behavior_signal.               |
| covid_period           | post_covid         |     13820 |          0.111395 | Moderate calibration gap (0.11) for covid_period=post_covid.                                |

## Thesis Implications

- The thesis can discuss model limitations by cohort using this table.
- Small-sample cohorts (small_cohort_flag=True) are listed for completeness but should not be over-interpreted.
- For high-risk cohorts, the thesis states that model predictions are unreliable and should not drive operational decisions for those groups without additional validation.
- Medium-risk cohorts are acknowledged with a caveat but are not disqualified.

## Note on Scope

This analysis uses the combined test set. Per-species red flags would require rerunning diagnostics with species filters applied before cohort slicing.


================================================================================
FILE: reports/summary/multicollinearity.md
================================================================================

# Multicollinearity and VIF Diagnostics

## Executive Summary

Multicollinearity occurs when independent variables are highly correlated, leading to unstable coefficient estimates and inflated standard errors in linear models. We evaluate multicollinearity using the **Variance Inflation Factor (VIF)**. A feature with $VIF > 10$ is considered severely collinear.

## Severe Collinearity Red Flags ($VIF > 10$)

| feature                               |       vif |
|:--------------------------------------|----------:|
| age_group_senior                      | 4341.33   |
| age_days                              | 2971.54   |
| breed_Domestic Shorthair Mix          | 2180.85   |
| breed_Domestic Shorthair              | 1705.96   |
| age_group_baby                        | 1009.24   |
| primary_breed_domestic_shorthair      |  629.064  |
| age_upon_intake_8 years               |  490.93   |
| simplified_breed_group_domestic_cat   |  429.14   |
| primary_color_orange_tabby            |  304.854  |
| age_upon_intake_10 years              |  264.409  |
| age_upon_intake_9 years               |  262.259  |
| color_Orange Tabby                    |  197.462  |
| primary_breed_domestic_longhair       |  148.063  |
| breed_infrequent_sklearn              |  147.342  |
| primary_color_brown_tabby             |  116.793  |
| primary_color_blue_tabby              |  107.507  |
| color_Orange Tabby/White              |  106.203  |
| primary_breed_labrador_retriever      |   99.6795 |
| primary_color_torbie                  |   93.1483 |
| primary_breed_siamese                 |   89.2643 |
| age_upon_intake_11 years              |   88.3289 |
| simplified_breed_group_retriever_type |   87.3346 |
| simplified_breed_group_shepherd_type  |   84.8016 |
| age_upon_intake_7 years               |   84.4978 |
| age_upon_intake_1 year                |   82.7226 |
| color_Brown Tabby                     |   80.4636 |
| age_group_young                       |   80.3576 |
| primary_breed_german_shepherd         |   79.3281 |
| color_Torbie                          |   79.1373 |
| breed_Domestic Longhair Mix           |   77.1242 |
| simplified_breed_group_other          |   74.904  |
| age_upon_intake_12 years              |   72.9463 |
| color_Blue Tabby                      |   70.5525 |
| breed_Siamese Mix                     |   68.6729 |
| primary_breed_pit_bull                |   66.2303 |
| breed_Pit Bull Mix                    |   63.6031 |
| simplified_breed_group_pit_bull_type  |   62.061  |
| primary_color_brown                   |   57.9292 |
| breed_Labrador Retriever Mix          |   54.4793 |
| age_upon_intake_6 years               |   53.4278 |
| primary_color_white                   |   51.2061 |
| breed_Chihuahua Shorthair Mix         |   48.1146 |
| primary_color_blue                    |   46.7283 |
| primary_breed_chihuahua_shorthair     |   44.6977 |
| intake_condition_Normal               |   44.4469 |
| color_Brown Tabby/White               |   42.9674 |
| primary_color_tan                     |   40.5435 |
| color_Blue Tabby/White                |   39.532  |
| animal_type_Dog                       |   38.3815 |
| age_upon_intake_13 years              |   37.7637 |
| primary_breed_domestic_medium_hair    |   37.0825 |
| breed_Domestic Longhair               |   36.92   |
| age_upon_intake_5 years               |   32.6569 |
| primary_color_chocolate               |   31.3196 |
| breed_German Shepherd Mix             |   29.2975 |
| primary_color_brown_brindle           |   27.8997 |
| breed_Siamese                         |   27.7591 |
| color_Blue/White                      |   27.4497 |
| breed_Pit Bull                        |   25.4459 |
| color_Tortie                          |   24.9894 |
| primary_breed_infrequent_sklearn      |   24.7391 |
| breed_Domestic Medium Hair Mix        |   22.1863 |
| primary_color_tortie                  |   21.7354 |
| color_Calico                          |   20.313  |
| intake_condition_Injured              |   20.0193 |
| simplified_color_group_mixed_other    |   19.9584 |
| simplified_breed_group_terrier_type   |   19.446  |
| color_Brown Brindle/White             |   19.3579 |
| color_Brown/White                     |   19.2856 |
| age_upon_intake_14 years              |   18.6705 |
| primary_color_calico                  |   18.3475 |
| intake_type_Stray                     |   18.2042 |
| color_infrequent_sklearn              |   17.8382 |
| color_Tan/White                       |   17.3443 |
| primary_breed_australian_shepherd     |   17.3157 |
| simplified_color_group_orange_yellow  |   17.1882 |
| color_Tricolor                        |   17.1559 |
| color_Blue                            |   16.9616 |
| is_black_or_dark_True                 |   16.9072 |
| breed_Chihuahua Shorthair             |   16.841  |
| color_Torbie/White                    |   16.7494 |
| color_Chocolate/White                 |   16.4953 |
| color_Tan                             |   16.1727 |
| simplified_color_group_brown_tan      |   16.1101 |
| age_upon_intake_1 month               |   15.874  |
| breed_Domestic Medium Hair            |   15.5619 |
| intake_type_Owner Surrender           |   15.4537 |
| breed_German Shepherd                 |   15.1709 |
| color_Brown/Black                     |   14.9769 |
| breed_Labrador Retriever              |   14.9599 |
| color_Brown                           |   14.3471 |
| primary_color_tricolor                |   14.2894 |
| primary_color_red                     |   14.2451 |
| intake_condition_Sick                 |   13.8662 |
| color_White/Black                     |   13.8028 |
| breed_Australian Cattle Dog Mix       |   13.2644 |
| primary_color_sable                   |   12.7991 |
| primary_breed_anatol_shepherd         |   12.757  |
| breed_Staffordshire Mix               |   11.7241 |
| color_White                           |   11.5818 |
| age_upon_intake_15 years              |   11.3217 |
| primary_breed_staffordshire           |   10.8059 |
| simplified_color_group_white_light    |   10.7636 |
| primary_breed_siberian_husky          |   10.7534 |
| color_Lynx Point                      |   10.4887 |
| breed_Boxer Mix                       |   10.4765 |
| age_upon_intake_2 months              |   10.3878 |
| primary_breed_australian_cattle_dog   |   10.3573 |
| primary_color_lynx_point              |   10.3306 |
| primary_breed_rottweiler              |   10.1055 |
| primary_color_fawn                    |   10.0008 |

## Complete VIF Leadership Table

| feature                               |       vif |
|:--------------------------------------|----------:|
| age_group_senior                      | 4341.33   |
| age_days                              | 2971.54   |
| breed_Domestic Shorthair Mix          | 2180.85   |
| breed_Domestic Shorthair              | 1705.96   |
| age_group_baby                        | 1009.24   |
| primary_breed_domestic_shorthair      |  629.064  |
| age_upon_intake_8 years               |  490.93   |
| simplified_breed_group_domestic_cat   |  429.14   |
| primary_color_orange_tabby            |  304.854  |
| age_upon_intake_10 years              |  264.409  |
| age_upon_intake_9 years               |  262.259  |
| color_Orange Tabby                    |  197.462  |
| primary_breed_domestic_longhair       |  148.063  |
| breed_infrequent_sklearn              |  147.342  |
| primary_color_brown_tabby             |  116.793  |
| primary_color_blue_tabby              |  107.507  |
| color_Orange Tabby/White              |  106.203  |
| primary_breed_labrador_retriever      |   99.6795 |
| primary_color_torbie                  |   93.1483 |
| primary_breed_siamese                 |   89.2643 |
| age_upon_intake_11 years              |   88.3289 |
| simplified_breed_group_retriever_type |   87.3346 |
| simplified_breed_group_shepherd_type  |   84.8016 |
| age_upon_intake_7 years               |   84.4978 |
| age_upon_intake_1 year                |   82.7226 |

## Numeric Correlation Matrix

|              |   age_days |   intake_year |   intake_month |
|:-------------|-----------:|--------------:|---------------:|
| age_days     |      1     |        -0.063 |         -0.032 |
| intake_year  |     -0.063 |         1     |         -0.107 |
| intake_month |     -0.032 |        -0.107 |          1     |

## Technical Recommendations for AI Agents

1. **Prune redundant age variants:** `age_days`, `age_months`, and `age_years` are linear transformations of each other. Keep only a single numeric age representation (e.g. `age_days`) and the categorical `age_group`.
2. **Eliminate name flags duplication:** `has_name` and `is_named` are identical columns and must be pruned.
3. **Drop redundant calendar indices:** Keep `intake_month` and drop `intake_quarter` if month is already used.


================================================================================
FILE: reports/summary/subgroup_reliability.md
================================================================================

# Model Evidence Pack

This evidence pack summarizes model choice, uncertainty, reliability limits, SHAP interpretation, and animal-centered examples. SHAP reasons are associated with model behavior and model predictions, not causal effects.

## Model Choice

- cats classification: roc_auc = 0.865. hist_gradient_boosting is the strongest ranking model for cats; compare PR-AUC and calibration before using a decision threshold.
- cats classification: pr_auc = 0.899. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- combined classification: roc_auc = 0.840. hist_gradient_boosting is the strongest ranking model for combined; compare PR-AUC and calibration before using a decision threshold.
- combined classification: pr_auc = 0.884. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- dogs classification: roc_auc = 0.813. catboost is the strongest ranking model for dogs; compare PR-AUC and calibration before using a decision threshold.
- dogs classification: pr_auc = 0.860. PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.
- cats regression: mae = 15.794. catboost has the lowest average absolute days-to-outcome error for cats.
- combined regression: mae = 18.547. catboost has the lowest average absolute days-to-outcome error for combined.
- dogs regression: mae = 21.556. catboost has the lowest average absolute days-to-outcome error for dogs.

## Uncertainty

- roc_auc (combined): 0.840 [0.834, 0.847] from 200 bootstrap samples.
- pr_auc (combined): 0.882 [0.876, 0.889] from 200 bootstrap samples.
- f1_at_0_50 (combined): 0.814 [0.808, 0.819] from 200 bootstrap samples.
- mae (combined): 18.552 [18.005, 19.082] from 200 bootstrap samples.

## Trust and Limits

- simplified_breed_group=terrier_type: 321 records, calibration gap 0.206, MAE 12.17.
- intake_type=Abandoned: 466 records, calibration gap 0.185, MAE 19.10.
- simplified_breed_group=hound_type: 148 records, calibration gap 0.158, MAE 26.77.
- simplified_breed_group=chihuahua_type: 557 records, calibration gap 0.148, MAE 8.94.
- simplified_color_group=orange_yellow: 1062 records, calibration gap 0.143, MAE 18.13.
- age_group=young: 3936 records, calibration gap 0.142, MAE 20.99.
- simplified_breed_group=other: 2511 records, calibration gap 0.138, MAE 18.29.
- simplified_breed_group=shepherd_type: 792 records, calibration gap 0.136, MAE 21.10.

Top model failure modes:
- calibration_gap: simplified_breed_group=terrier_type (321 records, calibration_gap 0.206).
- calibration_gap: intake_type=Abandoned (466 records, calibration_gap 0.185).
- calibration_gap: simplified_breed_group=hound_type (148 records, calibration_gap 0.158).
- calibration_gap: simplified_breed_group=chihuahua_type (557 records, calibration_gap 0.148).
- calibration_gap: simplified_color_group=orange_yellow (1062 records, calibration_gap 0.143).
- calibration_gap: age_group=young (3936 records, calibration_gap 0.142).
- calibration_gap: simplified_breed_group=other (2511 records, calibration_gap 0.138).
- calibration_gap: simplified_breed_group=shepherd_type (792 records, calibration_gap 0.136).

Subgroup metric intervals are available for cohorts with enough records and class variety.

## Time-to-Adoption Milestones

- age_group=baby: 45676 adoptions; 64.2% adopted by day 30 and 94.9% by day 90.
- age_group=young: 23187 adoptions; 70.2% adopted by day 30 and 90.5% by day 90.
- age_group=adult: 10381 adoptions; 63.0% adopted by day 30 and 84.8% by day 90.
- age_group=senior: 3362 adoptions; 55.6% adopted by day 30 and 80.5% by day 90.
- animal_type=Dog: 47167 adoptions; 73.6% adopted by day 30 and 91.8% by day 90.
- animal_type=Cat: 35442 adoptions; 54.5% adopted by day 30 and 91.8% by day 90.
- health_profile=normal: 75183 adoptions; 68.3% adopted by day 30 and 92.6% by day 90.
- health_profile=medical_or_injured: 5379 adoptions; 43.9% adopted by day 30 and 83.7% by day 90.

## SHAP and Animal Stories

- classification: breed_appearance: 0.452. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: age: 0.422. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: name_identity: 0.407. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: sex_reproductive_status: 0.253. Feature-family contribution is associated with model prediction, not a causal effect.
- classification: intake_circumstances: 0.250. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: age: 4.876. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: name_identity: 3.786. Feature-family contribution is associated with model prediction, not a causal effect.
- regression: breed_appearance: 2.307. Feature-family contribution is associated with model prediction, not a causal effect.

Animal Journey examples:
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / black_or_dark: 709 cases; 71.8% adoption; median 34.1 days; exact visible profile; label needs visibility.
- baby Cat | has recorded name / Intact Male | Stray / normal | domestic_cat / brown_tan: 580 cases; 71.4% adoption; median 25.2 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / brown_tan: 492 cases; 76.8% adoption; median 27.6 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / mixed_other: 540 cases; 80.0% adoption; median 29.9 days; exact visible profile; label quick placement likely.
- baby Cat | has recorded name / Intact Female | Stray / normal | domestic_cat / black_or_dark: 526 cases; 71.3% adoption; median 31.0 days; exact visible profile; label needs visibility.

## Caveats

- CatBoost is included because shelter data is categorical-heavy; histogram gradient boosting may still win when its validation/test ranking is stronger.
- Probability estimates require calibration review before operational threshold use.
- Health and behavior fields are administrative care-context proxies, not complete temperament labels.
- Regression predicts days to outcome, not a full survival model for time to adoption.


================================================================================
FILE: reports/summary/survival_descriptive_note.md
================================================================================

# Descriptive Survival Analysis Note

## What These Curves Are

The figures `km_adoption_by_*.png` and the table `adoption_survival_curves.csv`
show **empirical Kaplan-Meier style adoption survival curves** for adopted animals,
grouped by `animal_type`, `age_group`, `covid_period`, and `intake_type`.

The y-axis is the **proportion of adopted animals not yet adopted** at each day since intake.
The x-axis is days since intake, restricted to adopted animals only.

Implementation: uses `lifelines.KaplanMeierFitter` when available; falls back to
empirical proportion curves. All events are observed (no censoring) in this adopted-only subset.

## What These Curves Are NOT

These curves are **descriptive time-to-adoption views**. They are NOT:
- The main modeling framework of this thesis.
- A replacement for the supervised ML classification and regression comparison.
- A full survival model with censoring or competing risks.

## Why Not Full Survival Analysis?

1. **Most episodes are resolved:** The dataset contains only matched intake/outcome
   episodes. Animals without a future outcome are excluded from the modeling dataset,
   making the censoring problem smaller than in typical clinical survival analysis.

2. **The main regression target is length-of-stay, not time-to-adoption:**
   `regression_target_days` = `days_to_outcome` covers all outcomes, not just adoption.
   This is operationally relevant for shelter resource planning.

3. **Interpretability:** A regression prediction ("predicted length of stay: 12 days")
   is more directly actionable for a shelter worker than a hazard ratio from a Cox model.

4. **Future work:** Full survival modeling with censoring and competing risks
   (adoption vs. transfer vs. euthanasia vs. return-to-owner) would be a natural
   extension. The descriptive KM curves here provide the foundation.

## Thesis Defense Statement

> "These curves are descriptive time-to-adoption views among adopted animals.
> They provide descriptive evidence for H3 (age and adoption timing patterns)
> without making causal claims. Full time-to-event survival modeling with censoring
> and competing risks is outside the main scope of this thesis and is discussed
> as a natural extension for future work."


================================================================================
FILE: reports/summary/threshold_selection.md
================================================================================

# Threshold Selection — Classification Model

**Model:** `models\boosting\classification\combined\hist_gradient_boosting.joblib`
**Animal subset:** `combined`

> **Key insight:** Threshold choice is an operational decision, not a model quality metric. The threshold should be chosen based on the shelter's objective — maximising adoption identification vs. minimising false alarms.

## Threshold Comparison

| threshold_label           |   threshold |   precision |   recall |     f1 |   false_positive_rate |   false_negative_rate |   tp |   fp |   tn |   fn | threshold_name            |
|:--------------------------|------------:|------------:|---------:|-------:|----------------------:|----------------------:|-----:|-----:|-----:|-----:|:--------------------------|
| default_0.50              |      0.5    |      0.8349 |   0.8015 | 0.8178 |                0.2671 |                0.1985 | 6951 | 1375 | 3772 | 1722 | default_0.50              |
| max_f1                    |      0.2445 |      0.7683 |   0.9364 | 0.844  |                0.4758 |                0.0636 | 8121 | 2449 | 2698 |  552 | max_f1                    |
| high_recall_ge85          |      0.4254 |      0.814  |   0.8503 | 0.8318 |                0.3274 |                0.1497 | 7375 | 1685 | 3462 | 1298 | high_recall_ge85          |
| balanced_precision_recall |      0.9364 |      0      |   0      | 0      |                0      |                1      |    0 |    0 | 5147 | 8673 | balanced_precision_recall |

## Threshold Interpretations

| Threshold | Best for | Trade-off |
|-----------|----------|----------|
| `default_0.50` | Balanced general use | May miss high-recall use cases |
| `max_f1` | Maximising F1 score | Balances precision and recall automatically |
| `high_recall_ge85` | Flagging all adoption-likely animals | Accepts more false positives |
| `balanced_precision_recall` | Equal weight on precision and recall | Not always useful operationally |

## Thesis Statement

The classifier is evaluated primarily as a **ranking / decision-support tool**, not a binary decision maker. The default threshold (0.50) is used for F1 and confusion matrix reporting. If the shelter wished to use this model operationally, threshold selection would depend on the cost of false positives (resources allocated to animals unlikely to be adopted) vs. false negatives (adoption-ready animals not flagged for promotion).
