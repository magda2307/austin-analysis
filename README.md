# AAC Adoption ML Pipeline

Code foundation for the thesis project:

**Life-Saving Data: Analyzing Factors Affecting Adoptions at the Austin Animal Center via Machine Learning and Visualization**

This repository focuses first on a clean, reproducible data and ML pipeline for Austin Animal Center dog/cat adoption analysis.

## Current Status

Implemented:

- raw data downloader from Austin Open Data,
- reproducible modeling dataset builder,
- intake-time-only feature set,
- time-aware train/validation/test split,
- baseline models,
- histogram gradient boosting models,
- dog/cat/combined evaluation,
- model artifacts,
- EDA tables and figures,
- model comparison tables,
- H1/H3/H5 support tables,
- first-level interpretability outputs.

Not implemented yet:

- Streamlit dashboard,
- SHAP,
- Docker/DVC/MLflow,
- formal report-generation script.

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
- **H3:** age and adoption speed,
- **H5:** COVID-period change.

Secondary descriptive threads:

- **H2:** seasonality,
- **H4:** black dog/cat syndrome.

The code should first support robust dataset construction, baseline modeling, and clean outputs for H1/H3/H5. H2 and H4 should remain available through prepared variables such as `intake_month`, `intake_season`, `color`, and `color_group`, but they should not dominate the first modeling work.

Not in current scope:

- Streamlit dashboard,
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

## Build Modeling Dataset

Install dependencies:

```bash
pip install -r requirements.txt
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

## Run Thesis Analysis Tables

After baseline and boosting metrics exist, run:

```bash
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
```

Outputs:

```text
reports/tables/model_comparison_classification.csv
reports/tables/model_comparison_regression.csv
reports/tables/h1_intake_vs_appearance.csv
reports/tables/h3_age_adoption_speed.csv
reports/tables/h5_covid_period.csv
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

## Current Results Snapshot

Current full-data model comparison:

- best classification model by ROC-AUC: `hist_gradient_boosting`,
- combined classification ROC-AUC: about `0.840`,
- dogs classification ROC-AUC: about `0.806`,
- cats classification ROC-AUC: about `0.865`,
- best regression model by MAE: `hist_gradient_boosting`,
- combined regression MAE: about `20.21` days,
- dogs regression MAE: about `22.62` days,
- cats regression MAE: about `18.28` days.

Use these numbers as current pipeline outputs, not final thesis conclusions. Models show predictive association, not causality.

## Full Reproduction Flow

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
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
