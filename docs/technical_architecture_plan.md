# Technical Architecture Plan

This project is a reproducible Data Science system for Austin Animal Center adoption analysis, not a loose notebook or a single model demo.

## Core Framing

Strong thesis framing:

> I built a reproducible analytical system that transforms shelter intake and outcome data into interpretable predictions and operational insights about animal adoption.

## Scope Priority

Central analytical angles:

1. H1: intake circumstances vs appearance.
2. H3: age and adoption speed.
3. H5: COVID-period adoption dynamics.

Secondary descriptive angles:

1. H2: seasonality.
2. H4: black dog/cat syndrome / dark-color effect.

## Leakage Rule

Models must use intake-time-only predictors.

Allowed predictors include:

- `animal_type`
- `intake_type`
- `intake_condition`
- `sex_upon_intake`
- `age_upon_intake`
- `breed`
- `color`
- `intake_year`
- `intake_month`
- `intake_quarter`
- `intake_season`
- `has_name`
- `is_named`
- `is_mixed_breed`
- `primary_breed`
- `simplified_breed_group`
- `primary_color`
- `simplified_color_group`
- `is_black_or_dark`
- `age_group`
- `covid_period`

Outcome-derived columns must never be predictors:

- `outcome_type`
- `outcome_subtype`
- `outcome_datetime`
- `sex_upon_outcome`
- `age_upon_outcome`
- `days_to_outcome`
- `days_to_adoption`
- `length_of_stay`
- `adopted`
- `is_adopted`
- `classification_target`
- `regression_target_days`
- `target_adopted`

Outcome columns are allowed only for target creation, evaluation, and reporting.

## Current Status

Completed:

- raw AAC downloader,
- raw-to-processed modeling dataset pipeline,
- date-aware intake/outcome matching,
- intake-time feature engineering,
- leakage-safe feature lists,
- time-aware thesis split,
- baseline models,
- histogram gradient boosting models,
- dog/cat/combined evaluation,
- saved metrics and model artifacts,
- first-level interpretability outputs,
- H1/H3/H5 support tables.

Not yet implemented:

- Streamlit prototype,
- SHAP,
- Docker/DVC/MLflow,
- formal report-generation script,
- model comparison plots.

## Milestones

Milestone 1: data foundation. Completed.

- downloader,
- raw loading,
- cleaning,
- intake/outcome date-aware matching,
- feature engineering,
- processed modeling dataset,
- validation checks.

Milestone 2: EDA and baseline models. Completed.

- tables,
- figures,
- Dummy/Logistic/RandomForest classification,
- Dummy/Ridge/RandomForest regression,
- saved metrics.
- dog/cat/combined evaluation,
- saved baseline model artifacts,
- logistic regression coefficients,
- random forest feature importance.

Milestone 3: stronger models and interpretation. Mostly completed.

- gradient boosting,
- feature importance,
- permutation importance,
- tables for H1/H3/H5.
- SHAP later if feasible.

Remaining work for Milestone 3:

- model comparison plots,
- Markdown/CSV report summaries for thesis writing,
- optional SHAP after current outputs are stable.

Milestone 4: Streamlit prototype.

- prediction page,
- what-if page,
- feature importance page,
- thesis demo only, not production system.

## Current Rule

Do not add Docker, DVC, MLflow, or heavy tuning yet. Streamlit should wait until report tables and plots are thesis-ready.

## Recommended Next Step

Add a report-generation layer that turns existing metrics and hypothesis tables into:

- compact Markdown summaries,
- model comparison plots,
- H1/H3/H5 figures.

Then build Streamlit as a visualization prototype over stable artifacts.
