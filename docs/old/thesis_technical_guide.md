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
