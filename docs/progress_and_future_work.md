# Progress Review and Future Work Plan

Reviewed on 2026-05-31.

This document records the current state of the AAC adoption thesis project, the goals that are already visible in the repository, the work completed so far, the remaining gaps, and a proposed plan for future work. It is meant to be a working project-management and thesis-planning document, not a final thesis chapter.

## 1. Project Purpose

The project supports the thesis:

> Life-Saving Data: Analyzing Factors Affecting Adoptions at the Austin Animal Center via Machine Learning and Visualization

The repository is no longer just an exploratory notebook space. It is already structured as a reproducible data science pipeline that:

- downloads Austin Animal Center intake and outcome data,
- cleans and joins intake/outcome records into shelter-stay episodes,
- creates leakage-aware intake-time features,
- trains baseline and stronger machine learning models,
- evaluates classification and regression tasks for dogs, cats, and the combined population,
- generates EDA, model comparison, hypothesis-support, and first-level interpretability outputs.

The central analytical framing should remain:

> Build a reproducible analytical system that transforms shelter intake and outcome data into interpretable predictions and operational insights about animal adoption.

This framing is strong because it combines the scientific thesis goal with an implementation goal: the project is not only about fitting a model, but about building a repeatable analytical workflow.

## 2. Current Goals

The project currently has three main goal layers.

### 2.1 Scientific Goal

The scientific goal is to understand which intake-time characteristics are associated with adoption outcomes and length of stay for dogs and cats at Austin Animal Center.

The project should answer questions such as:

- Which intake-time variables are most predictive of adoption?
- How do simpler models compare with stronger tree-based models?
- Do dogs and cats show different adoption patterns?
- How do age, intake circumstances, appearance-related features, and COVID-period timing relate to adoption outcomes?

The project should avoid causal overclaiming. Current models estimate predictive associations, not proven causal effects.

### 2.2 Engineering Goal

The engineering goal is to create a reproducible pipeline, not a one-off analysis. The repo already follows this direction through package code under `src/aac_adoption`, command-line scripts under `scripts`, generated artifacts under `data`, `reports`, and `models`, and automated tests under `tests`.

The future engineering goal should be to make the existing outputs easier to regenerate, summarize, visualize, and present.

### 2.3 Thesis Delivery Goal

The thesis needs stable evidence for chapters on:

- data source and preprocessing,
- feature engineering,
- modeling methodology,
- evaluation results,
- hypothesis discussion,
- interpretability,
- engineering architecture,
- possible dashboard or demonstration layer.

The next stage should focus on converting existing pipeline outputs into thesis-ready tables, plots, and written summaries.

## 3. Hypothesis Priority

The repository already documents that the thesis hypotheses should not all receive equal implementation weight. This is the right decision.

Central hypotheses:

- **H1:** intake circumstances vs breed/color/appearance,
- **H3:** age and adoption speed,
- **H5:** COVID-period adoption dynamics.

Secondary descriptive hypotheses:

- **H2:** seasonality,
- **H4:** black dog/cat syndrome or dark-color effect.

This priority should remain. H1, H3, and H5 form a coherent analytical spine. H2 and H4 should be included through EDA and descriptive tables, but they should not pull the modeling work away from the main story unless later statistical analysis makes them especially compelling.

## 4. Current Repository State

The repository is currently pre-commit: Git reports no commits yet and all files are untracked. That means the current workspace should be treated as a working project snapshot rather than a committed baseline.

Main directories:

- `src/aac_adoption`: reusable package code,
- `scripts`: command-line entry points,
- `tests`: automated test suite,
- `docs`: project and thesis documentation,
- `data/raw`: downloaded raw AAC CSV files,
- `data/processed`: generated modeling dataset and feature/target metadata,
- `reports/tables`: generated EDA, comparison, hypothesis, and interpretability tables,
- `reports/figures`: generated EDA figures,
- `reports/metrics`: generated model metrics,
- `models/baseline` and `models/boosting`: generated model artifacts.

Generated data, reports, and model files are ignored by Git, with `.gitkeep` files preserving the directory structure. That is appropriate for a data science project where raw data and trained artifacts are too large or too volatile to commit.

## 5. Completed Work

### 5.1 Data Acquisition

Implemented:

- raw AAC data downloader,
- historical Socrata source configuration,
- overwrite protection,
- tests for expected Socrata IDs and overwrite behavior.

Current raw files exist locally:

- `data/raw/intakes.csv`,
- `data/raw/outcomes.csv`.

The README documents both historical and current data sources, but the current thesis flow uses the historical 2013-2025 data.

### 5.2 Data Cleaning and Loading

Implemented:

- column-name standardization,
- datetime parsing,
- filtering to cats and dogs,
- duplicate removal,
- required-column validation,
- cleaning functions for intakes and outcomes.

Tests cover:

- mixed AAC datetime formats,
- timezone handling without unintended clock shifting,
- required validation behavior.

This is a solid foundation because timestamp handling and column consistency are common failure points in shelter-record data.

### 5.3 Intake/Outcome Matching

Implemented:

- date-aware matching from each intake to a valid future outcome,
- prevention of negative length-of-stay values,
- repeated-animal handling by matching each intake to the nearest unused future outcome.

This is one of the most important parts of the project. AAC animals can appear multiple times, and naive joins by `animal_id` can duplicate outcomes or create impossible timelines. The current matching logic directly addresses that risk.

### 5.4 Feature Engineering

Implemented intake-time features include:

- animal type,
- intake type,
- intake condition,
- sex upon intake,
- raw and numeric age representations,
- age groups,
- breed simplification,
- mixed-breed flag,
- Found Location coarse taxonomy and flags,
- color simplification,
- black/dark color flag,
- name availability flags,
- intake year, month, quarter, and season,
- COVID-period classification.

Feature metadata is saved in:

- `data/processed/feature_columns.json`.

Target metadata is saved in:

- `data/processed/target_columns.json`.

Current configured feature list contains 31 intake-time predictors. Current configured target list contains 8 target or outcome-derived columns.

### 5.5 Leakage Control

Implemented:

- explicit intake-time feature list,
- explicit target and metadata column lists,
- leakage-column validation,
- tests verifying that outcome-derived columns are excluded.

This is essential. The project predicts adoption outcomes, so outcome-time columns such as `outcome_type`, `outcome_datetime`, `days_to_outcome`, and `length_of_stay` must never be used as predictors.

The current leakage rule is clear: outcome fields may be used for labels, evaluation, and reporting, but not as model features.

### 5.6 Modeling Dataset

Current generated modeling dataset:

- `data/processed/modeling_dataset.csv`.

Current documented build result:

- 162,390 matched intake/outcome rows,
- dogs and cats only,
- time-aware split available,
- leakage-safe model features.

The dataset builder validates:

- required columns,
- animal-type restriction,
- non-negative `days_to_outcome`,
- target creation,
- feature-list leakage control.

### 5.7 EDA Outputs

Implemented:

- EDA table generation,
- basic yearly intake/adoption figures,
- adoption and length-of-stay summaries by key variables.

Current figure outputs include:

- `reports/figures/intakes_by_year.png`,
- `reports/figures/adoptions_by_year.png`.

Current table outputs include summaries by:

- animal type,
- age group,
- intake type,
- intake condition,
- color group,
- COVID period,
- intake season,
- black/dark color flag,
- breed group.

These outputs are useful for thesis data-description and exploratory-analysis sections.

### 5.8 Train/Test Split

Implemented:

- thesis-oriented time-aware split:
  - train: 2013-2021,
  - validation: 2022-2023,
  - test: 2024-2025,
- animal-subset filtering for combined, dogs, and cats,
- fallback behavior when the time split is unavailable.

The time-aware split is appropriate because the thesis is partly about future prediction and changing patterns over time. It is stronger than a purely random split for this use case.

### 5.9 Baseline Models

Implemented classification baselines:

- `DummyClassifier`,
- `LogisticRegression`,
- `RandomForestClassifier`.

Implemented regression baselines:

- `DummyRegressor`,
- `Ridge`,
- `RandomForestRegressor`.

Models are evaluated for:

- combined dog/cat data,
- dogs only,
- cats only.

Artifacts are saved under:

- `models/baseline`.

Metrics are saved under:

- `reports/metrics`.

First-level interpretability tables are saved under:

- `reports/tables/logistic_regression_coefficients.csv`,
- `reports/tables/random_forest_feature_importance.csv`.

### 5.10 Gradient Boosting Models

Implemented:

- histogram gradient boosting classifier,
- histogram gradient boosting regressor,
- combined/dog/cat training and evaluation,
- model artifact saving,
- permutation importance output.

Artifacts are saved under:

- `models/boosting`.

Metrics and tables include:

- `reports/metrics/boosting_classification_metrics.csv`,
- `reports/metrics/boosting_regression_metrics.csv`,
- `reports/metrics/boosting_metrics.csv`,
- `reports/tables/permutation_importance_classification.csv`,
- `reports/tables/permutation_importance_regression.csv`.

This is currently the strongest model family in the project.

### 5.11 Model Comparison

Implemented:

- classification comparison table,
- regression comparison table,
- ranking across baseline and boosting outputs.

Current classification results:

- best combined ROC-AUC: histogram gradient boosting, about 0.840,
- best dog ROC-AUC: histogram gradient boosting, about 0.806,
- best cat ROC-AUC: histogram gradient boosting, about 0.865.

Current regression results:

- best combined MAE: histogram gradient boosting, about 20.21 days,
- best dog MAE: histogram gradient boosting, about 22.62 days,
- best cat MAE: histogram gradient boosting, about 18.28 days.

Interpretation:

- gradient boosting is the current best model family by the primary metrics,
- cats appear easier to classify than dogs in the current feature setup,
- regression remains difficult and should be presented carefully,
- MAE is likely the most thesis-friendly regression metric because it is easy to explain operationally.

### 5.12 Hypothesis Support Tables

Implemented:

- H1 table: `reports/tables/h1_intake_vs_appearance.csv`,
- H3 table: `reports/tables/h3_age_adoption_speed.csv`,
- H5 table: `reports/tables/h5_covid_period.csv`.

H1 currently compares variables such as:

- intake type,
- intake condition,
- simplified breed group,
- simplified color group.

Current H1 descriptive pattern:

- `Owner Surrender` and `Abandoned` records show relatively high adoption rates,
- `Public Assist` and `Euthanasia Request` show much lower adoption rates,
- intake-related features have a strong model-importance signal in the current support table.

H3 currently summarizes adoption by age group.

Current H3 descriptive pattern:

- baby animals have the highest adoption rate,
- young animals are below babies but above adults,
- adult and senior animals show lower adoption rates,
- age-related features have a meaningful importance signal.

H5 currently summarizes adoption by COVID period.

Current H5 descriptive pattern:

- post-COVID and COVID periods show higher adoption rates than pre-COVID in the current table,
- median days to outcome are also higher in COVID and post-COVID periods,
- this needs careful interpretation because period effects may reflect policy, intake volume, data collection, public behavior, and changing population mix.

### 5.13 Tests

Current test command:

```bash
pytest
```

Current result:

- 22 tests passed.

Test coverage currently checks:

- data cleaning,
- downloading safeguards,
- dataset construction,
- feature engineering,
- leakage-safe feature sets,
- time-aware splitting,
- EDA output generation,
- baseline training outputs,
- boosting training outputs,
- artifact path naming,
- analysis table generation.

This is a strong start for a thesis project. The tests focus on pipeline correctness rather than just model scores, which is the right priority.

## 6. Main Strengths

### 6.1 The Pipeline Is Reproducible

The repository already has a coherent reproduction flow:

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
pytest
```

This is already enough to describe a real engineering workflow in the thesis.

### 6.2 The Leakage Boundary Is Explicit

The project clearly separates intake-time predictors from outcome-derived labels and reporting columns. This is a major methodological strength and should be emphasized in the thesis methodology chapter.

### 6.3 The Modeling Story Is Coherent

The comparison between dummy models, linear/interpretable models, random forests, and gradient boosting creates a clean experimental progression:

1. trivial baseline,
2. interpretable baseline,
3. tree-based nonlinear model,
4. stronger gradient boosting model.

This supports the research question about simple versus more complex models.

### 6.4 The Thesis Focus Is Not Overloaded

The decision to prioritize H1, H3, and H5 keeps the project focused. This matters because the thesis could easily become scattered across too many animal, seasonal, color, breed, and dashboard subtopics.

## 7. Current Gaps

### 7.1 Report Generation Is Missing

The biggest current gap is not modeling. The biggest gap is synthesis.

The pipeline generates many CSV files, but there is not yet a formal report-generation script that turns them into:

- thesis-ready Markdown summaries,
- model comparison plots,
- hypothesis-specific figures,
- concise narrative result blocks.

This should be the next technical priority.

### 7.2 Model Comparison Plots Are Missing

Current model comparison is table-based. The thesis will need visual summaries, especially:

- ROC-AUC by model and animal subset,
- F1 or recall by model and animal subset,
- MAE by model and animal subset,
- RMSE by model and animal subset.

These plots will make the model-comparison chapter easier to read.

### 7.3 Hypothesis Figures Are Missing

H1, H3, and H5 have support tables, but they need thesis-ready figures.

Recommended figures:

- H1: adoption rate by intake type, with sample sizes visible or discussed,
- H1: adoption rate by intake condition,
- H1: model-importance comparison for intake-related vs appearance-related features,
- H3: adoption rate and median days to outcome by age group,
- H5: adoption rate and median days to outcome by COVID period.

### 7.4 Historical Note: SHAP Was Not Implemented At This Review Point

Superseded by later work. The thesis draft mentioned SHAP before the repository had SHAP artifacts. The current system now includes sampled CatBoost SHAP global outputs, feature-family SHAP summaries, and local SHAP explanations for representative Animal Journey Cards.

Historical options at that time were:

- implement SHAP if time allows and the dependency/runtime burden is acceptable,
- revise thesis wording to say "permutation importance and model-specific importance" instead of promising SHAP,
- keep SHAP as optional future work.

Current rule: SHAP is implemented; keep SHAP language as predictive association with model behavior, not causal effect.

### 7.5 Historical Note: Streamlit Was Not Implemented At This Review Point

Superseded by later work. The repository now includes an artifact-driven Streamlit dashboard with model quality, interpretability, risk exploration, animal stories, model sensitivity, adoption timeline, and trust/limits views.

Current rule: the dashboard should continue consuming stable CSVs and model artifacts rather than becoming a parallel analysis path.

### 7.6 Formal MLOps Is Not Implemented

Docker, DVC, and MLflow are mentioned in the thesis draft as future engineering topics, but they are not implemented.

This is acceptable for now. Adding them too early would create overhead. The thesis can either:

- describe them as planned/future extensions,
- implement only a minimal Dockerfile later,
- avoid claiming full MLOps implementation unless those pieces are actually added.

### 7.7 Thesis Markdown Encoding Needs Attention

`docs/thesis.md` appears to contain Polish text with mojibake encoding artifacts, for example characters rendered as `Ä`, `Ĺ`, and similar sequences.

This should be fixed before the thesis text becomes central. The code pipeline can continue, but thesis documentation quality will suffer unless the encoding problem is corrected from the original source or converted carefully.

### 7.8 Historical Note: Git Baseline Was Missing At This Review Point

Superseded by later work. The repository now has a baseline commit and several feature commits for reporting, diagnostics, dashboard, animal stories, and Journey Cards.

## 8. Recommended Future Work

Future work should happen in phases. The goal is to protect the current stable base while moving toward thesis-ready outputs.

### Phase 1: Freeze and Document the Current Baseline

Priority: high.

Recommended tasks:

- create an initial Git commit of the current code and documentation,
- keep generated data, models, metrics, and figures ignored unless there is a deliberate reason to version selected lightweight outputs,
- add a short `CHANGELOG.md` or continue using docs for progress notes,
- record the current successful test result.

Acceptance criteria:

- repository has a first commit,
- `pytest` passes,
- README and progress docs accurately describe the project state.

Why this matters:

The project is already useful. A baseline commit prevents future experiments from blurring what is currently known to work.

### Phase 2: Add Report Generation

Priority: highest technical next step.

Recommended script:

- `scripts/generate_report_outputs.py`.

Recommended package module:

- `src/aac_adoption/reporting/report.py` or similar.

Recommended outputs:

- `reports/summary/current_results.md`,
- `reports/figures/model_comparison_classification.png`,
- `reports/figures/model_comparison_regression.png`,
- `reports/figures/h1_intake_type_adoption_rate.png`,
- `reports/figures/h3_age_group_adoption_rate.png`,
- `reports/figures/h5_covid_period_adoption_rate.png`.

The generated Markdown should summarize:

- dataset size,
- split strategy,
- best classification model by subset,
- best regression model by subset,
- key H1/H3/H5 descriptive findings,
- interpretation caveats.

Acceptance criteria:

- one command regenerates thesis-ready summary outputs,
- generated plots are saved in `reports/figures`,
- generated text is concise enough to reuse in thesis drafting,
- tests cover at least the existence and basic schema of generated outputs.

### Phase 3: Strengthen Thesis Interpretation

Priority: high.

Recommended tasks:

- review top logistic-regression coefficients,
- review top random-forest and permutation-importance features,
- group feature importance into conceptual families:
  - intake circumstances,
  - age,
  - breed/appearance,
  - color,
  - timing,
  - name/identity signals,
- connect those groups to H1, H3, and H5.

Acceptance criteria:

- thesis-ready notes explain not just which model won, but what the results mean,
- interpretation language avoids causal claims,
- H1/H3/H5 discussion distinguishes descriptive patterns from model-importance evidence.

### Phase 4: Add Model Comparison and Hypothesis Plots

Priority: high.

Recommended plots:

- classification ROC-AUC by model and subset,
- classification F1 by model and subset,
- regression MAE by model and subset,
- regression RMSE by model and subset,
- H1 adoption rate by intake type,
- H1 adoption rate by intake condition,
- H3 adoption rate by age group,
- H3 median days to outcome by age group,
- H5 adoption rate by COVID period,
- H5 median days to outcome by COVID period.

Acceptance criteria:

- plots have readable labels,
- plots use consistent model and subset ordering,
- figures are thesis-ready without manual spreadsheet editing,
- small groups are either flagged or filtered to avoid misleading bars.

### Phase 5: Historical Note: Decide on SHAP

Superseded by later work. SHAP is now implemented for sampled CatBoost outputs, feature families, and representative Animal Journey Cards.

The implemented approach is:

- start with a sampled dataset,
- focus on the best combined classification model first,
- generate global SHAP summary only before attempting local explanations,
- document runtime constraints.

Current acceptance criteria:

- SHAP output is reproducible,
- sample size is documented,
- SHAP figures are saved as report artifacts,
- thesis text clearly explains the difference between SHAP explanations and causal effects.

### Phase 6: Historical Note: Build Streamlit Prototype

Superseded by later work. The Streamlit prototype exists and remains artifact-driven.

Implemented pages include:

- overview page with dataset and model summary,
- model comparison page,
- hypothesis exploration page,
- feature importance page,
- single-animal model sensitivity page.

Recommended rule:

The Streamlit app should read existing artifacts. It should not retrain models or rebuild datasets during normal use.

Acceptance criteria:

- app starts with one command,
- app loads generated metrics and tables,
- app can load one selected trained model artifact,
- app has clear caveats about prediction uncertainty and non-causal interpretation,
- app works as a thesis demo rather than a production shelter system.

### Phase 7: Optional Engineering Hardening

Priority: low to medium.

Possible tasks:

- add a minimal Dockerfile,
- add a `Makefile` or task runner for the reproduction flow,
- add data checks with richer validation,
- add CI after the repository is hosted remotely,
- add DVC only if data/model versioning becomes necessary,
- add MLflow only if experiment tracking becomes a real need.

Acceptance criteria:

- each added tool solves a clear thesis or reproducibility problem,
- tooling does not make the project harder to run locally,
- documentation stays aligned with what is actually implemented.

## 9. Suggested Immediate Next Sprint

The reporting layer is now implemented. The next sprint should be a consolidation sprint, not a modeling-expansion sprint.

Recommended sprint goal:

> Freeze thesis evidence artifacts, verify dashboard presentation, and prepare a clean source/test/docs commit.

Current implemented state:

- Found Location coarse features are complete and leakage-safe.
- H2 and H4 supporting descriptive outputs are generated by `scripts/run_analysis.py`.
- Evidence pack and reliability artifacts are implemented.
- Artifact manifest generation is implemented.
- Streamlit includes a Generated Artifacts reader for the manifest and thesis methodology reports.

Recommended tasks:

1. Clean remaining encoding artifacts in touched source, tests, and generated summaries.
2. Regenerate analysis summaries and artifact manifest from current code.
3. Inspect the Streamlit dashboard artifact tab locally.
4. Keep H1, H3, and H5 central; keep H2 and H4 as supporting descriptive analyses.
5. Commit source code, tests, docs, and lightweight generated manifests only after tests pass.

Recommended out-of-scope items for this sprint:

- retraining models,
- changing target definitions,
- Docker,
- DVC,
- MLflow,
- hyperparameter tuning.

This keeps the project moving toward thesis submission without destabilizing the working ML pipeline.

## 10. Risks and Mitigations

### Risk: Generated Results Are Treated as Final Conclusions

Mitigation:

Use careful language. The current outputs are reproducible model results and descriptive patterns, not final causal claims.

### Risk: Too Many Hypotheses Dilute the Thesis

Mitigation:

Keep H1, H3, and H5 central. Treat H2 and H4 as supporting descriptive analyses unless stronger evidence is added.

### Risk: Dashboard Work Distracts From Thesis Evidence

Mitigation:

Use Streamlit as an artifact reader and explanation layer. The app should present stable generated outputs, not become the main analysis engine.

### Risk: SHAP Adds Complexity Late

Mitigation:

Make SHAP optional. Current interpretability outputs are already enough for a defensible first interpretation layer.

### Risk: Encoding Problems Affect Thesis Text

Mitigation:

Fix `docs/thesis.md` from a known-good UTF-8 source before expanding the thesis draft.

### Risk: Large Generated Files Make Version Control Awkward

Mitigation:

Keep raw data, processed data, models, and generated reports ignored by default. Commit source code, tests, docs, and lightweight templates.

## 11. Current Verification Snapshot

The test suite was run during this review.

Result:

```text
81 passed
```

Known warning:

```text
FutureWarning in evidence_pack.py during pandas concat with empty/all-NA entries.
```

This confirms that the current data-preparation, feature-engineering, modeling-output, dashboard-helper, artifact-manifest, and analysis-output tests pass in the local environment.

## 12. Practical Definition of Done

For the project to feel thesis-ready, the following should be true:

- the full reproduction flow runs from raw data to final reports,
- tests pass,
- model comparison results are available as tables and figures,
- H1/H3/H5 have both tables and figures,
- interpretation outputs are clearly explained,
- the thesis text matches what the code actually implements,
- the dashboard, if included, presents stable artifacts rather than unfinished experiments,
- limitations are explicit.

The current repository is already past the fragile prototype stage. The most valuable next work is consolidation: make the existing results easier to regenerate, inspect, explain, and place directly into the thesis.

## 13. 2026-05-31 Animal-Centered Story Upgrade

The dashboard and analysis plan now include an animal-first layer designed to make the thesis demo feel grounded in shelter cases rather than generic model metrics.

Implemented:

- advanced CatBoost modeling and diagnostics story layer,
- SHAP and feature-family summaries,
- Streamlit tabs for story mode, model quality, interpretability, risk exploration, campaign finding, model sensitivity, and adoption timeline,
- animal-centered exploratory research command,
- Animal Stories dashboard tab,
- Animal Journey Cards based on profile archetypes,
- pit-bull-type vs other dog comparison,
- black/dark cats vs other cats comparison,
- domestic-cat labels vs other cat breed groups,
- senior vs baby by species,
- named vs unnamed by species,
- health-profile and behavior-support proxy summaries.

New command:

```bash
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
```

New tables:

- `reports/tables/animal_archetypes.csv`
- `reports/tables/vulnerable_profiles.csv`
- `reports/tables/profile_contrasts.csv`
- `reports/tables/profile_model_error.csv`
- `reports/tables/health_behavior_profiles.csv`

New figures:

- `reports/figures/animal_archetypes_top.png`
- `reports/figures/vulnerable_profiles.png`
- `reports/figures/profile_contrasts_adoption_rate.png`

Interpretation guardrail:

The data has condition, subtype, breed, color, age, name, and intake context, but it does not contain a complete personality profile. The implemented behavior/personality angle uses conservative proxy language: `behavior_support_signal` means the administrative record contains behavior-like or special-handling context. It does not mean the animal has a fixed temperament, and it should never be used as a moral label.

Current verification snapshot:

```text
31 passed
```

## 14. 2026-06-04 Journey Cards v2

The next dashboard improvement is now implemented: Animal Journey Cards can move from descriptive archetypes to a prediction-backed demo card.

Implemented:

- representative model record builder for an animal archetype,
- CatBoost prediction display inside Animal Stories,
- prediction-derived visibility label,
- richer similar historical cases with exact/coarse fallback levels,
- similar-case outcome mix for adoption, transfer, return-to-owner, and euthanasia,
- local CatBoost SHAP explanations for the representative card,
- model-wide SHAP fallback reasons when local SHAP cannot run,
- tests for representative records, similar-case summaries, SHAP fallback reasons, and visibility labels.

Verification:

```text
35 passed
```

Browser smoke confirmed that Animal Stories renders:

- Animal Journey Cards,
- Model View for This Journey,
- Predicted adoption chance,
- Similar Historical Cases,
- Top SHAP Reasons.

## 15. 2026-06-04 Model Evidence Pack

The next ML-rigor layer is implemented as an artifact-driven evidence pack.

Implemented:

- PR-AUC as a first-class classification metric,
- `scripts/generate_evidence_pack.py`,
- model evidence summary table,
- bootstrap confidence intervals for ROC-AUC, PR-AUC, F1 at 0.50, and MAE,
- cohort limitation table with calibration gap, MAE, false positive rate, and false negative rate,
- Animal Journey evidence export,
- evidence-pack Markdown summary,
- Trust & Limits dashboard tab,
- report summary integration,
- dedicated documentation in `docs/model_evidence_pack.md`.

New command:

```bash
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
```

New outputs:

- `reports/tables/model_evidence_pack.csv`
- `reports/tables/model_limitations_by_cohort.csv`
- `reports/tables/metric_confidence_intervals.csv`
- `reports/tables/animal_journey_examples.csv`
- `reports/summary/model_evidence_pack.md`

Interpretation guardrail:

The evidence pack is designed for thesis defensibility. It should be used to discuss model choice, uncertainty, and limitations. It should not be used to claim that a feature causes adoption.

## 16. 2026-06-04 Subgroup Reliability and Adoption Milestones

The next ML-rigor layer is implemented. The evidence pack now includes animal-specific reliability evidence and descriptive time-to-adoption milestones.

Implemented:

- subgroup reliability outputs for species, age, intake type, health profile, breed group, color group, and named vs unnamed animals,
- cohort-level observed adoption rate, mean predicted probability, calibration gap, false positive rate, false negative rate, MAE, record count, and small-cohort flag,
- subgroup bootstrap confidence intervals where records and class variety are sufficient,
- model failure-mode table ranking the largest calibration gaps and error rates,
- descriptive adoption milestones at days 7, 30, 60, and 90,
- Trust & Limits dashboard subgroup selector, calibration-gap chart, failure-mode table, interval table, and adoption milestone chart,
- report/evidence summaries that frame subgroup evidence as reliability analysis, not causal inference.

New outputs:

- `reports/tables/subgroup_reliability.csv`
- `reports/tables/subgroup_metric_confidence_intervals.csv`
- `reports/tables/subgroup_adoption_milestones.csv`
- `reports/tables/model_failure_modes.csv`
- `reports/summary/subgroup_reliability.md`

Interpretation guardrail:

Subgroup reliability describes where the current model is more or less trustworthy. Adoption milestones describe historical timing patterns. Neither output proves that a breed, color, name, health descriptor, or intake circumstance causes adoption outcomes.
