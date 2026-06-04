# Current Results Summary

This document summarizes the current reproducible outputs of the AAC adoption ML pipeline. It is intended as a working reference for thesis chapters on data preparation, modeling, evaluation, and interpretation.

## Dataset

Current processed dataset:

```text
data/processed/modeling_dataset.csv
```

Current build result:

- matched intake/outcome rows: 162,390
- animal types: dogs and cats only
- split strategy: time-aware
- train: 2013-2021
- validation: 2022-2023
- test: 2024-2025
- leakage control: model features are intake-time-only

Raw files:

```text
data/raw/intakes.csv
data/raw/outcomes.csv
```

Raw data and generated outputs are ignored by Git.

## Model Families

Implemented model families:

- dummy baselines,
- logistic regression,
- random forest,
- histogram gradient boosting.

Tasks:

- classification: predict adoption vs non-adoption,
- regression: predict days to outcome / length of stay.

Each task is evaluated for:

- combined dog/cat dataset,
- dogs only,
- cats only.

## Classification Results

Main comparison file:

```text
reports/tables/model_comparison_classification.csv
```

Best ROC-AUC by subset:

- combined: `hist_gradient_boosting`, ROC-AUC about 0.840
- dogs: `hist_gradient_boosting`, ROC-AUC about 0.806
- cats: `hist_gradient_boosting`, ROC-AUC about 0.865

Interpretation:

- gradient boosting currently performs best by ROC-AUC across all subsets,
- logistic regression remains useful as an interpretable baseline,
- cats appear slightly easier to classify than dogs in the current feature setup.

## Regression Results

Main comparison file:

```text
reports/tables/model_comparison_regression.csv
```

Best MAE by subset:

- combined: `hist_gradient_boosting`, MAE about 20.21 days
- dogs: `hist_gradient_boosting`, MAE about 22.62 days
- cats: `hist_gradient_boosting`, MAE about 18.28 days

Interpretation:

- regression is harder than classification,
- dummy median remains a meaningful baseline because length of stay is skewed,
- MAE should be emphasized over R2 because it is easier to explain operationally.

## Hypothesis Support Tables

Central thesis hypotheses:

```text
reports/tables/h1_intake_vs_appearance.csv
reports/tables/h3_age_adoption_speed.csv
reports/tables/h5_covid_period.csv
```

H1: intake circumstances vs appearance.

- table combines `intake_type`, `intake_condition`, `simplified_breed_group`, and `simplified_color_group`,
- includes adoption rate, median days to outcome, and related model-importance signals.

H3: age and adoption speed.

- table summarizes adoption outcomes by `age_group`,
- current descriptive pattern: baby animals have higher adoption rate than adult/senior animals.

H5: COVID-period adoption dynamics.

- table summarizes adoption outcomes by `pre_covid`, `covid`, and `post_covid`,
- current descriptive pattern shows differences across periods, but causal claims should be avoided.

Secondary descriptive hypotheses:

```text
reports/tables/adoption_los_by_intake_season.csv
reports/tables/adoption_los_by_is_black_or_dark.csv
```

H2 and H4 should stay descriptive unless later analysis adds stronger statistical tests.

## Interpretability Outputs

Current interpretability files:

```text
reports/tables/logistic_regression_coefficients.csv
reports/tables/random_forest_feature_importance.csv
reports/tables/permutation_importance_classification.csv
reports/tables/permutation_importance_regression.csv
reports/tables/shap_global_classification.csv
reports/tables/shap_global_regression.csv
reports/tables/shap_feature_families_classification.csv
reports/tables/shap_feature_families_regression.csv
```

Use these to discuss predictive importance, not causality.

Recommended thesis language:

> The model identifies variables associated with adoption outcomes and length of stay. These associations do not prove causal effects, but they help indicate which intake-time characteristics are most informative for prediction.

## Reproduction Commands

Full current pipeline:

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
pytest
```

## Current Technical Focus

The subgroup validation layer is now implemented in the evidence pack. Use `reports/tables/subgroup_reliability.csv`, `reports/tables/subgroup_metric_confidence_intervals.csv`, `reports/tables/model_failure_modes.csv`, and `reports/tables/subgroup_adoption_milestones.csv` to discuss where the model is more or less reliable for dogs, cats, age groups, intake/health profiles, appearance groups, and named vs unnamed animals.

Recommended next step:

1. keep thesis text aligned with the subgroup evidence-pack outputs,
2. deepen selected cohort discussion where calibration gaps or MAE are high,
3. treat survival-style analysis as descriptive unless a full time-to-event model is added.
