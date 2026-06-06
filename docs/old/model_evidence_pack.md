# Model Evidence Pack

The model evidence pack is the ML-rigor layer for the AAC adoption thesis project. It gathers the strongest available model, reliability, uncertainty, SHAP, and animal-centered evidence into stable artifacts that can be used in the thesis and dashboard.

## Command

```bash
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
```

Run this after:

```bash
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
```

## Outputs

```text
reports/tables/model_evidence_pack.csv
reports/tables/model_limitations_by_cohort.csv
reports/tables/metric_confidence_intervals.csv
reports/tables/animal_journey_examples.csv
reports/tables/subgroup_reliability.csv
reports/tables/subgroup_metric_confidence_intervals.csv
reports/tables/subgroup_adoption_milestones.csv
reports/tables/model_failure_modes.csv
reports/summary/model_evidence_pack.md
reports/summary/subgroup_reliability.md
```

Useful options:

```bash
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv --bootstrap-samples 100 --milestone-min-records 50
```

## Subgroup Reliability

The subgroup reliability layer makes model limits animal-specific. It evaluates available cohorts such as dogs, cats, age groups, intake types, health-profile groups, breed groups, color groups, and named vs unnamed animals.

Key fields include:

- `records`: number of diagnostic records in the cohort,
- `observed_adoption_rate`: observed adoption share,
- `mean_predicted_probability`: average predicted adoption probability,
- `calibration_gap`: absolute difference between observed and predicted adoption rates,
- `false_positive_rate` and `false_negative_rate`: threshold-sensitive classification errors,
- `mae`: days-to-outcome regression error,
- `small_cohort`: whether the group is too small for strong claims.

`reports/tables/model_failure_modes.csv` ranks the largest calibration gaps, MAE values, false-negative rates, and false-positive rates. These rows are warning lights for thesis discussion, not proof that a group is intrinsically harder to place.

## Adoption Milestones

`reports/tables/subgroup_adoption_milestones.csv` reports descriptive adoption timing at days 7, 30, 60, and 90 by species, age group, intake type, and health profile. This is time-to-adoption evidence, but it is not a full survival model because censoring and competing events are not modeled rigorously.

## Interpretation Rules

- Use ROC-AUC for ranking quality.
- Use PR-AUC for adoption-positive precision/recall behavior.
- Use calibration and threshold tables before interpreting probabilities as decision-support scores.
- Use MAE as the main days-to-outcome regression metric.
- Use SHAP as model-explanation evidence only: features are associated with model predictions, not causes of adoption.
- Treat health and behavior fields as administrative care-context proxies, not complete personality or temperament labels.
- Treat animal profile findings as descriptive/exploratory unless supported by additional statistical testing.

## Dashboard Use

The Streamlit `Trust & Limits` tab reads the evidence-pack artifacts and shows:

- metric confidence intervals,
- cohort reliability limits,
- subgroup reliability selector and calibration-gap chart,
- top model failure modes,
- subgroup metric confidence intervals,
- descriptive adoption milestone chart,
- calibration/error-sensitive cohorts,
- Animal Journey evidence examples.

The dashboard remains artifact-driven and does not retrain models at runtime.
