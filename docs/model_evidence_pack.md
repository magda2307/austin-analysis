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
reports/summary/model_evidence_pack.md
```

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
- calibration/error-sensitive cohorts,
- Animal Journey evidence examples.

The dashboard remains artifact-driven and does not retrain models at runtime.
