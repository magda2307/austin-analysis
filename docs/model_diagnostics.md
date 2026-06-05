# Model Diagnostics and Decision-Support Views

This project now includes an advanced diagnostics layer for evaluating model reliability and practical usefulness.

## Advanced Models

CatBoost models are trained as an advanced model family because the AAC dataset contains many categorical predictors such as intake type, condition, breed group, color group, sex status, season, and COVID period.

Run:

```bash
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
```

The existing scikit-learn models remain baselines. CatBoost is compared under the same time-aware split.

## Reliability Diagnostics

Run:

```bash
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
```

The diagnostics layer creates:

- ROC and precision-recall curve artifacts,
- probability calibration tables and figures,
- threshold tradeoff tables,
- classification error slices,
- regression residual and error-slice tables,
- placement-risk quadrants,
- adoption timeline milestones,
- SHAP global and feature-family explanations.

## Interpretation Rules

SHAP values explain how features contributed to this model's prediction. They do not prove that changing a feature would causally change adoption probability.

Recommended thesis phrasing:

> The interpretability outputs show which intake-time characteristics are most associated with the model predictions. These associations help explain predictive behavior, but they should not be interpreted as direct causal effects on adoption. SHAP values explain how features contributed to this model's prediction. They do not prove that changing a feature would causally change adoption probability.

## Dashboard Views

The Streamlit app uses generated artifacts and does not retrain models at runtime. The key diagnostic views are:

- Model Quality,
- Interpretability,
- Risk Explorer,
- Campaign Finder,
- Adoption Timeline,
- Model Sensitivity Demo with similar historical cases.
