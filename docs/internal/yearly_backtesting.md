# Yearly Temporal Backtesting

This document describes the yearly temporal backtesting implementation, which evaluates model performance across rolling historical windows to assess temporal stability.

## Rolling Window Approach

The backtesting uses a **rolling historical window** approach:

- **Training window**: Always starts at 2013 and ends at (test_year - 1)
- **Test window**: A single calendar year

For the 2019-2024 dataset, this produces:

| Train Period | Test Year |
|--------------|-----------|
| 2013-2018    | 2019      |
| 2013-2019    | 2020      |
| 2013-2020    | 2021      |
| 2013-2021    | 2022      |
| 2013-2022    | 2023      |
| 2013-2023    | 2024      |

This approach:
- Evaluates whether models trained on historical data generalize to future years
- Provides stronger evidence of temporal stability than a single chronological split
- Reveals performance changes after events like COVID-19 or operational shifts

## Models Tested

Two classifier and two regressor variants are evaluated:

### Classification Models

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| CatBoostClassifier | Gradient boosting | iterations=100, depth=6, learning_rate=0.1, auto_class_weights="Balanced" |
| HistGradientBoostingClassifier | Gradient boosting | max_iter=100, max_depth=6, learning_rate=0.1, min_samples_leaf=10 |

### Regression Models

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| CatBoostRegressor | Gradient boosting | iterations=100, depth=6, learning_rate=0.1 |
| HistGradientBoostingRegressor | Gradient boosting | max_iter=100, max_depth=6, learning_rate=0.1, min_samples_leaf=10 |

All models receive categorical features (where applicable) and are trained with fixed random state for reproducibility.

## Metrics Computed

### Classification Metrics

| Metric | Description |
|--------|-------------|
| PR-AUC | Area under the Precision-Recall curve (primary for class-imbalanced tasks) |
| ROC-AUC | Area under the ROC curve |
| Brier Score | Mean squared difference between predicted probabilities and outcomes |
| ECE | Expected Calibration Error (fixed 10-bin calibration) |

### Regression Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |

### Cluster-Aware Bootstrap Confidence Intervals

95% confidence intervals are computed using **cluster-aware bootstrap**:

1. **Animal-level clustering**: When `animal_id` is available, bootstrap resamples animals with replacement and includes all observations from selected animals
2. **Fallback to row-level**: If animal IDs are unavailable, falls back to row-level bootstrap
3. **bootstrap iterations**: 100 (configurable, default in production: 1000)

This accounts for the non-independence of multiple episodes per animal.

## Limitations

### Batch-Dependent Rolling Features

The following features are computed from batch processing and are **not suitable for online inference**:

- `intake_volume_7d` / `intake_volume_30d`: Rolling intake counts by shelter
- `animal_311_requests_7d` / `animal_311_requests_30d`: Rolling 311 request counts by location

These features require historical context rebuilt for each batch, making them incompatible with single-record online inference unless the entire historical context is reconstructed.

### Temporal Data leakage

While rolling window evaluation helps detect temporal leakage, models may still use features that:
- Contain information from after the outcome date
- Have been updated post-intake (e.g., `sex_upon_intake`, `intake_condition`)

Always verify feature importance and SHAP values against known leakage patterns.

## Sample Output CSV Format

```csv
train_years,test_year,subset,model,pr_auc,roc_auc,brier,ece,mae,rmse,r2,pr_auc_lower,pr_auc_upper,roc_auc_lower,roc_auc_upper,brier_lower,brier_upper,ece_lower,ece_upper,mae_lower,mae_upper,rmse_lower,rmse_upper,r2_lower,r2_upper
2013-2018,2019,combined,catboost_classifier,0.752,0.821,0.112,0.045,,,,0.731,0.771,0.802,0.838,0.101,0.123,0.038,0.052,,,
2013-2018,2019,combined,histgradientboosting_classifier,0.748,0.815,0.115,0.048,,,,0.727,0.769,0.796,0.832,0.104,0.127,0.041,0.055,,,
2013-2018,2019,combined,catboost_regressor,,,,,0.245,0.312,0.423,,,,,,0.221,0.269,0.291,0.333,0.398,0.448
2013-2018,2019,combined,histgradientboosting_regressor,,,,,0.251,0.321,0.412,,,,,,0.227,0.275,0.298,0.342,0.389,0.436
2013-2019,2020,combined,catboost_classifier,0.761,0.828,0.108,0.042,,,,0.740,0.782,0.809,0.845,0.097,0.119,0.036,0.048,,,
...
2013-2023,2024,combined,catboost_classifier,0.758,0.825,0.110,0.044,,,,0.737,0.779,0.806,0.842,0.099,0.121,0.038,0.050,,,
2013-2023,2024,combined,histgradientboosting_classifier,0.754,0.821,0.113,0.046,,,,0.733,0.775,0.802,0.838,0.102,0.124,0.040,0.052,,,
```

Columns:
- `train_years`: Training window (e.g., "2013-2018")
- `test_year`: Calendar year used for testing
- `subset`: Animal subset ("combined", "dogs", "cats")
- `model`: Model identifier
- Standard metrics (PR-AUC, ROC-AUC, Brier, ECE, MAE, RMSE, R²)
- Lower/upper bounds for 95% bootstrap confidence intervals

## Acceptance Criteria

From ROADMAP.md [Slice 13 - Yearly Temporal Backtesting](../ROADMAP.md):

- [x] Rolling historical windows implemented
- [x] Output CSV with train_years, test_year, subset, model, and metrics
- [x] Bootstrap confidence intervals with cluster-aware option
- [x] Animal subsets: combined, dogs, cats
- [x] Classification metrics: PR-AUC, ROC-AUC, Brier, ECE
- [x] Regression metrics: MAE, RMSE, R²
- [x] Tests verify all models produce expected metrics
- [x] Empty train/test splits skipped gracefully

## References

- [`scripts/evaluate_backtesting.py`](../scripts/evaluate_backtesting.py): CLI script
- [`src/aac_adoption/models/yearly_backtesting.py`](../../src/aac_adoption/models/yearly_backtesting.py): Core implementation
- [`tests/test_yearly_backtesting.py`](../../../tests/test_yearly_backtesting.py): Regression tests
