```markdown
# Integration Summary: survival_diagnostics.py with survival_analysis.py

This module, `survival_diagnostics.py`, extends `survival_analysis.py` with comprehensive diagnostic capabilities for Cox proportional hazards models.

## Key Integrations

### 1. PH Assumption Validation
- Uses `CoxPHFitter` from lifelines (same as `survival_analysis.py`)
- Computes Schoenfeld residuals via `cph.proportional_hazard_test()`
- Returns DataFrame with chi-square stats and p-values per variable
- Follows same pandas-based output pattern

### 2. Calibration Testing (Hosmer-Lemeshow style)
- Uses risk scores from `cph.predict_partial_hazard()`
- Groups observations by risk quantiles
- Computes observed vs expected event rates
- Chi-square test with `n_bins - 2` degrees of freedom

### 3. Influence Diagnostics (dfbeta)
- Uses `cph.compute_dfbetas()` from lifelines
- Identifies observations with large coefficient changes
- Threshold: `|dfbeta| > 2` indicates influence

### 4. Deviance Residuals
- Uses `cph.residuals_` from lifelines
- Identifies outliers in model fit
- Q-Q plot for distribution assessment

### 5. Time-Dependent Covariate Analysis
- Linear regression of covariates vs time
- Slope t-test for time-dependent effects
- Threshold: `|t-statistic| > 1.96` indicates time-dependence

### 6. Data Validation Helpers
- `validate_data_for_survival()` mirrors `add_censoring_indicators()`
- Checks duration, event columns, missing values
- Returns quality assessment (good/acceptable/poor)

### 7. Plotting Functions
- Uses matplotlib same as `survival_analysis.py` visualization
- KaplanMeierFitter for survival curves
- Follows same file naming conventions
- Output Path pattern consistent with existing code

### 8. Integration with train_survival.py
- `SurvivalDiagnosticsRunner` integrates with `train_survival.cox` training
- Can be called after Cox model fitting in `train_survival.py`
- `save_reports()` outputs CSV files matching metrics directory structure

## Functions Exported
- check_proportional_hazards_assumption
- create_log_log_plot
- check_calibration_survival
- compute_dfbeta_values
- compute_deviance_residuals
- check_time_dependent_covariates
- plot_survival_curves_diagnostics
- plot_dfbeta_diagnostics
- plot_deviance_residuals
- validate_data_for_survival
- generate_comprehensive_diagnostics
- SurvivalDiagnosticsRunner

All functions share the `np.random.seed(42)` seed for reproducibility.
```
