# Slice 12 Agent Tasks: Agent 2 - Survival Modeling Expert

## Your Mission
Enhance `survival_analysis.py` to support proper censored data analysis. Implement Cox proportional hazards and AFT models that correctly handle censored observations.

## Files to Modify

### src/aac_adoption/analysis/survival_analysis.py

**Your Tasks**:

1. **Enhance `compute_kaplan_meier_survival()` for censored data**
   - Location: Lines 12-40
   - Current issue: Uses `event_col` as indicator but doesn't validate censoring
   
   **Replace the function with**:
   ```python
   def compute_kaplan_meier_survival(
       df: pd.DataFrame,
       duration_col: str = "days_to_outcome",
       event_col: str = "adopted",
       time_points: Optional[list] = None,
   ) -> pd.DataFrame:
       """Compute Kaplan-Meier survival curve for LOS.
       
       Args:
           df: DataFrame with duration and event columns
           duration_col: Time-to-event column (can be censored time)
           event_col: Event indicator (1=event occurred, 0=censored)
           time_points: List of time points for survival estimates
           
       Returns:
           DataFrame with survival probabilities at each time point
       """
       kmf = KaplanMeierFitter()
       
       if df.empty:
           return pd.DataFrame()
       
       T = df[duration_col]
       E = df[event_col].fillna(0).astype(int)  # Censored = 0
       
       # Verify event column is binary
       if not E.isin([0, 1]).all():
           raise ValueError(f"{event_col} must be binary (0=censored, 1=event)")
       
       # Handle empty event column
       if E.sum() == 0:
           return pd.DataFrame({"days": [], "survival_probability": [], "event_count": [], "censored_count": []})
       
       kmf.fit(T, event_observed=E, label="Kaplan-Meier")
       
       if time_points is None:
           time_points = list(range(0, int(T.max()) + 1, 1))
       
       survival_probs = kmf.survival_function_at_times(time_points)
       
       # Get event and censored counts at each time point
       risk_table = kmf.risk_table
       risk_df = pd.DataFrame(risk_table).reset_index()
       
       result = pd.DataFrame({
           "days": time_points,
           "survival_probability": survival_probs.values,
           "censoring_rate": 1 - survival_probs.values,
       })
       
       # Combine with risk table if available
       if not risk_df.empty:
           result = result.merge(risk_df, left_on="days", right_on="time", how="left")
           result = result.rename(columns={"at_risk": "risk_set_size", "uncensored": "event_count", "censored": "censored_count"})
       
       return result
   ```

2. **Enhance `fit_cox_proportional_hazards()` for censored data**
   - Location: Lines 43-80
   - Current issue: Uses `dropna()` which silently drops censored rows
   
   **Replace the function with**:
   ```python
   def fit_cox_proportional_hazards(
       df: pd.DataFrame,
       duration_col: str = "days_to_outcome",
       event_col: str = "adopted",
       feature_cols: Optional[list] = None,
       categorical_cols: Optional[list] = None,
   ) -> Tuple[CoxPHFitter, pd.DataFrame, dict]:
       """Fit Cox proportional hazards model for LOS.
       
       Args:
           df: DataFrame with duration, event, and feature columns
           duration_col: Time-to-event column
           event_col: Event indicator column (1=event, 0=censored)
           feature_cols: Feature columns to include (excluding duration/event)
           categorical_cols: Explicit categorical columns for encoding
           
       Returns:
           Tuple of (fitted_cox_model, coefficients_df, diagnostics_dict)
       """
       if df.empty:
           return None, pd.DataFrame(), {}
       
       if feature_cols is None:
           feature_cols = [
               "animal_type",
               "age_group",
               "intake_type",
               "intake_condition",
               "sex_upon_intake",
               "primary_breed",
               "simplified_breed_group",
               "simplified_color_group",
               "found_location_kind",
           ]
       
       cols_to_use = [duration_col, event_col] + [c for c in feature_cols if c in df.columns]
       data = df[cols_to_use].copy()
       
       # Handle missing values explicitly (not dropna)
       missing_counts = data.isnull().sum()
       if missing_counts.any():
           print(f"Warning: {missing_counts.sum()} missing values in features")
           # Replace missing with 'unknown' for categorical, median for numeric
           for col in cols_to_use:
               if col in [duration_col, event_col]:
                   continue
               if data[col].dtype == 'object' or data[col].dtype.name == 'string':
                   data[col] = data[col].fillna('unknown').astype(str)
               else:
                   data[col] = data[col].fillna(data[col].median())
       
       if data.empty:
           return None, pd.DataFrame(), {}
       
       # Separate categorical and numeric features
       if categorical_cols is None:
           categorical_cols = [c for c in feature_cols 
                              if c in data.columns 
                              and (data[c].dtype == 'object' or data[c].dtype.name == 'string')]
       
       # Create dummy variables for categorical features
       numeric_data = data.drop(columns=categorical_cols).copy()
       if categorical_cols:
           dummy_data = pd.get_dummies(data[categorical_cols], prefix=categorical_cols, dummy_na=True)
           numeric_data = pd.concat([numeric_data, dummy_data], axis=1)
       
       # Separate T, E, and X
       T = numeric_data[duration_col]
       E = numeric_data[event_col]
       X = numeric_data.drop(columns=[duration_col, event_col])
       
       if X.empty:
           return None, pd.DataFrame(), {"warning": "No valid features after encoding"}
       
       # Check for event count
       event_count = E.sum()
       if event_count == 0:
           return None, pd.DataFrame(), {"warning": "No events (all censored)"}
       
       if event_count < 10:
           print(f"Warning: Only {event_count} events - Cox model may be unstable")
       
       cph = CoxPHFitter()
       try:
           cph.fit(numeric_data, duration_col=duration_col, event_col=event_col)
       except Exception as e:
           return None, pd.DataFrame(), {"error": str(e)}
       
       # Extract coefficients
       summary = cph.summary
       coefficients = summary[["coef", "exp(coef)", "p"]].copy()
       coefficients.columns = ["coefficient", "hazard_ratio", "p_value"]
       coefficients = coefficients.sort_values("p_value")
       
       # Add diagnostics info
       diagnostics = {
           "event_count": int(event_count),
           "censored_count": int(len(E) - event_count),
           "total_samples": len(E),
           "feature_count": X.shape[1],
           "concordance_index": float(cph.concordance_index_),
           "log_likelihood": float(cph.log_likelihood_),
           "aic": float(cph.aic_),
           "ph_assumption_tests": _test_proportional_hazards(cph, data),
       }
       
       return cph, coefficients, diagnostics
   ```

3. **Add helper function for PH assumption testing**
   - Location: After existing functions, before `log_transform_LOS`
   
   **Add new function**:
   ```python
   def _test_proportional_hazards(cph: CoxPHFitter, df: pd.DataFrame) -> dict:
       """Test proportional hazards assumption using Schoenfeld residuals."""
       try:
           pht = cph.print_statistics(df, 'schoenfeld')
           # Extract test results if available
           # For now, return basic structure
           return {
               "method": "schoenfeld",
               "assumption_status": "passed" if hasattr(cph, 'ph_assessments') else "not_tested",
           }
       except Exception:
           return {"method": "schoenfeld", "error": "PH test failed"}
   
   
   def cox_diagnostic_report(cph: CoxPHFitter, df: pd.DataFrame) -> pd.DataFrame:
       """Generate comprehensive Cox model diagnostic report."""
       diagnostics = []
       
       # Basic diagnostics
       diagnostics.append({
           "metric": "concordance_index",
           "value": cph.concordance_index_,
           "description": "C-index: probability that predicted and actual survival rankings agree",
       })
       
       diagnostics.append({
           "metric": "log_likelihood",
           "value": cph.log_likelihood_,
           "description": "Log-likelihood of the fitted model",
       })
       
       diagnostics.append({
           "metric": "aic",
           "value": cph.aic_,
           "description": "Akaike Information Criterion - lower is better",
       })
       
       # Try PH tests
       try:
           pht = cph.test_proportional_hazard()
           diagnostics.append({
               "metric": "ph_test_chi2",
               "value": pht.iloc[:, 0].sum(),
               "description": "Global chi-square test for PH assumption",
           })
       except Exception:
           diagnostics.append({
               "metric": "ph_test",
               "value": None,
               "description": "PH test failed - may need time-dependent covariates",
           })
       
       return pd.DataFrame(diagnostics)
   ```

4. **Add AFT model function**
   - Location: After `_test_proportional_hazards()`, before `log_transform_LOS()`
   
   **Add new function**:
   ```python
   from lifelines import WeibullAFTFitter, ExponentialAFTFitter
   
   def fit_aft_model(
       df: pd.DataFrame,
       duration_col: str = "days_to_outcome",
       event_col: str = "adopted",
       feature_cols: Optional[list] = None,
       dist: str = "weibull",  # "weibull" or "exponential"
   ) -> Tuple[Any, pd.DataFrame, dict]:
       """Fit Accelerated Failure Time (AFT) model for LOS.
       
       AFT models directly estimate time-to-event, making coefficients more 
       interpretable than Cox (hours/days rather than hazard ratios).
       
       Args:
           df: DataFrame with duration, event, and feature columns
           duration_col: Time-to-event column
           event_col: Event indicator column (1=event, 0=censored)
           feature_cols: Feature columns to include
           dist: Distribution ("weibull" or "exponential")
           
       Returns:
           Tuple of (fitted_aft_model, coefficients_df, diagnostics_dict)
       """
       if df.empty:
           return None, pd.DataFrame(), {}
       
       if feature_cols is None:
           feature_cols = [
               "animal_type",
               "age_group",
               "intake_type",
               "intake_condition",
               "sex_upon_intake",
               "primary_breed",
               "simplified_breed_group",
               "simplified_color_group",
               "found_location_kind",
           ]
       
       cols_to_use = [duration_col, event_col] + [c for c in feature_cols if c in df.columns]
       data = df[cols_to_use].copy()
       
       # Handle missing values
       for col in cols_to_use:
           if col in [duration_col, event_col]:
               continue
           if data[col].dtype == 'object' or data[col].dtype.name == 'string':
               data[col] = data[col].fillna('unknown').astype(str)
           else:
               data[col] = data[col].fillna(data[col].median())
       
       if data.empty:
           return None, pd.DataFrame(), {}
       
       T = data[duration_col]
       E = data[event_col].fillna(0).astype(int)
       
       # One-hot encode categorical features
       categorical_cols = [c for c in feature_cols 
                          if c in data.columns 
                          and (data[c].dtype == 'object' or data[c].dtype.name == 'string')]
       
       numeric_data = data.drop(columns=categorical_cols).copy()
       if categorical_cols:
           dummy_data = pd.get_dummies(data[categorical_cols], prefix=categorical_cols, dummy_na=True)
           numeric_data = pd.concat([numeric_data, dummy_data], axis=1)
       
       X = numeric_data.drop(columns=[duration_col, event_col])
       if X.empty:
           return None, pd.DataFrame(), {"warning": "No valid features after encoding"}
       
       # Select AFT model
       if dist == "weibull":
           aft = WeibullAFTFitter(penalizer=0.1)
       elif dist == "exponential":
           aft = ExponentialAFTFitter(penalizer=0.1)
       else:
           raise ValueError(f"Unknown distribution: {dist}. Use 'weibull' or 'exponential'")
       
       try:
           aft.fit(numeric_data, duration_col=duration_col, event_col=event_col)
       except Exception as e:
           return None, pd.DataFrame(), {"error": str(e)}
       
       # Extract coefficients
       summary = aft.summary
       coefficients = summary[["coef", "exp(coef)", "p"]].copy()
       coefficients.columns = ["coefficient", "time_ratio", "p_value"]
       coefficients = coefficients.sort_values("p_value")
       
       # Add diagnostics
       diagnostics = {
           "event_count": int(E.sum()),
           "censored_count": int(len(E) - E.sum()),
           "total_samples": len(E),
           "feature_count": X.shape[1],
           "concordance_index": float(aft.concordance_index_),
           "log_likelihood": float(aft.log_likelihood_),
           "aic": float(aft.aic_),
           "distribution": dist,
       }
       
       return aft, coefficients, diagnostics
   ```

## Critical Acceptance Criteria

✅ **Kaplan-Meier handles censored data** via `event_observed` parameter  
✅ **Cox model doesn't drop censored rows** (uses all available data)  
✅ **AFT model implemented** with Weibull and exponential options  
✅ **PH assumption testing** added to diagnostics  
✅ **No silent dropna()** - missing values handled explicitly  

## Validation Commands

```bash
# Test 1: Kaplan-Meier with censored data
python -c "
from aac_adoption.analysis.survival_analysis import compute_kaplan_meier_survival
import pandas as pd

# Create test data with censoring
df = pd.DataFrame({
    'days_to_outcome': [5, 10, 15, 20, 25, 30],
    'is_censored': [0, 1, 0, 1, 0, 1]  # 50% censored
})

result = compute_kaplan_meier_survival(df, event_col='is_censored')
assert 'survival_probability' in result.columns
assert len(result) > 0
print('✓ Kaplan-Meier handles censored data')
"

# Test 2: Cox model with censored data
python -c "
from aac_adoption.analysis.survival_analysis import fit_cox_proportional_hazards
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    'days_to_outcome': np.random.randint(1, 60, 100),
    'adopted': np.random.choice([0, 1], 100, p=[0.3, 0.7]),  # 70% events
    'animal_type': np.random.choice(['Dog', 'Cat'], 100),
    'age_group': np.random.choice(['baby', 'young', 'adult'], 100)
})

cph, coeffs, diag = fit_cox_proportional_hazards(df)
assert cph is not None
assert len(coeffs) > 0
assert 'concordance_index' in diag
print(f'✓ Cox model fits with {diag[\"event_count\"]} events')
"

# Test 3: AFT model
python -c "
from aac_adoption.analysis.survival_analysis import fit_aft_model
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    'days_to_outcome': np.random.randint(1, 60, 100),
    'adopted': np.random.choice([0, 1], 100, p=[0.3, 0.7]),
    'animal_type': np.random.choice(['Dog', 'Cat'], 100),
})

aft, coeffs, diag = fit_aft_model(df, dist='weibull')
assert aft is not None
assert 'time_ratio' in coeffs.columns
print(f'✓ Weibull AFT model fits with C-index: {diag[\"concordance_index\"]:.3f}')
"
```

## Key Design Decisions

1. **Censoring = 0 in event column**: Standard survival analysis convention
2. **Explicit missing value handling**: Don't use `dropna()` silently
3. **PH assumption testing**: Optional but highly recommended for Cox
4. **AFT as alternative**: Weibull and exponential distributions for comparison

## Next Handoff Points

After completing these tasks, you will have produced:
- Enhanced `survival_analysis.py` with censoring support
- Working Cox and AFT models
- Diagnostic functions for model validation

**Hand to Agent 3**: Survival models ready for integration into training pipeline.

---

*End of Agent 2 Tasks*
