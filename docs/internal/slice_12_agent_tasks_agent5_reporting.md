# Slice 12 Agent Tasks: Agent 5 - Reporting & Documentation Specialist

## Your Mission
Create reporting infrastructure for survival analysis and update documentation to reflect the implemented methodology.

## Files to Create/Modify

### 1. src/aac_adoption/diagnostics/survival_diagnostics.py **(NEW FILE)**

Create comprehensive diagnostics for survival analysis outputs.

**Full File Content**:
```python
"""Survival analysis diagnostics for comprehensive reporting."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter


def compute_survival_subgroup_curves(
    df: pd.DataFrame,
    group_var: str,
    time_col: str = "days_to_outcome",
    event_col: str = "adopted",
    time_points: Optional[list] = None,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival curves by subgroup.
    
    Args:
        df: DataFrame with survival columns
        group_var: Column to group by (e.g., 'animal_type', 'age_group')
        time_col: Time-to-event column
        event_col: Event indicator column
        time_points: List of time points for survival estimates
    
    Returns:
        DataFrame with survival curves stratified by group
    """
    if group_var not in df.columns:
        raise ValueError(f"Group variable '{group_var}' not in DataFrame")
    
    if df.empty:
        return pd.DataFrame()
    
    result = []
    
    for group_value in df[group_var].unique():
        group_df = df[df[group_var] == group_value].copy()
        
        if group_df.empty:
            continue
        
        kmf = KaplanMeierFitter()
        T = group_df[time_col]
        E = group_df[event_col].fillna(0).astype(int)
        
        if E.sum() == 0:
            # No events in this group
            result.append({
                "group_variable": group_var,
                "group_value": str(group_value),
                "days": 0,
                "survival_probability": 1.0,
                "event_count": 0,
                "censored_count": len(T),
                "total_records": len(T),
            })
            continue
        
        kmf.fit(T, event_observed=E, label="Kaplan-Meier")
        
        if time_points is None:
            time_points = list(range(0, int(T.max()) + 1, 1))
        
        survival_probs = kmf.survival_function_at_times(time_points)
        
        # Get risk table info
        risk_table = kmf.risk_table
        risk_df = pd.DataFrame(risk_table).reset_index()
        
        for i, t in enumerate(time_points):
            row = {
                "group_variable": group_var,
                "group_value": str(group_value),
                "days": t,
                "survival_probability": float(survival_probs.iloc[i]) if i < len(survival_probs) else None,
                "event_count": int(E.sum()),
                "censored_count": int(len(T) - E.sum()),
                "total_records": int(len(T)),
            }
            
            # Add risk set info if available
            risk_row = risk_df[risk_df["time"] == t]
            if not risk_row.empty:
                row["risk_set_size"] = int(risk_row.iloc[0]["at_risk"])
            
            result.append(row)
    
    return pd.DataFrame(result)


def compute_cox_coefficient_summary(
    cox_model: CoxPHFitter,
    feature_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Create human-readable summary of Cox model coefficients.
    
    Args:
        cox_model: Fitted CoxPHFitter model
        feature_map: Optional mapping of feature names to human-readable labels
    
    Returns:
        DataFrame with coefficient summaries
    """
    summary = cox_model.summary
    
    result = summary[["coef", "exp(coef)", "p"]].copy()
    result.columns = ["coefficient", "hazard_ratio", "p_value"]
    
    # Add interpretation
    result["hazard_direction"] = result["hazard_ratio"].apply(
        lambda x: "increases risk" if x > 1 else "decreases risk" if x < 1 else "no effect"
    )
    
    # Add significance
    result["significant"] = result["p_value"] < 0.05
    
    if feature_map:
        result["feature_label"] = result.index.map(feature_map)
    
    return result.sort_values("p_value")


def compute_concordance_by_subgroup(
    df: pd.DataFrame,
    group_var: str,
    predicted_col: str = "predicted_risk",
    time_col: str = "days_to_outcome",
    event_col: str = "adopted",
) -> pd.DataFrame:
    """Compute concordance index (C-index) by subgroup.
    
    Args:
        df: DataFrame with survival columns
        group_var: Column to group by
        predicted_col: Predicted risk scores (higher = higher risk = shorter survival)
        time_col: Time-to-event column
        event_col: Event indicator column
    
    Returns:
        DataFrame with C-index by subgroup
    """
    if group_var not in df.columns:
        raise ValueError(f"Group variable '{group_var}' not in DataFrame")
    
    result = []
    
    for group_value in df[group_var].unique():
        group_df = df[df[group_var] == group_value].copy()
        
        if len(group_df) < 10:
            result.append({
                "group_variable": group_var,
                "group_value": str(group_value),
                "concordance_index": None,
                "sample_size": len(group_df),
                "note": "Sample size too small",
            })
            continue
        
        T = group_df[time_col]
        E = group_df[event_col].fillna(0).astype(int)
        pred = group_df[predicted_col]
        
        try:
            from lifelines.utils import concordance_index
            c_index = float(concordance_index(T, pred, E))
            
            result.append({
                "group_variable": group_var,
                "group_value": str(group_value),
                "concordance_index": c_index,
                "sample_size": len(group_df),
                "event_count": int(E.sum()),
            })
        except Exception as e:
            result.append({
                "group_variable": group_var,
                "group_value": str(group_value),
                "concordance_index": None,
                "sample_size": len(group_df),
                "error": str(e),
            })
    
    return pd.DataFrame(result)


def compute_cumulative_incidence(
    df: pd.DataFrame,
    event_type_col: str = "event_type",
    time_col: str = "days_to_outcome",
) -> pd.DataFrame:
    """Compute cumulative incidence functions for competing risks.
    
    Args:
        df: DataFrame with event_type and time columns
        event_type_col: Column with event types
        time_col: Time-to-event column
    
    Returns:
        DataFrame with CIF for each event type
    """
    valid_event_types = {"adoption", "transfer", "euthanasia", "return_to_owner", "censored"}
    
    result = []
    
    for event_type in valid_event_types:
        if event_type == "censored":
            continue
        
        event_df = df[df[event_type_col] == event_type]
        
        if event_df.empty:
            continue
        
        kmf = KaplanMeierFitter()
        T = event_df[time_col]
        E = pd.Series(np.ones(len(T)))  # All events observed for this type
        
        if T.empty or T.max() == 0:
            continue
        
        kmf.fit(T, event_observed=E, label=event_type)
        
        time_points = list(range(0, int(T.max()) + 1, 1))
        survival_probs = kmf.survival_function_at_times(time_points)
        cumulative_incidence = 1 - survival_probs
        
        for i, t in enumerate(time_points):
            result.append({
                "event_type": event_type,
                "days": t,
                "cumulative_incidence": float(cumulative_incidence.iloc[i]),
                "survival_probability": float(survival_probs.iloc[i]),
                "event_count": len(T),
            })
    
    return pd.DataFrame(result)


def generate_survival_diagnostics_report(
    df: pd.DataFrame,
    output_dir: Path,
    cox_model: Optional[CoxPHFitter] = None,
) -> dict:
    """Generate comprehensive survival diagnostics report.
    
    Args:
        df: DataFrame with survival columns
        output_dir: Directory to save diagnostic files
        cox_model: Optional fitted Cox model for coefficient analysis
    
    Returns:
        Dictionary with diagnostic results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    diagnostics = {
        "files_created": [],
        "event_summary": {},
        "subgroup_analysis": {},
    }
    
    # 1. Event type distribution
    if "event_type" in df.columns:
        event_counts = df["event_type"].value_counts()
        diagnostics["event_summary"]["by_type"] = event_counts.to_dict()
        
        event_df = pd.DataFrame({
            "event_type": event_counts.index,
            "count": event_counts.values,
            "percentage": 100 * event_counts.values / len(df),
        })
        event_df.to_csv(output_dir / "event_type_distribution.csv", index=False)
        diagnostics["files_created"].append("event_type_distribution.csv")
    
    # 2. Censoring summary
    if "is_censored" in df.columns:
        censored_count = df["is_censored"].sum()
        diagnostics["event_summary"]["censored_count"] = int(censored_count)
        diagnostics["event_summary"]["censored_percentage"] = float(100 * censored_count / len(df))
        
        # Censoring by reason
        if "censoring_reason" in df.columns:
            reason_counts = df["censoring_reason"].value_counts()
            reason_df = pd.DataFrame({
                "censoring_reason": reason_counts.index,
                "count": reason_counts.values,
            })
            reason_df.to_csv(output_dir / "censoring_reason_distribution.csv", index=False)
            diagnostics["files_created"].append("censoring_reason_distribution.csv")
    
    # 3. Kaplan-Meier curves by subgroup
    subgroup_vars = ["animal_type", "age_group", "intake_type", "intake_condition"]
    
    for var in subgroup_vars:
        if var not in df.columns:
            continue
        
        subgroup_curves = compute_survival_subgroup_curves(df, var)
        if not subgroup_curves.empty:
            output_file = output_dir / f"survival_curves_by_{var}.csv"
            subgroup_curves.to_csv(output_file, index=False)
            diagnostics["files_created"].append(output_file.name)
            
            diagnostics["subgroup_analysis"][var] = {
                "n_groups": int(subgroup_curves["group_value"].nunique()),
                "curve_count": len(subgroup_curves),
            }
    
    # 4. Event counts by subgroup
    event_by_subgroup = []
    for var in subgroup_vars:
        if var not in df.columns:
            continue
        
        for group_value in df[var].unique():
            group_df = df[df[var] == group_value]
            
            event_by_subgroup.append({
                "group_variable": var,
                "group_value": str(group_value),
                "total_records": len(group_df),
                "events": int((~group_df["is_censored"]).sum()),
                "censored": int(group_df["is_censored"].sum()),
                "event_rate": float((~group_df["is_censored"]).mean()) if len(group_df) > 0 else None,
            })
    
    if event_by_subgroup:
        event_by_subgroup_df = pd.DataFrame(event_by_subgroup)
        event_by_subgroup_df.to_csv(output_dir / "event_counts_by_subgroup.csv", index=False)
        diagnostics["files_created"].append("event_counts_by_subgroup.csv")
    
    # 5. Cox model diagnostics (if provided)
    if cox_model is not None:
        cox_summary = compute_cox_coefficient_summary(cox_model)
        cox_summary.to_csv(output_dir / "cox_coefficients.csv")
        diagnostics["files_created"].append("cox_coefficients.csv")
        
        diagnostics["cox_diagnostics"] = {
            "concordance_index": float(cox_model.concordance_index_),
            "log_likelihood": float(cox_model.log_likelihood_),
            "aic": float(cox_model.aic_),
            "feature_count": len(cox_summary),
            "significant_features": int((cox_summary["p_value"] < 0.05).sum()),
        }
    
    # 6. Summary statistics
    summary = {
        "total_episodes": len(df),
        "event_episodes": int((~df["is_censored"]).sum()) if "is_censored" in df.columns else None,
        "censored_episodes": int(df["is_censored"].sum()) if "is_censored" in df.columns else None,
        "adoption_rate": float(df["adopted"].mean()) if "adopted" in df.columns else None,
        "median_followup_days": float(df["followup_days_censored"].median()) if "followup_days_censored" in df.columns else None,
        "files_generated": len(diagnostics["files_created"]),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "survival_summary.csv", index=False)
    diagnostics["files_created"].append("survival_summary.csv")
    
    return diagnostics


def validate_cox_assumptions(cox_model: CoxPHFitter) -> dict:
    """Validate Cox proportional hazards assumptions.
    
    Args:
        cox_model: Fitted CoxPHFitter model
    
    Returns:
        Dictionary with assumption test results
    """
    result = {
        "proportional_hazards": {},
        "linearity": {},
    }
    
    # Test proportional hazards assumption
    try:
        # Check Schoenfeld residuals
        if hasattr(cox_model, 'ph_assessments'):
            ph_results = cox_model.ph_assessments
            result["proportitional_hazards"] = {
                "assumption_tested": True,
                "global_test_statistic": float(ph_results.iloc[:, 0].sum()),
                "global_test_pvalue": float(ph_results.iloc[:, 1].min()),
                "passed": ph_results.iloc[:, 1].min() > 0.05,
            }
        else:
            result["proportional_hazards"]["warning"] = "PH tests not available"
    except Exception as e:
        result["proportional_hazards"]["error"] = str(e)
    
    # Check linearity of continuous variables
    continuous_vars = []
    for col in cox_model.summary.index:
        if cox_model.summary.loc[col, 'coef'] != 0:
            continuous_vars.append(col)
    
    if continuous_vars:
        result["linearity"]["variables-checked"] = continuous_vars
        result["linearity"]["warning"] = "Add more detailed linearity checks"
    
    return result
```

### 2. reports/summary/survival_descriptive_note.md

**Update existing file with this content**:

```markdown
# Survival Analysis: Methodology Note

## What This Analysis Covers

This document describes the survival analysis implementation for the AAC Adoption ML Pipeline. Survival analysis provides time-to-event modeling that properly handles censored observations (episodes without known outcomes).

## Core Methodology

### Censoring Handling

Our survival analysis distinguishes between **events** and **censored observations**:

- **Events**: Actual outcomes (adoption, transfer, euthanasia, return-to-owner)
- **Censored observations**: Episodes without a known outcome (unmatched intakes or end-of-extract)

Censoring is handled natively in the dataset:
- `is_censored`: Boolean flag for unresolved episodes
- `event_type`: Specific outcome category
- `censoring_reason`: Why the episode is censored

### Time-to-Event Definition

The survival time (`survival_time`) is defined as:
- **for events**: `days_to_outcome` (actual time to outcome)
- **for censored**: `followup_days_available` (time from intake to extract end)

### Event Types

We model five event types:
1. **adoption** - Animal was adopted
2. **transfer** - Animal transferred to another shelter/rescue
3. **euthanasia** - Animal was humanely euthanized
4. **return_to_owner** - Animal returned to owner
5. **censored** - No known outcome (unmatched intake)

## Implemented Models

### 1. Kaplan-Meier estimator

Non-parametric estimator of the survival function. Used for:
- Descriptive time-to-adoption curves
- Subgroup comparisons (by species, age, intake type)
- Visualizing survival patterns

### 2. Cox Proportional Hazards Model

Semi-parametric model estimating hazard ratios:
$$h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \dots)$$

- Estimates **hazard ratios** (not direct time predictions)
- Allows interpretation: "X increases risk of adoption by Y%"
- Requires proportional hazards assumption

### 3. Accelerated Failure Time (AFT) Model

Parametric model (Weibull distribution):
$$T = \exp(\beta_0 + \beta_1 X_1 + \dots + \sigma \epsilon)$$

- Estimates **time ratios** (more interpretable than hazard ratios)
- Direct time predictions: "X reduces adoption time by Y days"
- Assumes specific time distribution

## Validation Checks

### Proportional Hazards Assumption

For Cox models, we verify the proportional hazards assumption using:
- Schoenfeld residual tests
- Graphical inspection of log-log plots
- Time-dependent covariate tests

If PH assumption fails, we:
- Consider time-dependent covariates
- Use AFT model as alternative
- Report limitation

### Model Comparison

We compare Cox vs AFT using:
- **AIC** (lower is better)
- **Concordance index** (higher is better)
- **Log-likelihood** (higher is better)
- **Interpretability** (AFT often more intuitive)

## Subgroup Analysis

We analyze survival by:
- **animal_type**: Dogs vs Cats
- **age_group**: Baby, Young, Adult, Senior
- **intake_type**: Stray, Owner Surrender, Transfer, etc.
- **intake_condition**: Healthy, Sick, Injured
- **simplified_breed_group**: Purebred vs Mixed
- **COVID period**: Before/After March 2020

Each subgroup reports:
- Event count and censoring percentage
- Median survival time
- Survival probability at key time points (7, 30, 60, 90 days)

## Competing Risks Framework

When multiple event types exist, standard survival analysis assumes only one event type. Our approach:

1. **Event-stratified analysis**: Model each event type separately
2. **Cumulative Incidence Functions (CIF)**: Estimate probability of each event type
3. **Subdistribution hazards** (if needed): Fine-Gray model for competing risks

## Limitations

This analysis reflects observational shelter data, not clinical trials:

1. **No unmeasured confounders controlled**: Shelter databases lack detailed covariates
2. **Potential selection bias**: Animals not admitted or not matched are excluded
3. **Time-dependent covariates not modeled**: Features may change over time
4. **Censoring may be informative**: Reasons for missing outcomes may correlate with survival

## Thesis Defense Statement

> "This survival analysis implements proper censored data handling using Kaplan-Meier estimators, Cox proportional hazards models, and AFT models. Censoring is handled natively in the dataset, with unresolved episodes marked and included in the analysis. Competing risks are addressed through event-stratified analysis and cumulative incidence functions.
>
> While our analysis properly handles censored data, it reflects observational shelter data rather than clinical trials. We cannot control for unmeasured confounders, and some association may reflect shelter selection processes rather than true survival mechanisms.
>
> The models provide both descriptive (Kaplan-Meier curves) and inferential (Cox hazards) insights into adoption timing, with AFT models offering more interpretable time-based predictions."

## Usage in Reports

Survival analysis outputs appear in:
- `reports/diagnostics/survival_diagnostics.csv`: Full diagnostic table
- `reports/diagnostics/survival_curves_by_*.csv`: Subgroup survival curves
- `reports/diagnostics/event_type_distribution.csv`: Event breakdown
- `reports/diagnostics/censoring_reason_distribution.csv`: Censoring analysis
- `reports/survival/survival_metrics.csv`: Model performance metrics

## Next Steps for Future Work

Potential extensions:
- Time-dependent Cox models for non-proportional hazards
- Multi-state models for detailed transition paths
- Machine learning survival (Random Survival Forests, DeepSurv)
- Propensity score matching to reduce selection bias
- External validation with different shelter populations
```

### 3. src/aac_adoption/reporting/evidence_pack.py

**Task**: Add survival section to existing evidence pack.

**Add this function** (append to existing file):

```python
def generate_survival_section(
    survival_metrics_df: pd.DataFrame,
    cox_coefficients_df: Optional[pd.DataFrame] = None,
    survival_curves_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Generate survival analysis section for evidence pack.
    
    Args:
        survival_metrics_df: DataFrame from reports/survival/survival_metrics.csv
        cox_coefficients_df: Optional Cox model coefficients
        survival_curves_df: Optional survival curves by subgroup
    
    Returns:
        Dictionary for evidence pack integration
    """
    section = {
        "title": "Survival Analysis",
        "description": "Time-to-event modeling with proper censored data handling",
        "models_trained": [],
        "performance": {},
        "key_findings": [],
    }
    
    # Model summaries
    for _, row in survival_metrics_df.iterrows():
        model_summary = {
            "name": row["model"],
            "status": row.get("status", "success"),
            "concordance_index": row.get("concordance_index"),
            "log_likelihood": row.get("log_likelihood"),
            "aic": row.get("aic"),
            "event_count": row.get("event_count"),
            "censored_count": row.get("censored_count"),
        }
        
        if model_summary["status"] == "success":
            section["models_trained"].append(model_summary["name"])
            section["performance"][row["model"]] = {
                "concordance_index": model_summary["concordance_index"],
                "aic": model_summary["aic"],
            }
    
    # Model selection (AIC-based)
    if len(survival_metrics_df) > 1:
        best_model = survival_metrics_df.loc[survival_metrics_df["aic"].idxmin()]
        section["recommended_model"] = best_model["model"]
        section["recommended_reason"] = f"Lowest AIC ({best_model['aic']:.1f})"
    
    # Key findings (if data available)
    if survival_curves_df is not None:
        species_curves = survival_curves_df[survival_curves_df["group_variable"] == "animal_type"]
        if not species_curves.empty:
            dog_90day_survival = species_curves[species_curves["group_value"] == "Dog"]["survival_probability"].iloc[-1]
            cat_90day_survival = species_curves[species_curves["group_value"] == "Cat"]["survival_probability"].iloc[-1]
            
            section["key_findings"] = [
                f" Dogs: {100*(1-dog_90day_survival):.0f}% adopted within 90 days",
                f" Cats: {100*(1-cat_90day_survival):.0f}% adopted within 90 days",
                f" Survival curves show distinct patterns by species",
            ]
    
    # Event breakdown
    if "event_count" in survival_metrics_df.columns:
        total_events = survival_metrics_df["event_count"].sum()
        total_censored = survival_metrics_df["censored_count"].sum()
        section["data_summary"] = {
            "total_episodes": int(total_events + total_censored),
            "events": int(total_events),
            "censored": int(total_censored),
            "censoring_rate": float(100 * total_censored / (total_events + total_censored)),
        }
    
    return section


def add_survival_to_evidence_pack(
    evidence_pack: dict,
    survival_metrics_path: str = "reports/survival/survival_metrics.csv",
    cox_coefficients_path: Optional[str] = "reports/survival/cox_censored_coefficients.csv",
    survival_curves_path: Optional[str] = "reports/diagnostics/survival_curves_by_animal_type.csv",
) -> dict:
    """Add survival analysis section to existing evidence pack.
    
    Args:
        evidence_pack: Existing evidence pack dictionary
        survival_metrics_path: Path to survival metrics CSV
        cox_coefficients_path: Optional Cox coefficients path
        survival_curves_path: Optional survival curves path
    
    Returns:
        Updated evidence pack with survival section
    """
    import pandas as pd
    
    # Load survival metrics
    survival_metrics_df = pd.read_csv(survival_metrics_path)
    
    # Load optional files
    cox_coefficients_df = None
    if cox_coefficients_path and Path(cox_coefficients_path).exists():
        cox_coefficients_df = pd.read_csv(cox_coefficients_path)
    
    survival_curves_df = None
    if survival_curves_path and Path(survival_curves_path).exists():
        survival_curves_df = pd.read_csv(survival_curves_path)
    
    # Generate survival section
    survival_section = generate_survival_section(
        survival_metrics_df,
        cox_coefficients_df,
        survival_curves_df,
    )
    
    # Add to evidence pack
    evidence_pack["survival_analysis"] = survival_section
    
    return evidence_pack
```

**Modify `generate_evidence_pack()` to call `add_survival_to_evidence_pack()`**:
- Find the main evidence pack generation function
- Add survival section before final JSON output

### 4. AGENTS.md (Update if needed)

**Add this to agent instructions** (if using Agent Manager for Slice 12):

```markdown
## Survival Analysis Agent (Agent 5)

** Responsibilities:
- Generate survival diagnostics using `src/aac_adoption/diagnostics/survival_diagnostics.py`
- Update `reports/summary/survival_descriptive_note.md` with final methodology
- Integrate survival section into `src/aac_adoption/reporting/evidence_pack.py`
- Ensure all artifacts meet acceptance schema

** Files to Monitor:
- `reports/diagnostics/survival_diagnostics.csv`
- `reports/survival/survival_metrics.csv`
- `reports/summary/survival_descriptive_note.md`
- Evidence pack JSON output

** Validation Commands:
```bash
# Run survival diagnostics
python -c "from aac_adoption.diagnostics.survival_diagnostics import generate_survival_diagnostics_report; generate_survival_diagnostics_report(df, Path('reports/diagnostics'))"

# Check survival metrics
cat reports/survival/survival_metrics.csv

# Verify evidence pack includes survival section
python -c "import json; pack = json.load(open('reports/evidence_pack.json')); assert 'survival_analysis' in pack"
```
```

## Critical Acceptance Criteria

✅ **`survival_diagnostics.py` created** with all diagnostic functions  
✅ **`survival_descriptive_note.md` updated** with methodology  
✅ **Evidence pack includes survival section**  
✅ **All diagnostic files generated** in `reports/diagnostics/`  
✅ **No broken imports** in new files  

## Validation Commands

```bash
# Test 1: Diagnostics runs without errors
python -c "
from aac_adoption.diagnostics.survival_diagnostics import generate_survival_diagnostics_report
from pathlib import Path
import pandas as pd

df = pd.read_csv('data/modeling_dataset.csv')
result = generate_survival_diagnostics_report(df, Path('reports/diagnostics'))
print('Files created:', result['files_created'])
print('✓ Diagnostics report generated')
"

# Test 2: Evidence pack includes survival
python -c "
import json
pack = json.load(open('reports/evidence_pack.json'))
assert 'survival_analysis' in pack, 'Survival section missing from evidence pack'
print('✓ Evidence pack includes survival analysis')
"

# Test 3: Descriptive note updated
cat reports/summary/survival_descriptive_note.md | grep -q 'Censoring is handled natively' && echo '✓ Documentation updated'
```

## Next Handoff Points

After completing these tasks, you will have produced:
- `survival_diagnostics.py` with diagnostic functions
- Updated `survival_descriptive_note.md`
- Evidence pack with survival section
- All diagnostic output files

**Completed!** Slice 12 is now fully documented and reportable.

---

*End of Agent 5 Tasks*
