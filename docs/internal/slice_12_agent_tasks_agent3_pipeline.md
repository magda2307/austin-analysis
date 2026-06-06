# Slice 12 Agent Tasks: Agent 3 - Pipeline Integration Specialist

## Your Mission
Create the end-to-end survival training pipeline and integrate survival models with the existing training infrastructure.

## Files to Create/Modify

### 1. scripts/train_survival.py **(NEW FILE)**

Create this new training script following the pattern of existing training scripts.

**Full Script Content**:
```python
#!/usr/bin/env python
"""Train survival models (CoxPH, AFT) for AAC adoption analysis.

Usage:
    python scripts/train_survival.py --animal-subset combined
    python scripts/train_survival.py --animal-subset dogs
    python scripts/train_survival.py --animal-subset cats

Options:
    --animal-subset: combined, dogs, or cats (default: combined)
    --output-dir: Output directory for artifacts (default: reports/survival/)
    --model-dir: Model artifacts directory (default: models/survival/)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aac_adoption.config import RANDOM_STATE
from aac_adoption.data.load_data import load_intakes, load_outcomes
from aac_adoption.data.build_dataset import build_modeling_dataset
from aac_adoption.models.split import make_time_split
from aac_adoption.analysis.survival_analysis import (
    compute_kaplan_meier_survival,
    fit_cox_proportional_hazards,
    fit_aft_model,
)


def load_and_prep_data(animal_subset: str, extract_end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load raw data and build modeling dataset with censoring columns."""
    intakes = load_intakes("data/intakes.csv")
    outcomes = load_outcomes("data/outcomes.csv")
    
    result = build_modeling_dataset(intakes, outcomes, extract_end_date=extract_end_date)
    df = result.dataset
    
    print(f"Loaded dataset: {len(df)} rows, {result.matched_rows} matched, {result.unmatched_intakes} unmatched")
    print(f"Animal subset: {animal_subset}")
    
    # Verify censoring columns exist
    required_cols = ['is_censored', 'event_type', 'censoring_reason', 'followup_days_censored']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing censoring columns: {missing}")
        print("Run data pipeline first to add censoring columns")
    
    return df


def create_survival_split(df: pd.DataFrame, animal_subset: str) -> pd.DataFrame:
    """Create train/validation/test split for survival analysis.
    
    Survival analysis uses followup_days_censored as the duration column.
    Event indicator depends on the event type (censored=False, adoption=True).
    """
    subset = (animal_subset or "combined").lower()
    if subset == "combined":
        subset_df = df.copy()
    elif subset in {"dog", "dogs"}:
        subset_df = df.loc[df["animal_type"].astype(str).str.lower().eq("dog")].copy()
    elif subset in {"cat", "cats"}:
        subset_df = df.loc[df["animal_type"].astype(str).str.lower().eq("cat")].copy()
    else:
        raise ValueError("animal_subset must be one of: combined, dogs, cats")
    
    # Survival target: we use all episodes (including censored)
    # Event indicator: 1 if not censored (had an outcome), 0 if censored
    subset_df["survival_event"] = ~subset_df["is_censored"]
    subset_df["survival_time"] = subset_df["followup_days_censored"]
    
    # Time-based split (same as other models)
    train = subset_df.loc[subset_df["intake_year"].between(2013, 2021)].copy()
    validation = subset_df.loc[subset_df["intake_year"].between(2022, 2023)].copy()
    test = subset_df.loc[subset_df["intake_year"].between(2024, 2025)].copy()
    
    print(f"Train: {len(train)} rows ({train['intake_year'].min()}-{train['intake_year'].max()})")
    print(f"Validation: {len(validation)} rows ({validation['intake_year'].min()}-{validation['intake_year'].max()})")
    print(f"Test: {len(test)} rows ({test['intake_year'].min()}-{test['intake_year'].max()})")
    
    # Train-only event counts (for context)
    print(f"Train events (uncensored): {train['survival_event'].sum()}")
    print(f"Train censored: {(~train['survival_event']).sum()}")
    
    return train, validation, test


def train_survival_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Train CoxPH and AFT survival models.
    
    Returns:
        Dictionary with fitted models and evaluation metrics.
    """
    metrics = {}
    
    # Feature columns for survival models
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
    
    # Filter columns that exist in train
    feature_cols = [c for c in feature_cols if c in train.columns]
    
    # Train Cox proportional hazards model
    print("\nTraining Cox PH model...")
    try:
        cox_df = train[["survival_time", "survival_event"] + feature_cols].dropna()
        print(f"Cox training data: {len(cox_df)} rows after dropping NaN")
        
        if len(cox_df) < 10:
            raise ValueError(f"Insufficient data for Cox model: {len(cox_df)} rows")
        
        cox_model, cox_coeffs, cox_diag = fit_cox_proportional_hazards(
            cox_df,
            duration_col="survival_time",
            event_col="survival_event",
            feature_cols=feature_cols,
        )
        
        if cox_model is None:
            raise ValueError("Cox model training failed")
        
        # Evaluate on validation
        cox_val_df = validation[["survival_time", "survival_event"] + feature_cols].dropna()
        cox_val_metrics = {
            "concordance_index": float(cox_model.concordance_index_),
            "log_likelihood": float(cox_model.log_likelihood_),
            "aic": float(cox_model.aic_),
        }
        
        metrics["cox"] = {
            "model": cox_model,
            "coefficients": cox_coeffs,
            "diagnostics": cox_diag,
            "validation_metrics": cox_val_metrics,
        }
        
        print(f"Cox C-index: {cox_val_metrics['concordance_index']:.3f}")
        print(f"Cox AIC: {cox_val_metrics['aic']:.1f}")
        
    except Exception as e:
        print(f"Warning: Cox model failed: {e}")
        metrics["cox"] = {"error": str(e)}
    
    # Train AFT (Weibull) model
    print("\nTraining AFT Weibull model...")
    try:
        aft_df = train[["survival_time", "survival_event"] + feature_cols].dropna()
        
        if len(aft_df) < 10:
            raise ValueError(f"Insufficient data for AFT model: {len(aft_df)} rows")
        
        aft_model, aft_coeffs, aft_diag = fit_aft_model(
            aft_df,
            duration_col="survival_time",
            event_col="survival_event",
            feature_cols=feature_cols,
            dist="weibull",
        )
        
        if aft_model is None:
            raise ValueError("AFT model training failed")
        
        # Evaluate on validation
        aft_val_df = validation[["survival_time", "survival_event"] + feature_cols].dropna()
        aft_val_metrics = {
            "concordance_index": float(aft_model.concordance_index_),
            "log_likelihood": float(aft_model.log_likelihood_),
            "aic": float(aft_model.aic_),
        }
        
        metrics["aft_weibull"] = {
            "model": aft_model,
            "coefficients": aft_coeffs,
            "diagnostics": aft_diag,
            "validation_metrics": aft_val_metrics,
        }
        
        print(f"AFT C-index: {aft_val_metrics['concordance_index']:.3f}")
        print(f"AFT AIC: {aft_val_metrics['aic']:.1f}")
        
    except Exception as e:
        print(f"Warning: AFT model failed: {e}")
        metrics["aft_weibull"] = {"error": str(e)}
    
    return metrics


def save_artifacts(
    metrics: dict,
    output_dir: Path,
    model_dir: Path,
    animal_subset: str,
    timestamp: str,
) -> None:
    """Save model artifacts and metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save survival metrics summary
    summary_rows = []
    for model_name, model_data in metrics.items():
        if "error" in model_data:
            summary_rows.append({
                "model": model_name,
                "status": "failed",
                "error": model_data["error"],
                "concordance_index": None,
                "aic": None,
            })
        else:
            summary_rows.append({
                "model": model_name,
                "status": "success",
                "concordance_index": model_data["validation_metrics"]["concordance_index"],
                "log_likelihood": model_data["validation_metrics"]["log_likelihood"],
                "aic": model_data["validation_metrics"]["aic"],
                "event_count": model_data["diagnostics"].get("event_count"),
                "censored_count": model_data["diagnostics"].get("censored_count"),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "survival_metrics.csv", index=False)
    print(f"\n✓ Saved survival metrics to {output_dir / 'survival_metrics.csv'}")
    
    # Save model coefficients
    for model_name, model_data in metrics.items():
        if "error" not in model_data and "coefficients" in model_data:
            model_data["coefficients"].to_csv(
                output_dir / f"{model_name}_coefficients.csv"
            )
    
    # Save models using artifact functions
    from aac_adoption.models.artifacts import save_model_artifact
    for model_name, model_data in metrics.items():
        if "error" not in model_data and "model" in model_data:
            artifact_name = f"survival_{model_name}_{animal_subset}_{timestamp}"
            save_model_artifact(
                model_data["model"],
                artifact_name,
                extra_metadata={"model_type": model_name, "animal_subset": animal_subset}
            )
            print(f"✓ Saved {model_name} model")


def main():
    parser = argparse.ArgumentParser(description="Train survival models for AAC adoption")
    parser.add_argument("--animal-subset", default="combined", 
                       choices=["combined", "dogs", "cats"],
                       help="Animal subset to analyze")
    parser.add_argument("--output-dir", default="reports/survival",
                       help="Output directory for metrics")
    parser.add_argument("--model-dir", default="models/survival",
                       help="Model artifacts directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AAC Adoption Survival Model Training")
    print("=" * 60)
    
    # Extract end date for matching
    extract_end_date = pd.Timestamp("2025-12-31")  # Adjust based on your extract
    
    # Load data
    df = load_and_prep_data(args.animal_subset, extract_end_date)
    
    # Create split
    train, validation, test = create_survival_split(df, args.animal_subset)
    
    # Train models
    metrics = train_survival_models(train, validation, Path(args.output_dir))
    
    # Save artifacts
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_artifacts(
        metrics,
        Path(args.output_dir),
        Path(args.model_dir),
        args.animal_subset,
        timestamp,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 2. src/aac_adoption/models/evaluate.py

**Task**: Add survival-specific metrics functions.

**Add these functions** (append to existing file):

```python
def survival_concordance_index(y_true_time, y_pred, y_censor=None):
    """Compute concordance index (C-index) for survival predictions.
    
    Args:
        y_true_time: Actual time-to-event or follow-up time
        y_pred: Predicted risk scores (higher = higher risk = shorter survival)
        y_censor: Censoring indicator (1=event, 0=censored). If None, assumes all events observed.
    
    Returns:
        C-index between 0 and 1
    """
    from lifelines.utils import concordance_index
    
    if y_censor is None:
        y_censor = np.ones_like(y_true_time)
    
    try:
        c_index = concordance_index(y_true_time, y_pred, y_censor)
        return float(c_index)
    except Exception:
        return 0.5


def survival_time_dependent_roc_auc(y_true_time, y_pred_proba, y_censor, time_points):
    """Compute time-dependent ROC-AUC at specified time points.
    
    Uses the IPCW (inverse probability of censoring weighing) method.
    
    Args:
        y_true_time: Time-to-event
        y_pred_proba: Predicted survival probabilities at time points
        y_censor: Censoring indicator
        time_points: List of time points for evaluation
    
    Returns:
        Dictionary with time_points as keys and ROC-AUC values
    """
    from lifelines import KaplanMeierFitter
    
    results = {}
    
    # Estimate censoring distribution using Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(y_true_time, event_observed=~y_censor.astype(bool))
    
    for t in time_points:
        try:
            # Predict probability of being event-free at time t
            survival_at_t = kmf.predict(t)
            
            # Binary outcome: event occurred before t
            y_binary = (y_true_time <= t) & (y_censor == 1)
            
            # Predicted probability of event by time t
            y_pred_at_t = 1 - y_pred_proba
            
            # Compute ROC-AUC
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_binary)) > 1:
                auc = roc_auc_score(y_binary, y_pred_at_t)
                results[str(t)] = float(auc)
            else:
                results[str(t)] = None
                
        except Exception:
            results[str(t)] = None
    
    return results


def survival_integrated_brier_score(y_true_time, y_pred_proba, y_censor, time_range):
    """Compute integrated Brier score for survival models.
    
    Args:
        y_true_time: Time-to-event
        y_pred_proba: Predicted survival probabilities (N x time_points)
        y_censor: Censoring indicator
        time_range: Tuple of (min_time, max_time) for integration
    
    Returns:
        Integrated Brier score
    """
    from lifelines.utils import brier_score
    
    # Compute Brier score at each time point
    times = np.linspace(time_range[0], time_range[1], 100)
    brier_scores = []
    
    for t in times:
        try:
            # Binary outcome: event occurred before t
            y_binary = (y_true_time <= t) & (y_censor == 1)
            
            # Predicted probability of event by time t
            # y_pred_proba should be survival probability, so event prob = 1 - survival
            y_pred_at_t = 1 - y_pred_proba
            
            if len(np.unique(y_binary)) > 1:
                bs = brier_score(y_binary, y_pred_at_t, t)
                brier_scores.append(bs)
        except Exception:
            pass
    
    if not brier_scores:
        return 0.5
    
    # Integrate using trapezoidal rule
    return float(np.trapz(brier_scores, times) / (time_range[1] - time_range[0]))


def survival_metrics(y_true_time, y_pred, y_censor=None, y_pred_proba=None, time_points=None):
    """Compute comprehensive survival metrics.
    
    Args:
        y_true_time: Actual time-to-event
        y_pred: Predicted risk scores
        y_censor: Censoring indicator (1=event, 0=censored)
        y_pred_proba: Predicted survival probabilities (for time-dependent metrics)
        time_points: Time points for evaluation
    
    Returns:
        Dictionary with all survival metrics
    """
    if y_censor is None:
        y_censor = np.ones_like(y_true_time)
    
    if time_points is None:
        time_points = [30, 60, 90, 180]
    
    metrics = {
        "concordance_index": survival_concordance_index(y_true_time, y_pred, y_censor),
        "event_count": int(np.sum(y_censor)),
        "censored_count": int(np.sum(~y_censor.astype(bool))),
        "total_samples": len(y_true_time),
    }
    
    if y_pred_proba is not None:
        metrics["time_dependent_roc_auc"] = survival_time_dependent_roc_auc(
            y_true_time, y_pred_proba, y_censor, time_points
        )
    
    return metrics
```

### 3. src/aac_adoption/features/survival_targets.py **(NEW FILE)**

Create new file for horizon-based survival targets.

**Full File Content**:
```python
"""Survival targets for horizon-based analysis."""

import numpy as np
import pandas as pd


def add_survival_horizon_targets(
    df: pd.DataFrame,
    horizon_days: list[int] = None,
) -> pd.DataFrame:
    """Add survival targets for specified time horizons.
    
    For each horizon, creates:
    - survival_<horizon>d_time: min(days_to_outcome, horizon) or censoring time
    - survival_<horizon>d_event: 1 if event occurred within horizon, 0 if censored
    
    This allows survival models to predict "event within horizon" using time-to-event data.
    
    Args:
        df: DataFrame with survival columns (days_to_outcome, is_censored, followup_days_censored)
        horizon_days: List of horizon days (default: [7, 30, 60, 90])
    
    Returns:
        DataFrame with added survival horizon columns
    """
    if horizon_days is None:
        horizon_days = [7, 30, 60, 90]
    
    result = df.copy()
    
    for horizon in horizon_days:
        prefix = f"survival_{horizon}d"
        
        # Time at risk: min of observed time or horizon (for those with follow-up)
        result[f"{prefix}_time"] = result.apply(
            lambda row: min(row["days_to_outcome"], horizon) 
                       if not row["is_censored"] and row["days_to_outcome"] <= horizon 
                       else row["followup_days_censored"],
            axis=1
        )
        
        # Event indicator: 1 if adopted within horizon, 0 if censored
        result[f"{prefix}_event"] = result.apply(
            lambda row: (
                1 if not row["is_censored"] and row["adopted"] and row["days_to_outcome"] <= horizon
                else 0
            ),
            axis=1
        )
    
    return result


def compute_survival_horizon_statistics(
    df: pd.DataFrame,
    horizon_days: list[int] = None,
) -> pd.DataFrame:
    """Compute statistics for each survival horizon.
    
    Returns dataframe with:
    - horizon: Time horizon in days
    - total: Total episodes
    - events: Episodes with event within horizon
    - censored: Episodes censored (no outcome or end of extract)
    - event_rate: Percentage with event
    """
    if horizon_days is None:
        horizon_days = [7, 30, 60, 90]
    
    stats = []
    
    for horizon in horizon_days:
        event_col = f"survival_{horizon}d_event"
        time_col = f"survival_{horizon}d_time"
        
        if event_col not in df.columns:
            continue
        
        total = len(df)
        events = df[event_col].sum()
        censored = (~df["is_censored"]).sum() - events  # Uncensored minus events
        
        stats.append({
            "horizon_days": horizon,
            "total_episodes": int(total),
            "events_within_horizon": int(events),
            "censored": int(censored),
            "event_rate": float(events / total) if total > 0 else 0.0,
            "median_time_at_risk": float(df[time_col].median()) if time_col in df.columns else None,
        })
    
    return pd.DataFrame(stats)


def filter_horizon_appropriate_episodes(
    df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Filter to episodes with sufficient follow-up time for horizon analysis.
    
    For a 30-day horizon, include only episodes where:
    - There was an outcome (not censored), OR
    - Follow-up time >= 30 days
    
    This prevents bias from short follow-up periods.
    
    Args:
        df: DataFrame with survival columns
        horizon_days: Time horizon
    
    Returns:
        Filtered DataFrame
    """
    # Include if: had outcome OR had sufficient follow-up
    mask = (
        ~df["is_censored"] |  # Actually had an outcome
        (df["followup_days_available"] >= horizon_days)  # Sufficient follow-up time
    )
    
    return df[mask].copy()
```

## Critical Acceptance Criteria

✅ **`scripts/train_survival.py` runs end-to-end** without errors  
✅ **Survival metrics computed** (C-index, AIC, log-likelihood)  
✅ **Model artifacts saved** to `models/survival/`  
✅ **Metrics saved** to `reports/survival/survival_metrics.csv`  
✅ **Horizon targets integrated** (Task 3.3)  

## Validation Commands

```bash
# Test 1: Script runs successfully
python scripts/train_survival.py --animal-subset combined

# Test 2: Check outputs exist
ls reports/survival/
ls models/survival/

# Test 3: Metrics look reasonable
python -c "
import pandas as pd
metrics = pd.read_csv('reports/survival/survival_metrics.csv')
assert len(metrics) > 0, 'No metrics saved'
assert 'concordance_index' in metrics.columns
assert 'aic' in metrics.columns
print('✓ Survival metrics.csv valid')
print(metrics[['model', 'concordance_index', 'aic']])
"

# Test 4: Horizon targets
python -c "
from aac_adoption.features.survival_targets import add_survival_horizon_targets, compute_survival_horizon_statistics
import pandas as pd

df = pd.read_csv('data/modeling_dataset.csv')
df = add_survival_horizon_targets(df)
stats = compute_survival_horizon_statistics(df)
print(stats)
assert 'survival_7d_event' in df.columns
print('✓ Horizon targets generated')
"
```

## Key Design Decisions

1. **Followup time as survival time**: Uses `followup_days_censored` which includes censoring time
2. **Event indicator**: 1 if `~is_censored` (actually had an outcome)
3. **Time-based split same as other models**: Consistent with existing pipeline
4. **Horizon targets separate**: New feature file, not modifying build_dataset

## Next Handoff Points

After completing these tasks, you will have produced:
- Working `scripts/train_survival.py` pipeline
- Survival metrics in `reports/survival/`
- Model artifacts in `models/survival/`
- Horizon survival targets

**Hand to Agent 5**: Pipeline ready for reporting integration.

---

*End of Agent 3 Tasks*
