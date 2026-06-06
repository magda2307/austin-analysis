"""Survival analysis utilities for LOS modeling."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index


def compute_kaplan_meier_survival(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
    time_points: Optional[list] = None,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival curve for LOS."""
    kmf = KaplanMeierFitter()
    
    if df.empty:
        return pd.DataFrame()
    
    T = df[duration_col]
    E = df[event_col]
    
    kmf.fit(T, event_observed=E, label="Kaplan-Meier")
    
    if time_points is None:
        time_points = list(range(0, int(T.max()) + 1, 1))
    
    survival_probs = kmf.survival_function_at_times(time_points)
    
    result = pd.DataFrame({
        "days": time_points,
        "survival_probability": survival_probs.values,
        "censoring_rate": 1 - survival_probs.values,
    })
    
    return result


def fit_cox_proportional_hazards(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
    feature_cols: Optional[list] = None,
) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """Fit Cox proportional hazards model for LOS."""
    if df.empty:
        return None, pd.DataFrame()
    
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
    data = data.dropna()
    
    if data.empty:
        return None, pd.DataFrame()
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col=duration_col, event_col=event_col)
    
    summary = cph.summary
    coefficients = summary[["coef", "exp(coef)", "p"]].copy()
    coefficients.columns = ["coefficient", "hazard_ratio", "p_value"]
    
    return cph, coefficients


def compute_concordance_index(
    df: pd.DataFrame,
    predicted_col: str = "predicted_days",
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
) -> float:
    """Compute concordance index (C-index) for survival predictions."""
    if df.empty:
        return 0.5
    
    T = df[duration_col]
    E = df[event_col]
    pred = df[predicted_col]
    
    try:
        c_index = concordance_index(T, -pred, E)
        return float(c_index)
    except Exception:
        return 0.5


def compute_LOS_quantiles(
    df: pd.DataFrame,
    quantiles: list = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
) -> pd.DataFrame:
    """Compute LOS quantiles by animal type."""
    if df.empty:
        return pd.DataFrame()
    
    result = []
    for animal_type in df["animal_type"].unique():
        type_df = df[df["animal_type"] == animal_type]
        
        for q in quantiles:
            result.append({
                "animal_type": animal_type,
                "quantile": q,
                "LOS_days": type_df["days_to_outcome"].quantile(q),
                "n": len(type_df),
            })
    
    return pd.DataFrame(result)


def add_censoring_indicators(
    df: pd.DataFrame,
    max_observation_period: int = 90,
) -> pd.DataFrame:
    """Add censoring indicators for LOS analysis."""
    result = df.copy()
    
    result["is_censored"] = result["days_to_outcome"] >= max_observation_period
    result["tracked_days"] = result["days_to_outcome"].clip(upper=max_observation_period)
    
    return result


def log_transform_LOS(df: pd.DataFrame, target_col: str = "days_to_outcome") -> pd.DataFrame:
    """Apply log transformation to LOS for regression."""
    result = df.copy()
    
    log_col = f"log_{target_col}"
    result[log_col] = np.log1p(result[target_col].clip(lower=0))
    
    return result
