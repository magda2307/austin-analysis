"""Survival analysis utilities for LOS modeling."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

np.random.seed(42)


def compute_kaplan_meier_survival(
    df: pd.DataFrame,
    duration_col: str = "days_to_adoption",
    event_col: str = "adopted",
    time_points: Optional[list] = None,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """Compute descriptive Kaplan-Meier survival curve for time to adoption among adopted animals.
    
    This is a distribution-of-time-among-observed-adoptions view, not a censor-aware
    estimate of adoption incidence.
    
    Args:
        df: Input dataframe with survival data. Should be pre-filtered to adopted animals only.
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event indicators (always 1 for adopted subset)
        time_points: Optional list of time points at which to compute survival probabilities
        group_by: Optional column name to compute separate curves for each group
    
    Returns:
        DataFrame with survival probabilities at each time point
    """
    if df.empty:
        return pd.DataFrame()
    
    if duration_col not in df.columns or event_col not in df.columns:
        return pd.DataFrame()
    
    kmf = KaplanMeierFitter()
    
    if group_by is not None:
        return _compute_kaplan_meier_grouped(df, duration_col, event_col, time_points, group_by)
    
    T = df[duration_col]
    E = df[event_col]
    
    if T.empty or E.empty:
        return pd.DataFrame()
    
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


def _compute_kaplan_meier_grouped(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    time_points: Optional[list],
    group_by: str,
) -> pd.DataFrame:
    """Helper function to compute KM curves for grouped data."""
    result = []
    
    if time_points is None:
        time_points = []
        for group_name, group_df in df.groupby(group_by):
            T = group_df[duration_col]
            max_time = int(T.max()) + 1
            time_points.extend(range(0, max_time, 1))
        time_points = sorted(set(time_points))
    
    for group_name, group_df in df.groupby(group_by):
        T = group_df[duration_col]
        E = group_df[event_col]
        
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E, label=str(group_name))
        
        survival_probs = kmf.survival_function_at_times(time_points)
        
        for time, prob in survival_probs.items():
            at_risk = kmf.event_table.at_risk.get(int(time), len(group_df))
            result.append({
                "days": int(time),
                group_by: group_name,
                "survival_probability": prob,
                "censoring_rate": 1 - prob,
                "n_at_risk": at_risk,
            })
    
    return pd.DataFrame(result)


def compute_LOS_quantiles(
    df: pd.DataFrame,
    quantiles: list = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    duration_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """Compute LOS quantiles by animal type.
    
    Args:
        df: Input dataframe with survival data
        quantiles: List of quantiles to compute
        duration_col_override: Optional explicit duration column to use
    
    Returns:
        DataFrame with LOS quantiles by animal type
    """
    if df.empty:
        return pd.DataFrame()
    
    if "animal_type" not in df.columns:
        return pd.DataFrame()
    
    if duration_col_override is not None:
        duration_col = duration_col_override
    else:
        duration_col = "days_to_outcome"
    result = []
    for animal_type in df["animal_type"].unique():
        type_df = df[df["animal_type"] == animal_type]
        
        for q in quantiles:
            result.append({
                "animal_type": animal_type,
                "quantile": q,
                "LOS_days": type_df[duration_col].quantile(q),
                "n": len(type_df),
                "n_events": len(type_df),
            })
    
    return pd.DataFrame(result)


def compute_cumulative_hazard(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "event_observed",
    groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute cumulative hazard function.
    
    Args:
        df: Input dataframe with survival data
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event indicators (1=event occurred)
        groups: Optional list of group levels to compute separately
    
    Returns:
        DataFrame with cumulative hazard values over time
    """
    required_cols = [duration_col, event_col]
    
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    result = []
    
    if groups is not None:
        for group_name, group_df in df.groupby(groups):
            T = group_df[duration_col]
            E = group_df[event_col]
            
            if T.empty or E.empty:
                continue
            
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E.astype(int))
            
            for time in range(1, int(T.max()) + 1):
                try:
                    survival = float(kmf.predict(time))
                    hazard = -np.log(survival) if survival > 0 else float('inf')
                    n_at_risk = int(kmf.event_table.loc[:time, 'at_risk'].iloc[-1]) if time >= kmf.event_table.index.min() else len(T)
                    result.append({
                        "time": time,
                        groups: group_name,
                        "cumulative_hazard": hazard,
                        "n_at_risk": n_at_risk,
                    })
                except Exception:
                    continue
    else:
        T = df[duration_col]
        E = df[event_col]
        
        if T.empty or E.empty:
            return pd.DataFrame()
        
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E.astype(int))
        
        for time in range(1, int(T.max()) + 1):
            try:
                survival = float(kmf.predict(time))
                hazard = -np.log(survival) if survival > 0 else float('inf')
                n_at_risk = int(kmf.event_table.loc[:time, 'at_risk'].iloc[-1]) if time >= kmf.event_table.index.min() else len(T)
                result.append({
                    "time": time,
                    "cumulative_hazard": hazard,
                    "n_at_risk": n_at_risk,
                })
            except Exception:
                continue
    
    return pd.DataFrame(result)


def log_transform_LOS(
    df: pd.DataFrame, 
    target_col: str = "days_to_outcome", 
    offset: float = 1.0,
    base: str = "natural"
) -> pd.DataFrame:
    """Apply log transformation to LOS for regression.
    
    Args:
        df: Input dataframe with LOS values
        target_col: Name of column containing LOS values
        offset: Value to add before log transform (to handle zero values)
        base: Logarithm base ("natural", "10", or "2")
    
    Returns:
        DataFrame with log-transformed LOS column appended
    """
    result = df.copy()
    
    log_col = f"log_{target_col}"
    
    if base == "natural":
        result[log_col] = np.log(result[target_col].clip(lower=offset))
    elif base == "10":
        result[log_col] = np.log10(result[target_col].clip(lower=offset))
    elif base == "2":
        result[log_col] = np.log2(result[target_col].clip(lower=offset))
    else:
        result[log_col] = np.log(result[target_col].clip(lower=offset))
    
    return result
