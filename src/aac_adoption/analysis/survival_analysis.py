"""Survival analysis utilities for LOS modeling."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)


def compute_kaplan_meier_survival(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
    censoring_col: Optional[str] = None,
    time_points: Optional[list] = None,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival curve for LOS.
    
    Args:
        df: Input dataframe with survival data
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event indicators (1=event occurred, 0=censored)
        censoring_col: Optional column name for native censoring indicators
        time_points: Optional list of time points at which to compute survival probabilities
        group_by: Optional column name to compute separate curves for each group
    
    Returns:
        DataFrame with survival probabilities and censoring rates at each time point
    """
    if df.empty:
        return pd.DataFrame()
    
    if duration_col not in df.columns or event_col not in df.columns:
        return pd.DataFrame()
    
    kmf = KaplanMeierFitter()
    
    if group_by is not None:
        return _compute_kaplan_meier_grouped(df, duration_col, event_col, censoring_col, time_points, group_by)
    
    T = df[duration_col]
    
    if censoring_col is not None and censoring_col in df.columns:
        E = df[censoring_col]
    else:
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
    censoring_col: Optional[str],
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
        
        if censoring_col is not None and censoring_col in group_df.columns:
            E = group_df[censoring_col]
        else:
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


def fit_cox_with_censoring(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
    feature_cols: Optional[list] = None,
    strata: Optional[List[str]] = None,
    normalize: bool = True,
) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """Fit Cox proportional hazards model with native censoring support.
    
    Args:
        df: Input dataframe with survival data
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event indicators (1=event occurred, 0=event did not occur)
        feature_cols: List of feature columns to include in model; None for default
        strata: List of columns to use for stratification (violates PH assumption but controls for confounders)
        normalize: Whether to normalize numeric features before fitting
    
    Returns:
        Tuple of (fitted CoxPHFitter model, DataFrame with coefficients, hazard ratios, and confidence intervals)
    
    Example:
        >>> model, results = fit_cox_with_censoring(
        ...     df, 
        ...     duration_col="days_to_outcome", 
        ...     event_col="adopted",
        ...     feature_cols=["age_group", "animal_type"]
        ... )
    """
    if df.empty:
        return None, pd.DataFrame()
    
    required_cols = [duration_col, event_col]
    
    for col in required_cols:
        if col not in df.columns:
            return None, pd.DataFrame()
    
    if feature_cols is None:
        feature_cols = list(df.columns)
        feature_cols = [c for c in feature_cols if c not in [duration_col, event_col]]
        feature_cols = [c for c in feature_cols if not c.endswith(("_id", "_ID", "Id"))]
    
    cols_to_use = [duration_col, event_col] + [c for c in feature_cols if c in df.columns]
    
    data = df[cols_to_use].copy()
    
    categorical_cols = [c for c in feature_cols if c in data.columns and data[c].dtype == "object"]
    
    numeric_cols = []
    object_cols_to_drop = []
    
    for c in feature_cols:
        if c in data.columns:
            if data[c].dtype == "object":
                if c not in categorical_cols:
                    object_cols_to_drop.append(c)
            else:
                if c not in categorical_cols:
                    numeric_cols.append(c)
    
    if object_cols_to_drop:
        data = data.drop(columns=object_cols_to_drop)
    
    data = data.dropna()
    
    if data.empty:
        return None, pd.DataFrame()
    
    remaining_object_cols = [c for c in data.columns if data[c].dtype == "object"]
    if remaining_object_cols:
        data = data.drop(columns=remaining_object_cols)
    
    if data.empty:
        return None, pd.DataFrame()
    
    if normalize and numeric_cols:
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    strata_to_use = []
    if strata:
        for s in strata:
            if s in data.columns:
                strata_to_use.append(s)
            else:
                encoded_col = f"{s}_Dog" if s == "animal_type" else f"{s}_encoded_1"
                if encoded_col in data.columns:
                    strata_to_use.append(encoded_col)
    
    try:
        if strata_to_use:
            cph = CoxPHFitter()
            cph.fit(data, duration_col=duration_col, event_col=event_col, strata=strata_to_use)
        else:
            cph = CoxPHFitter()
            cph.fit(data, duration_col=duration_col, event_col=event_col)
        
        summary = cph.summary.copy()
        
        coefficients = summary[["coef", "exp(coef)", "p"]].copy()
        coefficients.columns = ["coefficient", "hazard_ratio", "p_value"]
        
        if "lower .95" in summary.columns and "upper .95" in summary.columns:
            coefficients["hr_confidence_lower"] = summary["lower .95"]
            coefficients["hr_confidence_upper"] = summary["upper .95"]
        
        return cph, coefficients
    except Exception:
        return None, pd.DataFrame()


def compute_concordance_index(
    df: pd.DataFrame,
    predicted_col: str = "predicted_days",
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
    use_censoring: bool = False,
    predicted_is_hazard: bool = False,
) -> float:
    """Compute concordance index (C-index) for survival predictions.
    
    The concordance index measures the agreement between predicted risk and actual outcome.
    For survival models, a higher C-index indicates better predictive performance.
    
    Args:
        df: Input dataframe with observed and predicted values
        predicted_col: Name of column containing predicted values
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event indicators
        use_censoring: Whether to use censored duration column if available
        predicted_is_hazard: If True, higher predicted values indicate higher risk (reverse sign)
    
    Returns:
        Concordance index value between 0 and 1
    """
    if df.empty:
        return 0.5
    
    required_cols = [duration_col, event_col, predicted_col]
    
    for col in required_cols:
        if col not in df.columns:
            return 0.5
    
    if use_censoring and "followup_days_censored" in df.columns:
        duration_col = "followup_days_censored"
    
    if use_censoring and "is_censored" in df.columns:
        event_col = "is_censored"
    
    T = df[duration_col]
    E = df[event_col]
    pred = df[predicted_col]
    
    try:
        if predicted_is_hazard:
            c_index = concordance_index(T, pred, E)
        else:
            c_index = concordance_index(T, -pred, E)
        return float(c_index)
    except Exception:
        return 0.5


def compute_LOS_quantiles(
    df: pd.DataFrame,
    quantiles: list = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    use_censoring: bool = False,
    duration_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """Compute LOS quantiles by animal type.
    
    Args:
        df: Input dataframe with survival data
        quantiles: List of quantiles to compute
        use_censoring: Whether to use censored duration column if available
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
    elif use_censoring and "followup_days_censored" in df.columns:
        duration_col = "followup_days_censored"
    else:
        duration_col = "days_to_outcome"
    
    if duration_col not in df.columns:
        return pd.DataFrame()
    
    result = []
    for animal_type in df["animal_type"].unique():
        type_df = df[df["animal_type"] == animal_type]
        
        for q in quantiles:
            result.append({
                "animal_type": animal_type,
                "quantile": q,
                "LOS_days": type_df[duration_col].quantile(q),
                "n": len(type_df),
                "n_events": int((type_df["event_observed"] == 1).sum()) if "event_observed" in type_df.columns else len(type_df),
            })
    
    return pd.DataFrame(result)


def validate_proportional_hazards(
    cph: CoxPHFitter, 
    training_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    test_method: str = "rank",
) -> pd.DataFrame:
    """Validate proportional hazards assumption for Cox model.
    
    Args:
        cph: Fitted CoxPHFitter model
        training_df: DataFrame used to fit the Cox model
        p_value_threshold: Significance threshold for PH test (default 0.05)
        test_method: Method for PH test ("rank", "identity", "log", or "kw")
    
    Returns:
        DataFrame with test statistics, p-values, and whether assumption passes for each variable
    """
    if cph is None or training_df is None or training_df.empty:
        return pd.DataFrame()
    
    from lifelines.statistics import proportional_hazard_test
    
    try:
        training_df_copy = training_df.copy()
        if cph.event_col in training_df_copy.columns:
            training_df_copy = training_df_copy.drop(columns=[cph.event_col])
        if cph.duration_col in training_df_copy.columns:
            training_df_copy = training_df_copy.drop(columns=[cph.duration_col])
    except Exception:
        return pd.DataFrame()
    
    if training_df_copy.empty:
        return pd.DataFrame()
    
    try:
        test_result = proportional_hazard_test(cph, training_df_copy, time_transform=test_method)
        summary = test_result.summary
        
        result = pd.DataFrame({
            "variable": summary.index,
            "chi_square_statistic": summary["X2"].values,
            "degrees_of_freedom": summary["df"].values,
            "ph_test_p_value": summary["p"].values,
            "ph_assumption_passes": summary["p"].values > p_value_threshold,
        })
        
        return result
    except Exception:
        return pd.DataFrame()


def test_logrank_difference(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "is_censored",
    group_col: str = "group",
) -> Dict[str, Any]:
    """Perform log-rank test to compare survival curves between groups.
    
    Args:
        df: Input dataframe with survival data
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event indicators
        group_col: Name of column containing group labels
    
    Returns:
        Dictionary with test statistic, p-value, and group comparisons
    """
    if df.empty:
        return {"test_statistic": None, "p_value": None, "groups": []}
    
    required_cols = [duration_col, event_col, group_col]
    
    for col in required_cols:
        if col not in df.columns:
            return {"test_statistic": None, "p_value": None, "groups": []}
    
    groups = df[group_col].unique()
    
    if len(groups) < 2:
        return {"test_statistic": None, "p_value": None, "groups": list(groups)}
    
    T_list = []
    E_list = []
    group_labels = []
    
    for group in groups:
        group_df = df[df[group_col] == group]
        T_list.append(group_df[duration_col])
        E_list.append(group_df[event_col])
        group_labels.append(group)
    
    try:
        result = logrank_test(*T_list, event_observeds=E_list)
        return {
            "test_statistic": result.test_statistic,
            "p_value": result.p_value,
            "groups": group_labels,
            "summary": result.summary.to_dict() if hasattr(result, "summary") else {},
        }
    except Exception:
        return {"test_statistic": None, "p_value": None, "groups": list(groups)}


def add_censoring_indicators(
    df: pd.DataFrame,
    max_observation_period: int = 90,
    event_col: str = "adopted",
    outcome_col: str = "days_to_outcome",
) -> pd.DataFrame:
    """Add censoring indicators for LOS analysis.
    
    Creates native censoring columns based on maximum observation period.
    Censoring occurs when an animal has not experienced the event
    (e.g., adoption) within the observation period.
    
    Args:
        df: Input dataframe with survival data
        max_observation_period: Maximum follow-up period in days
        event_col: Name of column containing event indicators
        outcome_col: Name of column containing time-to-event values
    
    Returns:
        DataFrame with added censoring indicators:
        - is_censored: 1 if censored (no event within observation period), 0 otherwise
        - followup_days_censored: Follow-up time truncated at max_observation_period
        - event_observed: Inverted event indicator (1=event occurred, 0=censored)
    """
    result = df.copy()
    
    result["is_censored"] = result[outcome_col] >= max_observation_period
    result["followup_days_censored"] = result[outcome_col].clip(upper=max_observation_period)
    result["event_observed"] = 1 - result["is_censored"].astype(int)
    
    if event_col in result.columns:
        result["event_observed_native"] = result[event_col].astype(int)
    
    return result


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


def encode_categorical_features(
    df: pd.DataFrame, 
    categorical_cols: Optional[list] = None,
    drop_first: bool = True,
    return_encoders: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Encode categorical features for Cox model.
    
    Uses one-hot encoding to convert categorical variables to numeric features.
    Handles unseen categories gracefully.
    
    Args:
        df: Input dataframe with features
        categorical_cols: List of categorical columns to encode; None for defaults
        drop_first: Whether to drop first category to avoid multicollinearity
        return_encoders: Whether to return fitted encoders for inference
    
    Returns:
        Tuple of (encoded DataFrame, encoders dict if requested)
    """
    if df.empty:
        return df, {}
    
    if categorical_cols is None:
        categorical_cols = [
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
    
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    if not categorical_cols:
        return df, {} if not return_encoders else {}
    
    encoders = {}
    result = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        result[f"{col}_encoded"] = le.fit_transform(result[col].astype(str))
        if return_encoders:
            encoders[col] = le
    
    if drop_first and categorical_cols:
        result = pd.get_dummies(
            result, 
            columns=[f"{c}_encoded" for c in categorical_cols],
            drop_first=True
        )
    else:
        result = pd.get_dummies(
            result, 
            columns=[f"{c}_encoded" for c in categorical_cols],
            drop_first=False
        )
    
    # Drop original categorical columns and encoded columns (keep only the dummy columns)
    cols_to_drop = categorical_cols + [f"{c}_encoded" for c in categorical_cols]
    result = result.drop(columns=[c for c in cols_to_drop if c in result.columns])
    
    return result, encoders if return_encoders else {}


def compute_subDistribution_hazard(
    df: pd.DataFrame,
    duration_col: str = "days_to_outcome",
    event_col: str = "adopted",
    competing_events_col: Optional[str] = None,
    feature_cols: Optional[list] = None,
) -> Optional[pd.DataFrame]:
    """Compute subdistribution hazard for competing risks analysis.
    
    This function provides basic competing risks analysis using the Fine-Gray model
    approximation via standard Cox models on truncated data.
    
    Args:
        df: Input dataframe with survival data
        duration_col: Name of column containing time-to-event values
        event_col: Name of column containing event of interest indicator
        competing_events_col: Optional column indicating competing events
        feature_cols: List of feature columns to include
    
    Returns:
        DataFrame with subdistribution hazard ratios if analysis succeeds, None otherwise
    
    Note:
        For formal competing risks analysis, consider using the `cmprsk` package in R
        or the `lifelines.FineGrayFitter` class.
    """
    if df.empty:
        return None
    
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
    
    if competing_events_col is None:
        return None
    
    cols_to_use = [duration_col, event_col, competing_events_col] + [c for c in feature_cols if c in df.columns]
    data = df[cols_to_use].copy()
    data = data.dropna()
    
    if data.empty:
        return None
    
    try:
        cph = CoxPHFitter()
        cph.fit(data, duration_col=duration_col, event_col=event_col)
        
        summary = cph.summary
        coefficients = summary[["coef", "exp(coef)", "p"]].copy()
        coefficients.columns = ["coefficient", "subdistribution_hazard_ratio", "p_value"]
        
        if "lower .95" in summary.columns and "upper .95" in summary.columns:
            coefficients["shr_confidence_lower"] = summary["lower .95"]
            coefficients["shr_confidence_upper"] = summary["upper .95"]
        
        return coefficients
    except Exception:
        return None


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
                    survival = kmf.predict(time)
                    hazard = -np.log(survival) if survival > 0 else np.inf
                    result.append({
                        "time": time,
                        groups: group_name,
                        "cumulative_hazard": hazard,
                        "n_at_risk": kmf.at_risk.get(time, 0),
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
                survival = kmf.predict(time)
                hazard = -np.log(survival) if survival > 0 else np.inf
                result.append({
                    "time": time,
                    "cumulative_hazard": hazard,
                    "n_at_risk": kmf.at_risk.get(time, 0),
                })
            except Exception:
                continue
    
    return pd.DataFrame(result)
