"""Survival model training for LOS analysis with censoring support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    available_features_for_df,
    model_feature_columns,
)
from aac_adoption.features.target_encoder import OOFBayesianTargetEncoder
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import brier_score_loss
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, limit_rows
from aac_adoption.models.metadata import base_training_metadata



@dataclass(frozen=True)
class SurvivalTrainingOutputs:
    """Metric tables returned by survival training."""

    concordance_metrics: pd.DataFrame
    brier_metrics: pd.DataFrame
    calibration_metrics: pd.DataFrame


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Handle categorical encoding for Cox model preprocessing."""

    def __init__(self, categorical_cols: list[str] | None = None):
        self.categorical_cols = categorical_cols or []
        self.encodings_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> CategoricalEncoder:
        """Fit encoder by computing mean target for each category."""
        if y is None:
            raise ValueError("Target values are required for fitting")

        self.encodings_ = {}
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            stats = pd.DataFrame({"feat": X[col], "target": y}).groupby("feat")["target"].mean()
            self.encodings_[col] = stats.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to new data."""
        X_out = X.copy()
        global_mean = float(pd.Series(list(self.encodings_.values())).mean()) if self.encodings_ else 0.0

        for col, mapping in self.encodings_.items():
            if col not in X_out.columns:
                continue
            X_out[col] = X_out[col].map(mapping).fillna(global_mean).astype(float)

        return X_out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


def prepare_survival_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Prepare feature frame for survival analysis with proper encodings."""
    result = df[feature_columns].copy()

    categorical_cols = [
        col for col in feature_columns
        if col in ["animal_type", "age_group", "intake_type", "intake_condition",
                   "sex_upon_intake", "simplified_breed_group", "simplified_color_group",
                   "found_location_kind"]
    ]

    for col in categorical_cols:
        if col in result.columns:
            result[col] = result[col].astype("string").fillna("unknown").astype(str)

    numeric_cols = [col for col in feature_columns if col not in categorical_cols]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def compute_brier_score_survival(
    df: pd.DataFrame,
    predicted_survival_probs: np.ndarray,
    time_col: str = "days_to_outcome",
    event_col: str = "adopted",
    time_horizon: float = 30.0,
) -> float:
    """Compute time-specific Brier score for survival predictions."""
    if df.empty:
        return 0.0

    df_work = df.copy()
    df_work["event"] = df_work[event_col]
    df_work["time"] = df_work[time_col]
    df_work["predicted_risk"] = predicted_survival_probs

    mask = df_work["time"] <= time_horizon
    if mask.sum() < 2:
        return 0.0

    observed = df_work.loc[mask, "event"].values
    predicted = df_work.loc[mask, "predicted_risk"].values

    try:
        return float(brier_score_loss(1 - observed, 1 - predicted))
    except Exception:
        return 0.0


def compute_calibration_survival(
    df: pd.DataFrame,
    predicted_risks: np.ndarray,
    time_horizon: float = 30.0,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics for survival predictions."""
    if df.empty:
        return {}

    df_work = df.copy()
    df_work["predicted_risk"] = predicted_risks
    df_work["event"] = df_work["adopted"]
    df_work["time"] = df_work["days_to_outcome"]

    mask = df_work["time"] <= time_horizon
    if mask.sum() < 2:
        return {"calibration_slope": np.nan, "calibration_intercept": np.nan}

    observed = df_work.loc[mask, "event"].values
    predicted = df_work.loc[mask, "predicted_risk"].values

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    observed_rates = []
    for i in range(len(bins) - 1):
        if bins[i] == 1.0:
            bin_mask = (predicted >= bins[i]) & (predicted <= bins[i + 1])
        else:
            bin_mask = (predicted >= bins[i]) & (predicted < bins[i + 1])

        if bin_mask.sum() > 0:
            observed_rates.append(observed[bin_mask].mean())
        else:
            observed_rates.append(np.nan)

    observed_rates = np.array(observed_rates)
    bin_centers = bin_centers[~np.isnan(observed_rates)]
    observed_rates = observed_rates[~np.isnan(observed_rates)]

    if len(observed_rates) < 2:
        return {"calibration_slope": np.nan, "calibration_intercept": np.nan}

    try:
        coeffs = np.polyfit(bin_centers, observed_rates, 1)
        return {
            "calibration_slope": float(coeffs[0]),
            "calibration_intercept": float(coeffs[1]),
            "calibration_r2": float(np.corrcoef(observed_rates, np.polyval(coeffs, bin_centers))[0, 1] ** 2)
            if np.std(observed_rates) > 0 else 0.0,
        }
    except Exception:
        return {}


def train_all_survival(
    data_path: str | Path,
    metrics_dir: str | Path = "reports/metrics",
    models_dir: str | Path = "models/survival",
    tables_dir: str | Path = "reports/tables",
    max_rows: int | None = None,
    smoothing: float = 10.0,
    tuned_params_path: str | Path | None = None,
) -> SurvivalTrainingOutputs:
    """Train all survival models and save metric outputs."""
    raise NotImplementedError(
        "Thesis scope includes descriptive Kaplan-Meier analysis only; "
        "Cox and competing-risk model training is not an accepted pipeline step."
    )

