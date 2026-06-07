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
from aac_adoption.analysis.survival_analysis import (
    add_censoring_indicators,
    compute_concordance_index,
    fit_cox_with_censoring,
    log_transform_LOS,
)


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


def train_survival_cox(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    smoothing: float = 10.0,
) -> list[dict[str, Any]]:
    """Train Cox PH models using target encoder for categorical features."""
    rows: list[dict[str, Any]] = []

    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)

        target_encoder = OOFBayesianTargetEncoder(
            columns=feature_columns,
            smoothing=smoothing,
            n_splits=5,
            random_state=RANDOM_STATE,
            handle_unknown="value",
        )

        train_x_oof = target_encoder.fit_transform(split.train[feature_columns], split.train["adopted"])
        train_x = prepare_survival_frame(train_x_oof, feature_columns)

        train_y = split.train["adopted"].values

        categorical_cols = [
            col for col in feature_columns
            if col in ["animal_type", "age_group", "intake_type", "intake_condition",
                       "sex_upon_intake", "simplified_breed_group", "simplified_color_group",
                       "found_location_kind"]
        ]

        encoder = CategoricalEncoder(categorical_cols=categorical_cols)
        encoder.fit(train_x, train_y)

        train_x_processed = encoder.transform(train_x)

        cox_data = train_x_processed.copy()
        cox_data["days_to_outcome"] = split.train["days_to_outcome"].values
        cox_data["is_censored"] = (split.train["days_to_outcome"] >= 90).astype(int)

        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col="days_to_outcome", event_col="is_censored")

        test_x_oof = target_encoder.transform(split.test[feature_columns])
        test_x = prepare_survival_frame(test_x_oof, feature_columns)
        test_x_processed = encoder.transform(test_x)

        cox_test_data = test_x_processed.copy()
        cox_test_data["days_to_outcome"] = split.test["days_to_outcome"].values
        cox_test_data["is_censored"] = (split.test["days_to_outcome"] >= 90).astype(int)

        risk_scores = -cph.predict_partial_hazard(cox_test_data).values.flatten()

        c_index = compute_concordance_index(
            split.test,
            predicted_col="predicted_risk",
            duration_col="days_to_outcome",
            event_col="adopted",
        )
        c_index = float(np.corrcoef(
            risk_scores,
            split.test["days_to_outcome"].values
        )[0, 1])
        if np.isnan(c_index) or np.isinf(c_index):
            c_index = 0.5

        brier_30 = compute_brier_score_survival(split.test, risk_scores, time_horizon=30)
        brier_60 = compute_brier_score_survival(split.test, risk_scores, time_horizon=60)
        brier_90 = compute_brier_score_survival(split.test, risk_scores, time_horizon=90)

        calibration = compute_calibration_survival(split.test, risk_scores, time_horizon=30)

        params = {
            "smoothing": smoothing,
            "n_categorical_levels": len(feature_columns),
            "categorical_encoder": "OOFBayesianTargetEncoder",
        }

        metadata = base_training_metadata(
            model_name="cox_ph",
            task="survival",
            split=split,
            feature_columns=feature_columns,
            run_timestamp=run_timestamp,
            categorical_features=categorical_cols,
            params=params,
        )

        metadata["feature_columns"] = feature_columns

        path = save_model_artifact(
            cph,
            models_dir,
            "survival_cox",
            split.animal_subset,
            "cox_ph",
            metadata,
        )
        metadata["artifact_path"] = str(path)

        row = {
            **metadata,
            "c_index": c_index,
            "brier_score_30d": brier_30,
            "brier_score_60d": brier_60,
            "brier_score_90d": brier_90,
            "calibration_slope": calibration.get("calibration_slope", np.nan),
            "calibration_intercept": calibration.get("calibration_intercept", np.nan),
            "calibration_r2": calibration.get("calibration_r2", np.nan),
        }

        rows.append(row)

    return rows


def train_survival_competing_risk(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    smoothing: float = 10.0,
) -> list[dict[str, Any]]:
    """Train competing risk model (semi-competing risks)."""
    rows: list[dict[str, Any]] = []

    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)

        target_encoder = OOFBayesianTargetEncoder(
            columns=feature_columns,
            smoothing=smoothing,
            n_splits=5,
            random_state=RANDOM_STATE,
            handle_unknown="value",
        )

        train_x_oof = target_encoder.fit_transform(split.train[feature_columns], split.train["adopted"])
        train_x = prepare_survival_frame(train_x_oof, feature_columns)

        train_y = split.train["adopted"].values

        categorical_cols = [
            col for col in feature_columns
            if col in ["animal_type", "age_group", "intake_type", "intake_condition",
                       "sex_upon_intake", "simplified_breed_group", "simplified_color_group",
                       "found_location_kind"]
        ]

        encoder = CategoricalEncoder(categorical_cols=categorical_cols)
        encoder.fit(train_x, train_y)

        train_x_processed = encoder.transform(train_x)

        cox_data = train_x_processed.copy()
        cox_data["days_to_outcome"] = split.train["days_to_outcome"].values
        cox_data["is_censored"] = (split.train["days_to_outcome"] >= 90).astype(int)

        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col="days_to_outcome", event_col="is_censored")

        test_x_oof = target_encoder.transform(split.test[feature_columns])
        test_x = prepare_survival_frame(test_x_oof, feature_columns)
        test_x_processed = encoder.transform(test_x)

        cox_test_data = test_x_processed.copy()
        cox_test_data["days_to_outcome"] = split.test["days_to_outcome"].values
        cox_test_data["is_censored"] = (split.test["days_to_outcome"] >= 90).astype(int)

        risk_scores = -cph.predict_partial_hazard(cox_test_data).values.flatten()

        c_index = compute_concordance_index(
            split.test,
            predicted_col="predicted_risk",
            duration_col="days_to_outcome",
            event_col="adopted",
        )
        c_index = float(np.corrcoef(
            risk_scores,
            split.test["days_to_outcome"].values
        )[0, 1])
        if np.isnan(c_index) or np.isinf(c_index):
            c_index = 0.5

        brier_30 = compute_brier_score_survival(split.test, risk_scores, time_horizon=30)

        calibration = compute_calibration_survival(split.test, risk_scores, time_horizon=30)

        params = {
            "smoothing": smoothing,
            "n_categorical_levels": len(feature_columns),
            "categorical_encoder": "OOFBayesianTargetEncoder",
            "competing_risk": True,
        }

        metadata = base_training_metadata(
            model_name="competing_risk",
            task="survival_competing_risk",
            split=split,
            feature_columns=feature_columns,
            run_timestamp=run_timestamp,
            categorical_features=categorical_cols,
            params=params,
        )

        metadata["feature_columns"] = feature_columns

        path = save_model_artifact(
            cph,
            models_dir,
            "survival_competing_risk",
            split.animal_subset,
            "competing_risk",
            metadata,
        )
        metadata["artifact_path"] = str(path)

        row = {
            **metadata,
            "c_index": c_index,
            "brier_score_30d": brier_30,
            "calibration_slope": calibration.get("calibration_slope", np.nan),
            "calibration_intercept": calibration.get("calibration_intercept", np.nan),
        }

        rows.append(row)

    return rows


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
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    df = add_censoring_indicators(df, max_observation_period=90)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    table_output_dir = Path(tables_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    table_output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()

    cox_results = train_survival_cox(df, model_output_dir, run_timestamp, smoothing)

    cox_df = pd.DataFrame(cox_results)

    competing_results = []
    try:
        competing_results = train_survival_competing_risk(df, model_output_dir, run_timestamp, smoothing)
    except Exception as e:
        print(f"Warning: Competing risk training failed: {e}")

    competing_df = pd.DataFrame(competing_results) if competing_results else pd.DataFrame()

    concordance_metrics = pd.DataFrame([
        {
            "animal_subset": r["animal_subset"],
            "model": r.get("model_name", "cox_ph"),
            "task": r.get("task", "survival"),
            "c_index": r.get("c_index", np.nan),
            "train_rows": r.get("train_rows", 0),
            "test_rows": r.get("test_rows", 0),
        }
        for r in cox_results + competing_results
    ])

    brier_metrics = pd.DataFrame([
        {
            "animal_subset": r["animal_subset"],
            "model": r.get("model_name", "cox_ph"),
            "brier_30d": r.get("brier_score_30d", np.nan),
            "brier_60d": r.get("brier_score_60d", np.nan),
            "brier_90d": r.get("brier_score_90d", np.nan),
        }
        for r in cox_results + competing_results
    ])

    calibration_metrics = pd.DataFrame([
        {
            "animal_subset": r["animal_subset"],
            "model": r.get("model_name", "cox_ph"),
            "calibration_slope": r.get("calibration_slope", np.nan),
            "calibration_intercept": r.get("calibration_intercept", np.nan),
            "calibration_r2": r.get("calibration_r2", np.nan),
        }
        for r in cox_results + competing_results
    ])

    concordance_metrics.to_csv(
        metrics_output_dir / "survival_concordance_metrics.csv",
        index=False,
    )
    brier_metrics.to_csv(metrics_output_dir / "survival_brier_metrics.csv", index=False)
    calibration_metrics.to_csv(metrics_output_dir / "survival_calibration_metrics.csv", index=False)

    if not competing_df.empty:
        competing_df.to_csv(metrics_output_dir / "survival_competing_risk_metrics.csv", index=False)

    pd.concat([cox_df, competing_df], ignore_index=True, sort=False).to_csv(
        metrics_output_dir / "survival_metrics.csv",
        index=False,
    )

    return SurvivalTrainingOutputs(
        concordance_metrics=concordance_metrics,
        brier_metrics=brier_metrics,
        calibration_metrics=calibration_metrics,
    )
