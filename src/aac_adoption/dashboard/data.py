"""Data-loading and prediction helpers for the Streamlit thesis demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aac_adoption.features.feature_engineering import (
    age_group_from_days,
    covid_period_from_date,
    is_mixed_breed,
    primary_breed,
    primary_color,
    season_from_month,
    simplified_breed_group,
    simplify_color,
)
from aac_adoption.features.feature_sets import INTAKE_TIME_FEATURES
from aac_adoption.models.artifacts import artifact_path
from aac_adoption.models.train_advanced import prepare_catboost_frame


TABLE_FILES = {
    "classification": "model_comparison_classification.csv",
    "regression": "model_comparison_regression.csv",
    "h1": "h1_intake_vs_appearance.csv",
    "h3": "h3_age_adoption_speed.csv",
    "h5": "h5_covid_period.csv",
}

DIAGNOSTIC_FILES = {
    "thresholds": "classification_thresholds.csv",
    "calibration": "classification_calibration.csv",
    "classification_slices": "classification_error_slices.csv",
    "regression_slices": "regression_error_slices.csv",
    "risk_quadrants": "placement_risk_quadrants.csv",
    "predictions": "diagnostic_predictions_sample.csv",
}


def load_table(tables_dir: str | Path, key: str) -> pd.DataFrame:
    """Load one known dashboard table or return an empty frame."""
    filename = TABLE_FILES[key]
    path = Path(tables_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_optional_csv(base_dir: str | Path, filename: str) -> pd.DataFrame:
    """Load an optional CSV artifact."""
    path = Path(base_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_diagnostic(diagnostics_dir: str | Path, key: str) -> pd.DataFrame:
    """Load one known diagnostic artifact."""
    return load_optional_csv(diagnostics_dir, DIAGNOSTIC_FILES[key])


def load_summary(summary_dir: str | Path = "reports/summary") -> str:
    """Load generated Markdown summary for display."""
    path = Path(summary_dir) / "current_results.md"
    if not path.exists():
        return "Generate report outputs first with `python scripts/generate_report_outputs.py`."
    return path.read_text(encoding="utf-8")


def best_model_rows(classification: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    """Return compact best-model rows for overview cards."""
    rows: list[dict[str, Any]] = []
    if not classification.empty and {"animal_subset", "model_name", "roc_auc"}.issubset(classification.columns):
        for subset, group in classification.dropna(subset=["roc_auc"]).groupby("animal_subset"):
            best = group.sort_values("roc_auc", ascending=False).iloc[0]
            rows.append(
                {
                    "task": "classification",
                    "animal_subset": subset,
                    "model_name": best["model_name"],
                    "primary_metric": "roc_auc",
                    "score": float(best["roc_auc"]),
                }
            )
    if not regression.empty and {"animal_subset", "model_name", "mae"}.issubset(regression.columns):
        for subset, group in regression.dropna(subset=["mae"]).groupby("animal_subset"):
            best = group.sort_values("mae", ascending=True).iloc[0]
            rows.append(
                {
                    "task": "regression",
                    "animal_subset": subset,
                    "model_name": best["model_name"],
                    "primary_metric": "mae",
                    "score": float(best["mae"]),
                }
            )
    return pd.DataFrame(rows)


def build_prediction_record(
    *,
    animal_type: str,
    intake_type: str,
    intake_condition: str,
    sex_upon_intake: str,
    age_days: float,
    breed: str,
    color: str,
    has_name: bool,
    intake_date: pd.Timestamp,
) -> pd.DataFrame:
    """Create one intake-time feature row compatible with trained pipelines."""
    intake_date = pd.Timestamp(intake_date)
    age_days = float(age_days)
    age_months = age_days / 30.4375
    age_years = age_days / 365.25
    color_group = simplify_color(color)

    record = {
        "animal_type": animal_type,
        "intake_type": intake_type,
        "intake_condition": intake_condition,
        "sex_upon_intake": sex_upon_intake,
        "age_upon_intake": f"{age_years:.1f} years",
        "age_days": age_days,
        "age_months": age_months,
        "age_years": age_years,
        "age_group": age_group_from_days(age_days),
        "breed": breed,
        "primary_breed": primary_breed(breed),
        "is_mixed_breed": is_mixed_breed(breed),
        "simplified_breed_group": simplified_breed_group(breed),
        "color": color,
        "primary_color": primary_color(color),
        "simplified_color_group": color_group,
        "is_black_or_dark": color_group == "black_or_dark",
        "has_name": has_name,
        "is_named": has_name,
        "intake_year": intake_date.year,
        "intake_month": intake_date.month,
        "intake_quarter": intake_date.quarter,
        "intake_season": season_from_month(intake_date.month),
        "covid_period": covid_period_from_date(intake_date),
    }
    return pd.DataFrame([{column: record[column] for column in INTAKE_TIME_FEATURES}])


def load_model(models_dir: str | Path, task: str, subset: str = "combined", model_name: str = "catboost"):
    """Load a trained model artifact by canonical path."""
    path = artifact_path(models_dir, task, subset, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def predict_from_record(
    record: pd.DataFrame,
    models_dir: str | Path = "models/advanced",
    subset: str = "combined",
) -> dict[str, float]:
    """Predict adoption probability and expected days to outcome for one row."""
    classifier = load_model(models_dir, "classification", subset)
    regressor = load_model(models_dir, "regression", subset)
    catboost_record = prepare_catboost_frame(record, list(record.columns))
    probability = float(classifier.predict_proba(catboost_record)[:, 1][0])
    days = max(0.0, float(regressor.predict(catboost_record)[0]))
    return {
        "adoption_probability": probability,
        "predicted_days_to_outcome": days,
    }


def similar_historical_cases(data_path: str | Path, record: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    """Find exact/coarse historical matches for a what-if record."""
    path = Path(data_path)
    if not path.exists():
        return pd.DataFrame()
    columns = [
        "animal_type",
        "age_group",
        "intake_type",
        "intake_condition",
        "simplified_breed_group",
        "simplified_color_group",
        "sex_upon_intake",
        "classification_target",
        "days_to_outcome",
    ]
    df = pd.read_csv(path, usecols=lambda column: column in columns, nrows=max_rows)
    if df.empty:
        return pd.DataFrame()
    query = record.iloc[0]
    match_columns = [
        "animal_type",
        "age_group",
        "intake_type",
        "intake_condition",
        "simplified_breed_group",
        "simplified_color_group",
        "sex_upon_intake",
    ]
    mask = pd.Series(True, index=df.index)
    for column in match_columns:
        if column in df.columns and column in record.columns:
            mask &= df[column].astype(str).eq(str(query[column]))
    matches = df[mask]
    if matches.empty:
        match_columns = ["animal_type", "age_group", "intake_type", "intake_condition"]
        mask = pd.Series(True, index=df.index)
        for column in match_columns:
            mask &= df[column].astype(str).eq(str(query[column]))
        matches = df[mask]
    if matches.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "similar_records": len(matches),
                "historical_adoption_rate_pct": float(matches["classification_target"].mean() * 100),
                "median_days_to_outcome": float(matches["days_to_outcome"].median()),
                "matching_level": "coarse" if len(match_columns) == 4 else "exact_coarse_features",
            }
        ]
    )
