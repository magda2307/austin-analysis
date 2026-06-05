"""Data-loading and prediction helpers for the Streamlit thesis demo."""

from __future__ import annotations

from pathlib import Path
import json
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


AGE_GROUP_TO_YEARS = {
    "baby": 0.5,
    "young": 2.0,
    "adult": 5.0,
    "senior": 10.0,
    "unknown": 2.0,
}

BREED_GROUP_EXAMPLES = {
    "domestic_cat": "Domestic Shorthair Mix",
    "pit_bull_type": "Pit Bull Mix",
    "chihuahua_type": "Chihuahua Shorthair Mix",
    "retriever_type": "Labrador Retriever Mix",
    "shepherd_type": "German Shepherd Mix",
    "terrier_type": "Terrier Mix",
    "hound_type": "Hound Mix",
    "other": "Mixed Breed",
    "unknown": "Mixed Breed",
}

COLOR_GROUP_EXAMPLES = {
    "black_or_dark": "Black/White",
    "brown_tan": "Brown/White",
    "white_light": "White/Cream",
    "gray_blue": "Blue/White",
    "orange_yellow": "Orange/White",
    "mixed_other": "Tricolor",
    "unknown": "Black/White",
}

TABLE_FILES = {
    "classification": "model_comparison_classification.csv",
    "regression": "model_comparison_regression.csv",
    "h1": "h1_intake_vs_appearance.csv",
    "h3": "h3_age_length_of_stay.csv",
    "h5": "h5_covid_period.csv",
    "animal_archetypes": "animal_archetypes.csv",
    "vulnerable_profiles": "vulnerable_profiles.csv",
    "profile_contrasts": "profile_contrasts.csv",
    "profile_model_error": "profile_model_error.csv",
    "health_behavior_profiles": "health_behavior_profiles.csv",
    "model_evidence_pack": "model_evidence_pack.csv",
    "model_limitations_by_cohort": "model_limitations_by_cohort.csv",
    "metric_confidence_intervals": "metric_confidence_intervals.csv",
    "subgroup_reliability": "subgroup_reliability.csv",
    "subgroup_metric_confidence_intervals": "subgroup_metric_confidence_intervals.csv",
    "subgroup_adoption_milestones": "subgroup_adoption_milestones.csv",
    "model_failure_modes": "model_failure_modes.csv",
    "animal_journey_examples": "animal_journey_examples.csv",
    "context_model_comparison": "context_model_comparison.csv",
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
        "found_location_kind": "other",
        "found_location_area": "unknown",
        "is_austin_found_location": False,
        "is_outside_jurisdiction": False,
        "is_intersection_location": False,
        "is_address_like_location": False,
        "is_airport_location": False,
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
        "daily_temp_max": pd.NA,
        "daily_temp_min": pd.NA,
        "daily_precipitation": pd.NA,
        "is_extreme_heat": False,
        "is_rainy_day": False,
        "animal_311_requests_7d": 0.0,
        "animal_311_requests_30d": 0.0,
        "intake_volume_7d": 0.0,
        "intake_volume_30d": 0.0,
    }
    return pd.DataFrame([{column: record[column] for column in INTAKE_TIME_FEATURES}])


def build_profile_prediction_record(profile: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Create a representative model row from an animal archetype profile."""
    row = profile if isinstance(profile, pd.Series) else pd.Series(profile)
    age_group = str(row.get("age_group", "unknown"))
    breed_group = str(row.get("simplified_breed_group", "unknown"))
    color_group = str(row.get("simplified_color_group", "unknown"))
    age_years = AGE_GROUP_TO_YEARS.get(age_group, AGE_GROUP_TO_YEARS["unknown"])
    return build_prediction_record(
        animal_type=str(row.get("animal_type", "Dog")),
        intake_type=str(row.get("intake_type", "Stray")),
        intake_condition=str(row.get("intake_condition", "Normal")),
        sex_upon_intake=str(row.get("sex_upon_intake", "Unknown")),
        age_days=age_years * 365.25,
        breed=BREED_GROUP_EXAMPLES.get(breed_group, BREED_GROUP_EXAMPLES["unknown"]),
        color=COLOR_GROUP_EXAMPLES.get(color_group, COLOR_GROUP_EXAMPLES["unknown"]),
        has_name=bool(row.get("is_named", False)),
        intake_date=pd.Timestamp("2024-06-01"),
    )


def load_model(models_dir: str | Path, task: str, subset: str = "combined", model_name: str = "catboost"):
    """Load a trained model artifact by canonical path."""
    path = artifact_path(models_dir, task, subset, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def load_model_metadata(models_dir: str | Path, task: str, subset: str = "combined", model_name: str = "catboost") -> dict[str, Any]:
    """Load sidecar model metadata when available."""
    path = artifact_path(models_dir, task, subset, model_name).with_suffix(".json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_feature_columns(
    record: pd.DataFrame,
    models_dir: str | Path,
    task: str,
    subset: str = "combined",
    model_name: str = "catboost",
) -> list[str]:
    """Return feature columns expected by a saved model."""
    metadata = load_model_metadata(models_dir, task, subset, model_name)
    expected = metadata.get("feature_columns")
    if expected:
        return [column for column in expected if column in record.columns]
    return list(record.columns)


def predict_from_record(
    record: pd.DataFrame,
    models_dir: str | Path = "models/advanced",
    subset: str = "combined",
) -> dict[str, float]:
    """Predict adoption probability and expected days to outcome for one row."""
    classifier = load_model(models_dir, "classification", subset)
    regressor = load_model(models_dir, "regression", subset)
    classification_features = model_feature_columns(record, models_dir, "classification", subset)
    regression_features = model_feature_columns(record, models_dir, "regression", subset)
    classification_record = prepare_catboost_frame(record, classification_features)
    regression_record = prepare_catboost_frame(record, regression_features)
    probability = float(classifier.predict_proba(classification_record)[:, 1][0])
    days = max(0.0, float(regressor.predict(regression_record)[0]))
    return {
        "adoption_probability": probability,
        "predicted_days_to_outcome": days,
    }


def visibility_need_from_prediction(adoption_probability: float, predicted_days: float) -> str:
    """Label a profile from prediction pair used by the dashboard journey card."""
    if adoption_probability >= 0.60 and predicted_days < 14:
        return "quick placement likely"
    if adoption_probability >= 0.50 and predicted_days >= 14:
        return "needs visibility"
    if adoption_probability < 0.40 and predicted_days >= 14:
        return "long-stay risk"
    return "outcome support priority"


def local_shap_explanations(
    record: pd.DataFrame,
    models_dir: str | Path = "models/advanced",
    *,
    task: str = "classification",
    subset: str = "combined",
    top_n: int = 8,
) -> pd.DataFrame:
    """Compute local CatBoost SHAP explanations for a single prediction row."""
    try:
        import numpy as np
        import shap
    except ImportError:
        return pd.DataFrame()

    model = load_model(models_dir, task, subset)
    feature_columns = model_feature_columns(record, models_dir, task, subset)
    catboost_record = prepare_catboost_frame(record, feature_columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(catboost_record)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = np.asarray(shap_values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] == len(feature_columns) + 1:
        values = values[:, :-1]
    row_values = values[0]
    table = pd.DataFrame(
        {
            "feature": feature_columns,
            "value": [record.iloc[0][column] for column in feature_columns],
            "shap_value": row_values,
            "abs_shap": np.abs(row_values),
        }
    )
    table["association"] = table["shap_value"].map(lambda value: "raises prediction" if value > 0 else "lowers prediction")
    return table.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def profile_global_shap_reasons(profile: pd.Series | dict[str, Any], shap_global: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Select model-wide SHAP rows that map to fields visible on a profile card."""
    if shap_global.empty or "feature" not in shap_global.columns:
        return pd.DataFrame()
    row = profile if isinstance(profile, pd.Series) else pd.Series(profile)
    visible_features = [
        "animal_type",
        "age_group",
        "age_upon_intake",
        "age_days",
        "age_months",
        "age_years",
        "intake_type",
        "intake_condition",
        "sex_upon_intake",
        "breed",
        "primary_breed",
        "simplified_breed_group",
        "color",
        "primary_color",
        "simplified_color_group",
        "is_named",
        "has_name",
    ]
    view = shap_global[shap_global["feature"].isin(visible_features)].copy()
    if view.empty:
        view = shap_global.copy()
    values = []
    for feature in view["feature"]:
        if feature in row.index:
            values.append(row[feature])
        elif feature in {"breed", "primary_breed"}:
            values.append(row.get("simplified_breed_group", "profile breed group"))
        elif feature in {"color", "primary_color"}:
            values.append(row.get("simplified_color_group", "profile color group"))
        elif feature in {"has_name", "is_named"}:
            values.append(bool(row.get("is_named", False)))
        elif feature.startswith("age_"):
            values.append(row.get("age_group", "profile age group"))
        else:
            values.append("")
    view["profile_value"] = values
    sort_column = "mean_abs_shap" if "mean_abs_shap" in view.columns else view.columns[-1]
    return view.sort_values(sort_column, ascending=False).head(top_n).reset_index(drop=True)


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
        "is_named",
        "classification_target",
        "days_to_outcome",
        "outcome_type",
    ]
    df = pd.read_csv(path, usecols=lambda column: column in columns, nrows=max_rows)
    if df.empty:
        return pd.DataFrame()
    query = record.iloc[0]
    match_levels = [
        (
            "exact visible profile",
            [
                "animal_type",
                "age_group",
                "intake_type",
                "intake_condition",
                "simplified_breed_group",
                "simplified_color_group",
                "sex_upon_intake",
                "is_named",
            ],
        ),
        (
            "profile without name",
            [
                "animal_type",
                "age_group",
                "intake_type",
                "intake_condition",
                "simplified_breed_group",
                "simplified_color_group",
                "sex_upon_intake",
            ],
        ),
        ("coarse care context", ["animal_type", "age_group", "intake_type", "intake_condition"]),
    ]

    matches = pd.DataFrame()
    matching_level = ""
    used_columns: list[str] = []
    for level, match_columns in match_levels:
        mask = pd.Series(True, index=df.index)
        for column in match_columns:
            if column not in df.columns or column not in record.columns:
                continue
            mask &= df[column].astype(str).eq(str(query[column]))
        matches = df[mask]
        if not matches.empty:
            matching_level = level
            used_columns = match_columns
            break
    if matches.empty:
        return pd.DataFrame()

    outcome_rates: dict[str, float] = {}
    if "outcome_type" in matches.columns:
        for outcome in ["Adoption", "Transfer", "Return to Owner", "Euthanasia"]:
            outcome_rates[f"{outcome.lower().replace(' ', '_')}_rate_pct"] = float(matches["outcome_type"].eq(outcome).mean() * 100)

    return pd.DataFrame(
        [
            {
                "similar_records": len(matches),
                "historical_adoption_rate_pct": float(matches["classification_target"].mean() * 100),
                "median_days_to_outcome": float(matches["days_to_outcome"].median()),
                "matching_level": matching_level,
                "matched_fields": ", ".join(used_columns),
                **outcome_rates,
            }
        ]
    )
