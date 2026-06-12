"""Data-loading and prediction helpers for the Streamlit thesis demo."""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any
from dataclasses import dataclass

import joblib
import pandas as pd
import numpy as np
import streamlit as st

from aac_adoption.config import PROJECT_ROOT

DASHBOARD_TABLE_SCHEMAS = {
    "final_model_selection": {"selected": "bool", "task": "str", "animal_subset": "str", "model_name": "str"},
    "classification": {"model_name": "str", "roc_auc": "float", "pr_auc": "float"},
    "regression": {"model_name": "str", "mae": "float"},
    "animal_archetypes": {"profile_label": "str", "is_named": "bool"},
    "model_limitations_by_cohort": {"cohort": "str", "small_cohort_flag": "bool"},
    "subgroup_reliability": {"cohort": "str", "small_cohort_flag": "bool"},
    "context_model_comparison": {"task": "str", "higher_is_better": "bool"},
}

def parse_strict_boolean(value: Any) -> bool | None:
    """Safely extract boolean state, avoiding truthiness of strings. Returns None if null/nan."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "t", "y"):
            return True
        if v in ("false", "0", "no", "f", "n"):
            return False
    raise ValueError(f"Cannot parse strict boolean from {value!r}")

def _file_fingerprint(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(path.resolve()), 0, 0)

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
    "final_model_selection": "final_model_selection.csv",
}

DIAGNOSTIC_FILES = {
    "thresholds": "classification_thresholds.csv",
    "calibration": "classification_calibration.csv",
    "classification_slices": "classification_error_slices.csv",
    "regression_slices": "regression_error_slices.csv",
    "risk_quadrants": "placement_risk_quadrants.csv",
    "predictions": "diagnostic_predictions_sample.csv",
}


@st.cache_data(show_spinner=False)
def _cached_safe_load_csv(path: Path, fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

def _safe_load_csv(path: Path) -> pd.DataFrame:
    return _cached_safe_load_csv(path, _file_fingerprint(path))

def load_table(tables_dir: str | Path, key: str) -> pd.DataFrame:
    """Load one known dashboard table or return an empty frame."""
    filename = TABLE_FILES.get(key)
    if not filename:
        return pd.DataFrame()
    path = Path(tables_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    df = _safe_load_csv(path)
    if df.empty:
        return df

    schema = DASHBOARD_TABLE_SCHEMAS.get(key)
    if schema:
        missing = [c for c in schema if c not in df.columns]
        if missing:
            return pd.DataFrame()
            
        # Parse strict booleans
        for c, t in schema.items():
            if t == "bool":
                try:
                    df[c] = df[c].apply(parse_strict_boolean)
                except ValueError:
                    return pd.DataFrame()
                    
    return df

def load_optional_csv(base_dir: str | Path, filename: str) -> pd.DataFrame:
    """Load an optional CSV artifact."""
    path = Path(base_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    return _safe_load_csv(path)


def load_diagnostic(diagnostics_dir: str | Path, key: str) -> pd.DataFrame:
    """Load one known diagnostic artifact."""
    return load_optional_csv(diagnostics_dir, DIAGNOSTIC_FILES[key])


@st.cache_data(show_spinner=False)
def _cached_load_summary(path: Path, fingerprint: tuple[str, int, int]) -> str:
    return path.read_text(encoding="utf-8")

def load_summary(summary_dir: str | Path = "reports/summary") -> str:
    """Load generated Markdown summary for display."""
    path = Path(summary_dir) / "current_results.md"
    if not path.exists():
        return "Generate report outputs first with `python scripts/generate_report_outputs.py`."
    return _cached_load_summary(path, _file_fingerprint(path))


def best_model_rows(classification: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    """Return compact best-model rows for overview cards."""
    rows: list[dict[str, Any]] = []
    if not classification.empty and {"animal_subset", "model_name", "roc_auc"}.issubset(classification.columns):
        has_pr_auc = "pr_auc" in classification.columns
        if has_pr_auc:
            for subset, group in classification.dropna(subset=["pr_auc"]).groupby("animal_subset"):
                best = group.sort_values(["pr_auc", "roc_auc"], ascending=[False, False]).iloc[0]
                rows.append(
                    {
                        "task": "classification",
                        "animal_subset": subset,
                        "model_name": best["model_name"],
                        "primary_metric": "pr_auc",
                        "score": float(best["pr_auc"]),
                    }
                )
        else:
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
    return pd.DataFrame([record])


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


@st.cache_resource(show_spinner=False)
def _cached_load_model(path: Path, fingerprint: tuple[str, int, int]) -> Any:
    return joblib.load(path)

def load_model(models_dir: str | Path, task: str, subset: str = "combined", model_name: str = "catboost"):
    """Load a trained model artifact by canonical path."""
    # Resolve from PROJECT_ROOT
    base_dir = Path(models_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
        
    path = artifact_path(base_dir, task, subset, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return _cached_load_model(path, _file_fingerprint(path))


@st.cache_data(show_spinner=False)
def _cached_load_metadata(path: Path, fingerprint: tuple[str, int, int]) -> dict[str, Any]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    from aac_adoption.models.metadata import validate_model_metadata
    validate_model_metadata(meta)
    return meta

def load_model_metadata(models_dir: str | Path, task: str, subset: str = "combined", model_name: str = "catboost") -> dict[str, Any]:
    """Load sidecar model metadata when available."""
    # Resolve from PROJECT_ROOT
    base_dir = Path(models_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
        
    path = artifact_path(base_dir, task, subset, model_name).with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"Missing model metadata: {path}")
    return _cached_load_metadata(path, _file_fingerprint(path))


def model_feature_columns(
    record: pd.DataFrame,
    models_dir: str | Path,
    task: str,
    subset: str = "combined",
    model_name: str = "catboost",
) -> list[str]:
    """Return feature columns expected by a saved model."""
    metadata = load_model_metadata(models_dir, task, subset, model_name)
    expected = metadata["feature_columns"]
    missing = [c for c in expected if c not in record.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return expected


def _infer_models_dir(model_name: str) -> str:
    if "catboost" in model_name:
        return "models/advanced"
    if "boosting" in model_name:
        return "models/boosting"
    return "models/baseline"

def los_days_to_bucket(days: float) -> str:
    """Map length-of-stay days to category bucket."""
    if math.isnan(days):
        return "unknown"
    if days < 0:
        return "invalid"
    if days <= 7:
        return "0-7d"
    elif days <= 30:
        return "8-30d"
    elif days <= 60:
        return "31-60d"
    elif days <= 90:
        return "61-90d"
    else:
        return "90+d"


@dataclass(frozen=True)
class PredictionResult:
    ok: bool
    adoption_probability: float | None
    predicted_days_to_outcome: float | None
    los_bucket: str | None
    is_calibrated: bool
    model_artifacts: dict[str, str]
    error_code: str | None
    error_message: str | None

def predict_from_record(
    record: pd.DataFrame,
    models_dir: str | Path | None = None,
    subset: str = "combined",
) -> PredictionResult:
    """Predict adoption probability and expected days to outcome for one row."""
    selection = load_table(PROJECT_ROOT / "reports/tables", "final_model_selection")
    
    if selection.empty or "selected" not in selection.columns:
        return PredictionResult(
            ok=False,
            adoption_probability=None,
            predicted_days_to_outcome=None,
            los_bucket=None,
            is_calibrated=False,
            model_artifacts={},
            error_code="MISSING_SELECTION",
            error_message="final_model_selection.csv is missing or empty — cannot determine selected model",
        )

    clf_name = "catboost"
    clf_dir = "models/advanced"
    reg_name = "catboost"
    reg_dir = "models/advanced"

    clf_rows = selection[(selection["selected"] == True) & (selection["task"] == "classification") & (selection["animal_subset"] == subset)]
    if not clf_rows.empty:
        clf_name = clf_rows.iloc[0]["model_name"]
        clf_dir = _infer_models_dir(clf_name)
        
    reg_rows = selection[(selection["selected"] == True) & (selection["task"] == "regression") & (selection["animal_subset"] == subset)]
    if not reg_rows.empty:
        reg_name = reg_rows.iloc[0]["model_name"]
        reg_dir = _infer_models_dir(reg_name)

    if models_dir:
        supplied_dir = Path(models_dir)
        models_root = (
            supplied_dir.parent
            if supplied_dir.name in {"advanced", "baseline", "boosting", "calibrated"}
            else supplied_dir
        )
    else:
        models_root = PROJECT_ROOT / "models"

    clf_dir_path = models_root / Path(clf_dir).name
    reg_dir_path = models_root / Path(reg_dir).name
    calibrated_base_dir = models_root / "calibrated"

    calibrated_path = artifact_path(
        base_dir=calibrated_base_dir,
        task="classification_calibrated",
        animal_subset=subset,
        model_name=f"{clf_name}_calibrated"
    )

    is_calibrated = False
    artifacts = {}

    # Classification
    try:
        if calibrated_path.exists():
            classifier = _cached_load_model(calibrated_path, _file_fingerprint(calibrated_path))
            clf_features = model_feature_columns(
                record=record,
                models_dir=calibrated_base_dir,
                task="classification_calibrated",
                subset=subset,
                model_name=f"{clf_name}_calibrated"
            )
            artifacts["classifier"] = str(calibrated_path)
            clf_record = prepare_catboost_frame(record, clf_features) if "catboost" in clf_name else record[clf_features]
            probability = float(classifier.predict_proba(clf_record)[:, 1][0])
            is_calibrated = True
        else:
            classifier = load_model(clf_dir_path, "classification", subset, clf_name)
            clf_features = model_feature_columns(record, clf_dir_path, "classification", subset, clf_name)
            artifacts["classifier"] = str(artifact_path(clf_dir_path, "classification", subset, clf_name))
            clf_record = prepare_catboost_frame(record, clf_features) if "catboost" in clf_name else record[clf_features]
            probability = float(classifier.predict_proba(clf_record)[:, 1][0])
            
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            return PredictionResult(False, None, None, None, False, artifacts, "INVALID_PROB", f"Probability {probability} out of bounds")
            
    except Exception as e:
        return PredictionResult(False, None, None, None, False, artifacts, "CLF_ERROR", str(e))

    # Regression
    try:
        regressor = load_model(reg_dir_path, "regression", subset, reg_name)
        reg_features = model_feature_columns(record, reg_dir_path, "regression", subset, reg_name)
        artifacts["regressor"] = str(artifact_path(reg_dir_path, "regression", subset, reg_name))
        
        reg_meta = load_model_metadata(reg_dir_path, "regression", subset, reg_name)
        transform = reg_meta.get("prediction_inverse_transform", "none")
        
        reg_record = prepare_catboost_frame(record, reg_features) if "catboost" in reg_name else record[reg_features]
        raw_days = float(regressor.predict(reg_record)[0])
        
        if transform == "expm1":
            days = math.expm1(raw_days)
        elif transform == "none":
            days = raw_days
        else:
            days = raw_days # fallback
            
        if not math.isfinite(days) or days < 0.0:
            return PredictionResult(False, None, None, None, False, artifacts, "INVALID_DAYS", f"Days {days} out of bounds")
            
        # Apply operational bound if present in metadata (or assume 4000 as absolute upper bound)
        if days > 4000:
            return PredictionResult(False, None, None, None, False, artifacts, "OUT_OF_BOUNDS", f"Days {days} exceeds operational bound")
            
    except Exception as e:
        return PredictionResult(False, None, None, None, False, artifacts, "REG_ERROR", str(e))
    
    return PredictionResult(
        ok=True,
        adoption_probability=probability,
        predicted_days_to_outcome=days,
        los_bucket=los_days_to_bucket(days),
        is_calibrated=is_calibrated,
        model_artifacts=artifacts,
        error_code=None,
        error_message=None,
    )


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


@st.cache_data(show_spinner=False)
def _cached_dataset(path: Path, max_rows: int, fingerprint: tuple[str, int, int]) -> pd.DataFrame:
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
    header_cols = list(pd.read_csv(path, nrows=0).columns)
    use_cols = [col for col in columns if col in header_cols]
    return pd.read_csv(path, usecols=use_cols, nrows=max_rows)

def similar_historical_cases(data_path: str | Path, record: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    """Find exact/coarse historical matches for a model-sensitivity record."""
    path = Path(data_path)
    if not path.exists():
        return pd.DataFrame()
    df = _cached_dataset(path, max_rows, _file_fingerprint(path))

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
            val = query[column]
            if pd.isna(val):
                mask &= df[column].isna()
            else:
                mask &= df[column].astype(str).eq(str(val))
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

    res = {
        "similar_records": len(matches),
        "matching_level": matching_level,
        "matched_fields": ", ".join(used_columns),
        **outcome_rates,
    }

    if "classification_target" in matches.columns and not matches["classification_target"].isna().all():
        res["historical_adoption_rate_pct"] = float(matches["classification_target"].mean() * 100)
    else:
        res["historical_adoption_rate_pct"] = 0.0

    if "days_to_outcome" in matches.columns and not matches["days_to_outcome"].isna().all():
        res["median_days_to_outcome"] = float(matches["days_to_outcome"].median())
    else:
        res["median_days_to_outcome"] = 0.0

    return pd.DataFrame([res])
