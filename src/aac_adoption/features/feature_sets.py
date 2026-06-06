"""Feature-set definitions and leakage checks."""

import pandas as pd

BASE_INTAKE_TIME_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "age_days",
    "age_group",
    "primary_breed",
    "is_mixed_breed",
    "simplified_breed_group",
    "found_location_kind",
    "found_location_area",
    "is_austin_found_location",
    "is_outside_jurisdiction",
    "is_intersection_location",
    "is_address_like_location",
    "is_airport_location",
    "primary_color",
    "simplified_color_group",
    "is_black_or_dark",
    "is_named",
    "intake_year",
    "intake_month",
    "covid_period",
]

CONTEXT_FEATURES = [
    "daily_temp_max",
    "daily_temp_min",
    "daily_precipitation",
    "is_extreme_heat",
    "is_rainy_day",
    "animal_311_requests_7d",
    "animal_311_requests_30d",
    "intake_volume_7d",
    "intake_volume_30d",
]

INTAKE_TIME_FEATURES = BASE_INTAKE_TIME_FEATURES + CONTEXT_FEATURES

TARGET_COLUMNS = [
    "adopted",
    "is_adopted",
    "target_adopted",
    "classification_target",
    "regression_target_days",
    "days_to_outcome",
    "days_to_adoption",
    "length_of_stay",
]

METADATA_COLUMNS = [
    "animal_id",
    "intake_datetime",
    "outcome_datetime",
    "outcome_type",
    "outcome_subtype",
    "sex_upon_outcome",
    "age_upon_outcome",
]

LEAKAGE_COLUMNS = set(TARGET_COLUMNS + METADATA_COLUMNS) - {"animal_id", "intake_datetime"}

INTAKE_CONDITION = "intake_condition"
SEX_UPON_INTAKE = "sex_upon_intake"

NUMERIC_FEATURES = [
    "age_days",
    "age_in_days",
    "age_in_months",
    "age_in_years",
    "age_months",
    "age_years",
    "daily_temp_max",
    "daily_temp_min",
    "daily_precipitation",
    "animal_311_requests_7d",
    "animal_311_requests_30d",
    "intake_volume_7d",
    "intake_volume_30d",
    "days_to_outcome",
    "length_of_stay",
    "regression_target_days",
    "days_to_adoption",
]

CATEGORICAL_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "primary_breed",
    "simplified_breed_group",
    "primary_color",
    "simplified_color_group",
    "found_location_kind",
    "found_location_area",
    "age_group",
    "intake_season",
    "covid_period",
    "color_group",
    "simplified_color_group",
]


def available_intake_features(columns: list[str] | set[str]) -> list[str]:
    """Return configured intake-time features present in a dataset."""
    column_set = set(columns)
    return [column for column in INTAKE_TIME_FEATURES if column in column_set]


def validate_no_leakage(feature_columns: list[str]) -> None:
    """Raise when outcome-derived columns are used as model features."""
    leakage = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    leakage.extend(
        sorted(
            column
            for column in set(feature_columns)
            if "future" in column.lower() or "_next_" in column.lower() or column.lower().startswith("next_")
        )
    )
    if leakage:
        raise ValueError(f"Leakage columns cannot be model features: {leakage}")


def available_features_for_df(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return features from `columns` that exist in df, after leakage check."""
    features = [col for col in columns if col in df.columns]
    validate_no_leakage(features)
    return features


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return available intake-time model features for a given DataFrame."""
    features = available_intake_features(
        available_features_for_df(df, INTAKE_TIME_FEATURES)
    )
    validate_no_leakage(features)
    return features


def feature_set_label(feature_columns: list[str] | set[str]) -> str:
    """Return stable feature-set label for model metadata."""
    columns = set(feature_columns)
    if columns & set(CONTEXT_FEATURES):
        return "intake_time_context_v2"
    return "intake_time_v2"
