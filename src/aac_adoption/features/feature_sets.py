"""Feature-set definitions and leakage checks."""

BASE_INTAKE_TIME_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "age_upon_intake",
    "age_days",
    "age_months",
    "age_years",
    "age_group",
    "breed",
    "primary_breed",
    "is_mixed_breed",
    "simplified_breed_group",
    "color",
    "primary_color",
    "simplified_color_group",
    "is_black_or_dark",
    "has_name",
    "is_named",
    "intake_year",
    "intake_month",
    "intake_quarter",
    "intake_season",
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


def feature_set_label(feature_columns: list[str] | set[str]) -> str:
    """Return stable feature-set label for model metadata."""
    columns = set(feature_columns)
    if columns & set(CONTEXT_FEATURES):
        return "intake_time_context_v1"
    return "intake_time_v1"
